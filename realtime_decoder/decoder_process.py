import os
import time
import glob
import numpy as np

from realtime_decoder import (
    base, utils, position, messages, transitions,
    binary_record
)

####################################################################################
# Interfaces
####################################################################################

class DecoderMPISendInterface(base.StandardMPISendInterface):

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    def send_posterior(self, dest, msg):
        self.comm.Send(
            buf=msg.tobytes(),
            dest=dest,
            tag=messages.MPIMessageTag.POSTERIOR
        )

    def send_velocity_position(self, dest, msg):
        self.comm.Send(
            msg.tobytes(),
            dest=dest,
            tag=messages.MPIMessageTag.VEL_POS
        )

    def send_dropped_spikes(self, dest, msg):
        self.comm.Send(
            msg.tobytes(),
            dest=dest,
            tag=messages.MPIMessageTag.DROPPED_SPIKES
        )

class SpikeRecvInterface(base.MPIRecvInterface):

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)
        self._msg_dtype = messages.get_dtype(
            "SpikePosJointProb", config=config
        )
        self._msg_buffer = bytearray(self._msg_dtype.itemsize)
        self._req = self.comm.Irecv(
            buf=self._msg_buffer,
            tag=messages.MPIMessageTag.SPIKE_DECODE_DATA
        )

    def receive(self):
        rdy = self._req.Test()
        if rdy:
            # perform a copy because while we are receiving the next message,
            # the message buffer might be mutated. this would also change the
            # numpy array data
            msg = np.frombuffer(
                self._msg_buffer,
                dtype=self._msg_dtype).copy()
            self._req = self.comm.Irecv(
                buf=self._msg_buffer,
                tag=messages.MPIMessageTag.SPIKE_DECODE_DATA
            )

            return msg

        return None

class LFPTimeInterface(base.MPIRecvInterface):

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)
        self._req = self.comm.irecv(
            source=self.config['rank']['ripples'][0],
            tag=messages.MPIMessageTag.LFP_TIMESTAMP
        )

    def receive(self):

        rdy, msg = self._req.test()
        if rdy:
            self._req = self.comm.irecv(
                source=self.config['rank']['ripples'][0],
                tag=messages.MPIMessageTag.LFP_TIMESTAMP
            )
            return msg

        return None


####################################################################################
# Data handlers/managers
####################################################################################

class ClusterlessDecoder(base.Decoder):

    def __init__(self, rank, config, pos_bin_struct):
        super().__init__()
        self._rank = rank
        self._config = config
        self._pos_bin_struct = pos_bin_struct

        self._position = 0

        num_bins = self._config['encoder']['position']['num_bins']
        self._posterior = utils.normalize_to_probability(np.ones(num_bins))
        self._prev_posterior = self._posterior.copy()
        self._likelihood = self._posterior.copy()

        # be aware that this starts from zero. in order to be correct,
        # we must have self._config['encoder']['position']['lower'] be 0
        self._pos_bins = np.arange(num_bins)
        self._arm_coords = np.array(self._config['encoder']['position']['arm_coords'])
        self._init_transitions()

        if config['preloaded_model']:
            self._load_model()
        else:
            # for initialization, assume non-zero uniform occupancy
            self._occupancy = np.ones(num_bins)
            self._occupancy_ct = 0

        self._firing_rate = {
            elec_grp_id: np.ones(num_bins)
            for elec_grp_id in self._config['decoder_assignment'][self._rank]
        }

        self._dt = (
            self._config['decoder']['time_bin']['samples'] /
            self._config['sampling_rate']['spikes']
        )

        self._init_params()

    def _load_model(self):
        files = glob.glob(
            os.path.join(
                self._config['files']['saved_model_dir'],
                f'*decoder_rank_{self._rank}.occupancy.npz'
            )
        )

        if files == []:
            raise ValueError(
                f"Could not load decoder occupancy successfully!")

        elif len(files) != 1:
            raise ValueError(
                f"Found multiple occupancy files {files} in directory "
                f"{self._config['files']['saved_model_dir']}. "
                "Make sure there is only one."
            )
        else:
            with np.load(files[0]) as f:
                self._occupancy = f['occupancy']
                self._occupancy_ct = f['occupancy_ct'][0]
            self.class_log.info(f"Loaded occupancy from {files[0]}")

    def _init_transitions(self):
        if self._config['algorithm'] == 'clusterless_decoder':
            self._transmat = transitions.sungod_transition_matrix(
                self._pos_bins, self._arm_coords,
                self._config['clusterless_decoder']['transmat_bias']
            )
        elif config['algorithm'] == 'clusterless_classifier':
            pass
        else:
            raise NotImplementedError(
                f"Cannot set up model for algorithm {config['algorithm']}"
            )

    def _init_params(self):

        self.p = {}
        self.p['algorithm'] = self._config['algorithm']
        self.p['num_occupancy_disp'] = self._config['display']['decoder']['occupancy']

    def compute_posterior(self, spike_arr):

        # update firing rates
        if spike_arr.shape[0] > 0:
            for data in spike_arr:
                # extract elec_grp_id, pos
                elec_grp_id = data[1]
                pos = data[2]
                bin_idx = self._pos_bin_struct.get_bin(pos)
                # do not need to apply no-animal condition because
                # it is impossible for spikes to be observed in
                # the gap regions
                self._firing_rate[elec_grp_id][bin_idx] += 1

        # likelihood contribution from no-spike probability
        self._likelihood = np.ones_like(self._occupancy)
        norm_occupancy = self._occupancy / np.nansum(self._occupancy)
        for elec_grp_id, fr in self._firing_rate.items():
            norm_fr = fr / fr.sum()
            lk_no_spike = np.exp(-self._dt * norm_fr / norm_occupancy)
            self._likelihood *= lk_no_spike
            self._likelihood /= np.nansum(self._likelihood)
            self._likelihood[~np.isfinite(self._likelihood)] = 0

        # at this point, there should be no nan's in the arrays

        # likelihood contribution from observed spikes
        if spike_arr.shape[0] > 0:
            for data in spike_arr:
                # can ignore dt factor since we normalize likelihood
                # anyway
                self._likelihood *= data[5:]
                self._likelihood /= self._likelihood.sum()

        self._prev_posterior = self._posterior

        if self.p['algorithm'] == 'clusterless_decoder':
            self._posterior = (
                self._likelihood[None, :] *
                (self._prev_posterior @ self._transmat)
            )
        elif self.p['algorithm'] == 'clusterless_classifier':
            pass

        self._posterior /= self._posterior.sum()

        return self._posterior, self._likelihood

    def add_observation(self):
        raise ValueError(
            "add_observation() should not be called for this "
            "type of decoder"
        )

    def update_position(self, position, update_occupancy:bool):
        self._position = position

        if update_occupancy:

            bin_idx = self._pos_bin_struct.get_bin(self._position)
            self._occupancy[bin_idx] += 1
            utils.apply_no_anim_boundary(
                self._pos_bins, self._arm_coords, self._occupancy, np.nan)

            self._occupancy_ct += 1

            if self._occupancy_ct % self.p['num_occupancy_disp'] == 0:
                print(f"Number of decoder occupancy points: {self._occupancy_ct}")

        return self._occupancy

    def save_occupancy(self):
        filename = os.path.join(
            self._config['files']['output_dir'],
            f"{self._config['files']['prefix']}_" +
            f"decoder_rank_{self._rank}.occupancy.npz"
        )
        np.savez(
            filename,
            occupancy=self._occupancy,
            occupancy_ct=np.atleast_1d(self._occupancy_ct)
        )
        self.class_log.info(f"Saved occupancy to {filename}")

class DecoderManager(base.BinaryRecordBase, base.MessageHandler):

    def __init__(
        self, rank, config, send_interface, spike_interface,
        pos_interface, lfp_interface, pos_mapper
    ):

        if config['algorithm'] == 'clusterless_decoder':
            state_labels = config['clusterless_decoder']['state_labels']
        elif config['algorithm'] == 'clusterless_classifier':
            state_labels = config['clusterless_classifier']['state_labels']

        n_bins = config['encoder']['position']['num_bins']
        dig = len(str(n_bins))
        n_arms = len(config['encoder']['position']['arm_coords'])

        pos_labels = [f'x{v:0{dig}d}_{l}' for l in state_labels for v in range(n_bins)]
        arm_labels = [f'arm{a}' for a in range(n_arms)]
        likelihood_labels = [f'x{v:0{dig}d}' for v in range(n_bins)]
        occupancy_labels = likelihood_labels

        # note: remove wall time! also changed position of arm labels!
        super().__init__(
            rank=rank,
            rec_ids=[
                binary_record.RecordIDs.DECODER_OUTPUT,
                binary_record.RecordIDs.LIKELIHOOD_OUTPUT,
                binary_record.RecordIDs.DECODER_MISSED_SPIKES,
                binary_record.RecordIDs.OCCUPANCY
            ],
            rec_labels=[
                ['bin_timestamp_l', 'bin_timestamp_r', 'velocity', 'mapped_pos',
                'raw_x', 'raw_y', 'raw_x2', 'raw_y2', 'x', 'y',
                'spike_count', 'task_state', 'cred_int_post', 'cred_int_lk',
                'dec_rank', 'dropped_spikes', 'duplicated_spikes', 'vel_thresh',
                'frozen_model'] +
                pos_labels + state_labels,
                ['bin_timestamp_l', 'bin_timestamp_r', 'mapped_pos', 'spike_count', 'dec_rank',
                 'vel_thresh', 'frozen_model'] +
                likelihood_labels,
                ['timestamp', 'elec_grp_id', 'real_bin', 'late_bin'],
                ['timestamp', 'raw_x', 'raw_y', 'raw_x2', 'raw_y2', 'x', 'y',
                 'segment', 'pos_on_seg', 'mapped_pos', 'velocity', 'dec_rank',
                 'vel_thresh', 'frozen_model'] +
                 occupancy_labels
            ],
            rec_formats=[
                'qqddddddddqqqqqqqd?' + 'd'*len(pos_labels) + 'd'*len(state_labels),
                'qqdqqd?' + 'd'*len(likelihood_labels),
                'qiii',
                'qddddddqdddqd?' + 'd'*len(occupancy_labels)
            ],
            send_interface=send_interface,
            manager_label='state'
        )

        self._config = config
        self._spike_interface = spike_interface
        self._pos_interface = pos_interface
        self._lfp_interface = lfp_interface

        if not isinstance(pos_mapper, base.PositionMapper):
            raise TypeError(f"Invalid 'pos_mapper' type {type(pos_mapper)}")
        self._pos_mapper = pos_mapper

        self._kinestimator = position.KinematicsEstimator(
            scale_factor=config['kinematics']['scale_factor'],
            dt=1/config['sampling_rate']['position'],
            xfilter=config['kinematics']['smoothing_filter'],
            yfilter=config['kinematics']['smoothing_filter'],
            speedfilter=config['kinematics']['smoothing_filter'],
        )

        # data messages sent to other processes
        self._posterior_msg = np.zeros(
            (1, ),
            dtype=messages.get_dtype(
                "Posterior", config=config
            )
        )
        self._vel_pos_msg = np.zeros(
            (1, ),
            dtype=messages.get_dtype("VelocityPosition")
        )
        self._dropped_spikes_msg = np.zeros(
            (1, ),
            dtype=messages.get_dtype("DroppedSpikes")
        )

        self._init_decoder()

        # timestamp, elec_grp_id, pos, cred_int, used, histogram
        self._spike_buf = np.zeros(
            (self._config['decoder']['bufsize'], n_bins+5)
        )
        self._sb_ind = 0
        self._dropped_spikes = 0
        self._duplicate_spikes = 0

        self._task_state = 1
        self._save_early = True

        self._spike_msg_ct = 0

        self._pos_ct = 0
        self._pos_timestamp = 0
        self._current_pos = 0 # mapped position
        self._current_vel = 0
        self._raw_x = 0
        self._raw_y = 0
        self._raw_x2 = 0
        self._raw_y2 = 0
        self._x = 0
        self._y = 0

        self._times = {}
        self._times_ind = {}

        self._init_params()
        self._init_timings()
        self._set_up_trodes()

    def next_iter(self):

        spike_msg = self._spike_interface.receive()
        if spike_msg is not None:

            self._process_spike(spike_msg)

        timestamp = self._lfp_interface.receive()
        if timestamp is not None:
            # self.class_log.info(f"Got LFP timestamp {timestamp}")
            self._process_lfp_timestamp(timestamp)

        pos_msg = self._pos_interface.__next__()
        if pos_msg is not None:
            self._process_pos(pos_msg)


    def handle_message(self, msg, mpi_status):

        if isinstance(msg, messages.BinaryRecordCreate):
            self.set_record_writer_from_message(msg)
        elif isinstance(msg, messages.StartRecordMessage):
            self.class_log.info("Starting records")
            self.start_record_writing()
        elif isinstance(msg, messages.ActivateDataStreams):
            self.class_log.info("Activating datastreams")
            self._pos_interface.activate()
        elif isinstance(msg, messages.TerminateSignal):
            rank = mpi_status.source
            self.class_log.info(f"Got terminate signal from rank {rank}")
            raise StopIteration()
        elif isinstance(msg, messages.VerifyStillAlive):
            self.send_interface.send_alive_message()
        elif isinstance(msg, messages.GuiEncodingModelParameters):
            self._update_gui_params(msg)
        else:
            self._class_log.warning(
                f"Received message of type {type(msg)} "
                f"from source: {mpi_status.source}, "
                f" tag: {mpi_status.tag}, ignoring"
            )

    def _update_gui_params(self, gui_msg):
        self.class_log.info("Updating GUI decoder parameters")
        self.p['vel_thresh'] = gui_msg.encoding_velocity_threshold
        self.p['frozen_model'] = gui_msg.freeze_model

    def _init_decoder(self):
        config = self._config
        rank = self.rank

        if config['algorithm'] in ('clusterless_decoder, clusterless_classifier'):
            self._decoder = ClusterlessDecoder(
                rank, config,
                position.PositionBinStruct(
                    config['encoder']['position']['lower'],
                    config['encoder']['position']['upper'],
                    config['encoder']['position']['num_bins']
                )
            )

    def _set_up_trodes(self):
        trodes = self._config['decoder_assignment'][self.rank]
        for trode in trodes:
            self._init_timings(trode=trode)

    def _init_timings(self, *, trode=None):

        if trode is None:
            dt = np.dtype([
                ('decoder_rank', '=i4'),
                ('bin_timestamp_l', '=i8'),
                ('bin_timestamp_r', '=i8'),
                ('t_start_post', '=i8'),
                ('t_end_post', '=i8')
            ])
            self._times['posterior'] = np.zeros(
                self.p['timings_bufsize'],
                dtype=dt
            )
            self._times_ind['posterior'] = 0
        else:
            dt = np.dtype([
                ('elec_grp_id', '=i4'),
                ('timestamp', '=i8'),
                ('t_decoder', '=i8'),
            ])
            self._times[trode] = np.zeros(
                self.p['timings_bufsize'],
                dtype=dt
            )
            self._times_ind[trode] = 0

    def _init_params(self):

        self.p = {}
        self.p['taskstate_file'] = self._config.get('trodes').get('taskstate_file')
        self.p['algorithm'] = self._config['algorithm']
        self.p['preloaded_model'] = self._config['preloaded_model']
        self.p['frozen_model'] = self._config['frozen_model']
        self.p['algorithm'] = self._config['algorithm']
        self.p['kinematics_sf'] = self._config['kinematics']['scale_factor']
        self.p['smooth_x'] = self._config['kinematics']['smooth_x']
        self.p['smooth_y'] = self._config['kinematics']['smooth_y']
        self.p['smooth_speed'] = self._config['kinematics']['smooth_speed']
        self.p['vel_thresh'] = self._config['encoder']['vel_thresh']
        self.p['cred_interval'] = self._config['cred_interval']['val']
        self.p['cred_int_max'] = self._config['cred_interval']['max_num']
        self.p['cred_int_bufsize'] = self._config['decoder']['cred_int_bufsize']
        self.p['timings_bufsize'] = self._config['decoder']['timings_bufsize']
        self.p['num_pos_points'] = self._config['decoder']['num_pos_points']
        self.p['num_spikes_disp'] = self._config['display']['decoder']['total_spikes']
        self.p['tbin_samples'] = self._config['decoder']['time_bin']['samples']
        self.p['tbin_delay_samples'] = self._config['decoder']['time_bin']['delay_samples']
        self.p['first_decoder_rank'] = self._config['rank']['decoders'][0]


    def _process_spike(self, spike_msg):

        self._record_timings(
            spike_msg[0]['elec_grp_id'],
            spike_msg[0]['timestamp'],
            time.time_ns()
        )

        self._spike_buf[self._sb_ind, 0] = spike_msg[0]['timestamp']
        self._spike_buf[self._sb_ind, 1] = spike_msg[0]['elec_grp_id']
        self._spike_buf[self._sb_ind, 2] = spike_msg[0]['current_pos']
        self._spike_buf[self._sb_ind, 3] = spike_msg[0]['cred_int']
        self._spike_buf[self._sb_ind, 4] = 0
        self._spike_buf[self._sb_ind, 5:] = spike_msg[0]['hist']
        self._sb_ind = (self._sb_ind + 1) % self._spike_buf.shape[0]

        self._spike_msg_ct += 1

        # compute the dropped spikes percentage now since we are at the
        # head of the buffer (the index at which the next spike will be
        # stored)
        if self._sb_ind == 0:
            x = self._spike_buf.shape[0] - np.sum(self._spike_buf[:, 4])
            self._dropped_spikes += int(x)
            pct = self._dropped_spikes/self._spike_msg_ct*100
            self.class_log.info(
                f"Dropped spikes: {self._dropped_spikes}, "
                f"Total: {self._spike_msg_ct}, ({pct:.4f} %)"
            )

            self._dropped_spikes_msg[0]['rank'] = self.rank
            self._dropped_spikes_msg[0]['pct'] = pct
            self.send_interface.send_dropped_spikes(
                self._config['rank']['gui'][0],
                self._dropped_spikes_msg
            )

        if self._spike_msg_ct % self.p['num_spikes_disp'] == 0:
            self.class_log.info(f"Received {self._spike_msg_ct} spikes so far")

    def _process_pos(self, pos_msg):

        if pos_msg.timestamp <= self._pos_timestamp:
            self.class_log.warning(
                f"Duplicate or backwards timestamp. New timestamp: {pos_msg.timestamp}, "
                f"Most recent timestamp: {self._pos_timestamp}"
            )
            return

        self._pos_timestamp = pos_msg.timestamp

        if (
            self._pos_ct % self.p['num_pos_points'] == 0 and
            self.p["taskstate_file"] is not None
        ):

            self._task_state = utils.get_last_num(self.p['taskstate_file'])

        # calculate velocity using the midpoints
        xmid = (pos_msg.x + pos_msg.x2)/2
        ymid = (pos_msg.y + pos_msg.y2)/2
        xv, yv, self._current_vel = self._kinestimator.compute_kinematics(
            xmid, ymid,
            smooth_x=self.p['smooth_x'],
            smooth_y=self.p['smooth_y'],
            smooth_speed=self.p['smooth_speed']
        )

        self._x = xv / self.p['kinematics_sf']
        self._y = yv / self.p['kinematics_sf']

        # map position to linear coordinates
        self._current_pos = self._pos_mapper.map_position(pos_msg)

        self._raw_x = pos_msg.x
        self._raw_y = pos_msg.y
        self._raw_x2 = pos_msg.x2
        self._raw_y2 = pos_msg.y2

        #############################################################################################################
        # compute angle
        # No! Do in main process
        #############################################################################################################

        update_occupancy = self._is_training_epoch()
        occupancy = self._decoder.update_position(
            self._current_pos, update_occupancy
        )

        if self._task_state != 1 and self._save_early:
            # we also save decoder at the end of the program,
            # but we do it here as well just to be safe
            self._decoder.save_occupancy()
            self._save_early = False

        # write record
        self.write_record(
            binary_record.RecordIDs.OCCUPANCY,
            pos_msg.timestamp, pos_msg.x, pos_msg.y,
            pos_msg.x2, pos_msg.y2,
            self._x, self._y,
            pos_msg.segment,
            pos_msg.position, self._current_pos,
            self._current_vel, self.rank,
            self.p['vel_thresh'], self.p['frozen_model'],
            *occupancy
        )

        if self.rank == self.p['first_decoder_rank']:
            self._vel_pos_msg[0]['rank'] = self.rank
            self._vel_pos_msg[0]['timestamp'] = pos_msg.timestamp
            self._vel_pos_msg[0]['segment'] = pos_msg.segment
            self._vel_pos_msg[0]['raw_x'] = pos_msg.x
            self._vel_pos_msg[0]['raw_y'] = pos_msg.y
            self._vel_pos_msg[0]['raw_x2'] = pos_msg.x2
            self._vel_pos_msg[0]['raw_y2'] = pos_msg.y2
            self._vel_pos_msg[0]['mapped_pos'] = self._current_pos
            self._vel_pos_msg[0]['velocity'] = self._current_vel
            self.send_interface.send_velocity_position(
                self._config['rank']['supervisor'][0], self._vel_pos_msg
            )

        self._pos_ct += 1

    def _record_timings(self, trode, timestamp, t_decoder):

        ind = self._times_ind[trode]

        # expand timings array if necessary
        if ind == len(self._times[trode]):
            self._times[trode] = np.hstack((
                self._times[trode],
                np.zeros(
                    self.p['timings_bufsize'],
                    dtype=self._times[trode].dtype
                )
            ))

        # write to timings array
        tarr = self._times[trode]
        tarr[ind]['elec_grp_id'] = trode
        tarr[ind]['timestamp'] = timestamp
        tarr[ind]['t_decoder'] = t_decoder
        self._times_ind[trode] += 1

    def _time_posterior(
        self, bin_timestamp_l, bin_timestamp_r,
        t_start_post, t_end_post,
    ):

        ind = self._times_ind['posterior']

        # expand timings array if necessary
        if ind == len(self._times['posterior']):
            self._times['posterior'] = np.hstack((
                self._times['posterior'],
                np.zeros(
                    self.p['timings_bufsize'],
                    dtype=self._times['posterior'].dtype
                )
            ))

        # write to timings array
        tarr = self._times['posterior']
        tarr[ind]['decoder_rank'] = self.rank
        tarr[ind]['bin_timestamp_l'] = bin_timestamp_l
        tarr[ind]['bin_timestamp_r'] = bin_timestamp_r
        tarr[ind]['t_start_post'] = t_start_post
        tarr[ind]['t_end_post'] = t_end_post
        self._times_ind['posterior'] += 1

    def _save_timings(self):

        filename = os.path.join(
            self._config['files']['output_dir'],
            f"{self._config['files']['prefix']}_decoder_rank_{self.rank}." +
            f"{self._config['files']['timing_postfix']}.npz"
        )
        data = self._times['posterior']
        ind = self._times_ind['posterior']
        np.savez(filename, timings=data[:ind])
        self.class_log.info(
            f"Wrote timings file for decoder rank {self.rank} to {filename}"
        )
        self._times.pop('posterior')
        self._times_ind.pop('posterior')

        for trode in self._times:
            filename = os.path.join(
                self._config['files']['output_dir'],
                f"{self._config['files']['prefix']}_decoder_trode_{trode}." +
                f"{self._config['files']['timing_postfix']}.npz"
            )
            data = self._times[trode]
            ind = self._times_ind[trode]
            np.savez(filename, timings=data[:ind])
            self.class_log.info(
                f"Wrote timings file for trode {trode} to {filename}"
            )


    def _is_training_epoch(self):

        res = (
            abs(self._current_vel) >= self.p['vel_thresh'] and
            self._task_state == 1 and
            not self.p['frozen_model']
        )
        return res

    def _process_lfp_timestamp(self, timestamp):

        # these are default values. if there are relevant spikes
        # in the time bin of interest, these will be populated
        # accordingly
        enc_cred_intervals = np.zeros(self.p['cred_int_bufsize'], dtype=int)
        enc_argmaxes = np.zeros(self.p['cred_int_bufsize'], dtype=int)

        lb = int(timestamp - self.p['tbin_delay_samples'] - self.p['tbin_samples'])
        ub = int(timestamp - self.p['tbin_delay_samples'])
        spikes_in_bin_mask = np.logical_and(
            self._spike_buf[:, 0] >= lb,
            self._spike_buf[:, 0] < ub
        )

        if np.sum(spikes_in_bin_mask) > 0:

            # these spikes are being used. mark them with a 1
            self._spike_buf[spikes_in_bin_mask, 4] = 1

            spikes_before = np.atleast_2d(
                self._spike_buf[spikes_in_bin_mask]
            )

            unique_inds = self._get_unique(spikes_before[:, 0])
            spikes_after = np.atleast_2d(
                spikes_before[unique_inds]
            )

            num_before = len(spikes_before)
            num_after = len(spikes_after)
            if num_before != num_after:
                self._duplicate_spikes += (num_before - num_after)

            if num_after > 0:
                # Question: do we want credible interval array for only one time bin
                # or persistent?

                # check credible interval for each spike, if good add elec_grp_id to list.
                # main process will check for non-nan elements
                order = np.argsort(spikes_after[:, 0])
                ordered_spikes = spikes_after[order]
                for ii, data in enumerate(ordered_spikes):
                    if data[3] <= self.p['cred_int_max']:
                        enc_cred_intervals[ii % self.p['cred_int_bufsize']] = data[1]
                        enc_argmaxes[ii % self.p['cred_int_bufsize']] = np.argmax(data[5:])

            # Note: the decoder can automatically handle the no-spike case
            spikes_in_bin_count = num_after
            t0 = time.time_ns()
            posterior, likelihood = self._decoder.compute_posterior(
                spikes_after
            )
            t1 = time.time_ns()
            self._time_posterior(lb, ub, t0, t1)
        else:
            # no spikes in time bin. however the decoder can automatically
            # handle this case
            spikes_in_bin_count = 0
            t0 = time.time_ns()
            posterior, likelihood = self._decoder.compute_posterior(
                np.atleast_2d(self._spike_buf[spikes_in_bin_mask])
            )
            t1 = time.time_ns()
            self._time_posterior(lb, ub, t0, t1)

        # new method: rather than computing posterior replay target,
        # base, etc. here, have the stimulation decider do this.
        # that is where most of the customized stuff should go

        cred_int_post, cred_int_lk = self._compute_credible_interval(
            posterior, likelihood
        )

        # populate message data and send to main and gui
        self._posterior_msg[0]['rank'] = self.rank
        self._posterior_msg[0]['lfp_timestamp'] = timestamp
        self._posterior_msg[0]['bin_timestamp_l'] = lb
        self._posterior_msg[0]['bin_timestamp_r'] = ub
        self._posterior_msg[0]['posterior'] = posterior
        self._posterior_msg[0]['likelihood'] = likelihood
        self._posterior_msg[0]['velocity'] = self._current_vel
        self._posterior_msg[0]['cred_int_post'] = cred_int_post
        self._posterior_msg[0]['cred_int_lk'] = cred_int_lk
        self._posterior_msg[0]['enc_cred_intervals'] = enc_cred_intervals
        self._posterior_msg[0]['enc_argmaxes'] = enc_argmaxes
        self._posterior_msg[0]['spike_count'] = spikes_in_bin_count
        self.send_interface.send_posterior(
            self._config['rank']['supervisor'][0], self._posterior_msg
        )
        self.send_interface.send_posterior(
            self._config['rank']['gui'][0], self._posterior_msg
        )

        if self.p['algorithm'] == 'clusterless_classifier':
            state_prob = posterior.sum(axis=1)
        else:
            state_prob = np.ones(1)

        # write all records
        self.write_record(
            binary_record.RecordIDs.LIKELIHOOD_OUTPUT,
            lb, ub, self._current_pos, spikes_in_bin_count,
            self.rank, self.p['vel_thresh'], self.p['frozen_model'],
            *likelihood
        )

        self.write_record(
            binary_record.RecordIDs.DECODER_OUTPUT,
            lb, ub, self._current_vel, self._current_pos,
            self._raw_x, self._raw_y,
            self._raw_x2, self._raw_y2, self._x, self._y,
            spikes_in_bin_count, self._task_state,
            cred_int_post, cred_int_lk, self.rank,
            self._dropped_spikes, self._duplicate_spikes,
            self.p['vel_thresh'], self.p['frozen_model'],
            *posterior.flatten(), *state_prob
        )

    def _get_unique(self, spike_times):
        # remove duplicates and return view of array
        _, inds, counts = np.unique(
            spike_times, return_index=True, return_counts=True
        )
        unique_inds = np.atleast_1d(
            inds[np.argwhere(counts == 1).squeeze()]
        )
        return unique_inds

    def _compute_credible_interval(self, posterior, likelihood):
        # compute credible interval

        post = posterior.sum(axis=0)
        cs_post = np.cumsum(np.sort(post)[::-1])
        cred_int_post = np.searchsorted(cs_post, self.p['cred_interval']) + 1

        cs_lk = np.cumsum(np.sort(likelihood)[::-1])
        cred_int_lk = np.searchsorted(cs_lk, self.p['cred_interval']) + 1

        return cred_int_post, cred_int_lk

    def finalize(self):
        self._save_timings()
        self._pos_interface.deactivate()
        self._decoder.save_occupancy()
        self.stop_record_writing()

####################################################################################
# Processes
####################################################################################

class DecoderProcess(base.RealtimeProcess):

    def __init__(self, comm, rank, config, pos_interface, pos_mapper):
        super().__init__(comm, rank, config)

        try:
            self._decoder_manager = DecoderManager(
                rank, config, DecoderMPISendInterface(comm, rank, config),
                SpikeRecvInterface(comm, rank, config), pos_interface,
                LFPTimeInterface(comm, rank, config), pos_mapper
            )
        except:
            self.class_log.exception("Exception in init!")

        self._mpi_recv = base.StandardMPIRecvInterface(
            comm, rank, config, messages.MPIMessageTag.COMMAND_MESSAGE,
            self._decoder_manager
        )

        self._gui_recv = base.StandardMPIRecvInterface(
            comm, rank, config, messages.MPIMessageTag.GUI_PARAMETERS,
            self._decoder_manager
        )

    def main_loop(self):

        try:
            self._decoder_manager.setup_mpi()
            while True:
                self._mpi_recv.receive()
                self._gui_recv.receive()
                self._decoder_manager.next_iter()

        except StopIteration:
            self.class_log.info("Exiting normally")
        except Exception as e:
            self.class_log.exception(
                "Decoder process exception occurred!"
            )

        self._decoder_manager.finalize()
        self.class_log.info("Exited main loop")

