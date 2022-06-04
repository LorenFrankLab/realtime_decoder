import os
import glob
import time
import fcntl
import numpy as np

from mpi4py import MPI
from typing import Sequence, List

from realtime_decoder import (
    base, utils, position, datatypes, messages, binary_record
)

####################################################################################
# Data classes
####################################################################################

class EncoderJointProbEstimate(object):

    def __init__(self, nearby_spikes, weights, positions, hist):
        self.nearby_spikes = nearby_spikes
        self.weights = weights
        self.positions = positions
        self.hist = hist

####################################################################################
# Interfaces
####################################################################################

class EncoderMPISendInterface(base.StandardMPISendInterface):

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    def send_joint_prob(self, dest, msg):
        self.comm.Send(
            buf=msg.tobytes(),
            dest=dest,
            tag=messages.MPIMessageTag.SPIKE_DECODE_DATA
        )

####################################################################################
# Data handlers/managers
####################################################################################

class Encoder(base.LoggingClass):
    """Note: this class only handles 1D position currently
    """

    def __init__(self, config, trode, pos_bin_struct):

        super().__init__()
        self._config = config
        self._trode = trode
        self._pos_bin_struct = pos_bin_struct

        sigma = self._config['encoder']['mark_kernel']['std']
        self._k1 = 1 / (np.sqrt(2*np.pi) * sigma)
        self._k2 = -0.5 / (sigma**2)

        self._position = 0

        # be aware that this starts from zero. in order to be correct,
        # we must have self._config['encoder']['position']['lower'] be 0
        self._pos_bins = np.arange(
            self._config['encoder']['position']['num_bins']
        )

        self._arm_coords = np.array(
            self._config['encoder']['position']['arm_coords']
        )

        if config['preloaded_model']:
            self._load_model()
        else:
            N = self._config['encoder']['bufsize']
            dim = self._config['encoder']['mark_dim']
            self._marks = np.zeros((N, dim), dtype='<f8')
            self._positions = np.zeros(N, dtype='<f4')
            self._mark_idx = 0
            self._occupancy = np.zeros(self._config['encoder']['position']['num_bins'])
            self._occupancy_ct = 0

        self._init_params()

    def _load_model(self):
        files = glob.glob(
            os.path.join(
                self._config['files']['saved_model_dir'],
                f'*trode_{self._trode}.encoder.npz'
            )
        )
        if files == []:
            raise ValueError(
                f"Could not load encoding model successfully!")

        elif len(files) != 1:
            raise ValueError(
                "Found multiple encoders in directory "
                f"{self._config['files']['saved_model_dir']}. "
                "Make sure there is only one."
            )
        else:
            with np.load(files[0]) as f:
                self._marks = f['marks']
                self._positions = f['positions']
                self._mark_idx = f['mark_idx'][0]
                self._occupancy = f['occupancy']
                self._occupancy_ct = f['occupancy_ct'][0]
            self.class_log.info(f"Loaded encoding model from {files[0]}")

    def _init_params(self):
        self.p = {}
        self.p['mark_dim'] = self._config['encoder']['mark_dim']
        self.p['use_filter'] = self._config['encoder']['mark_kernel']['use_filter']
        self.p['filter_std'] = self._config['encoder']['mark_kernel']['std']
        self.p['filter_n_std'] = self._config['encoder']['mark_kernel']['n_std']
        self.p['n_marks_min'] = self._config['encoder']['mark_kernel']['n_marks_min']
        self.p['num_occupancy_points'] = self._config['display']['encoder']['occupancy']

    def add_new_mark(self, mark):
        if self._mark_idx == self._marks.shape[0]:
            self._marks = np.vstack((
                self._marks,
                np.zeros_like(self._marks)
            ))
            self._positions = np.hstack((
                self._positions,
                np.zeros_like(self._positions)
            ))

        self._marks[self._mark_idx] = mark
        self._positions[self._mark_idx] = self._position
        self._mark_idx += 1

    def get_joint_prob(self, mark):

        # on the very first spike, there are no marks with which to evaluate
        # the kernel. therefore, return immediately
        if self._mark_idx == 0:
            return None

        in_range = np.ones(self._mark_idx, dtype=bool)
        if self.p['use_filter']:
            std = self.p['filter_std']
            n_std = self.p['filter_n_std']
            for ii in range(self._marks.shape[1]):
                in_range = np.logical_and(
                    np.logical_and(
                        self._marks[:self._mark_idx, ii] > mark[ii] - n_std * std,
                        self._marks[:self._mark_idx, ii] < mark[ii] + n_std * std
                    ),
                    in_range
                )

            # not enough spikes within n-cube
            if np.sum(in_range) < self.p['n_marks_min']:
                return None

        # evaluate Gaussian kernel on distance in mark space
        squared_distance = np.sum(
            np.square(self._marks[:self._mark_idx] - mark),
            axis=1
        )
        weights = self._k1 * np.exp(squared_distance * self._k2)
        positions = self._positions[:self._mark_idx]

        # print(positions.shape)
        # print("")
        # print(self._pos_bin_struct.pos_bin_edges)
        # print("")
        # print(weights)

        hist, hist_edges = np.histogram(
            a=positions,
            bins=self._pos_bin_struct.pos_bin_edges,
            weights=weights, normed=False
        )

        hist += 0.0000001

        # normalize by occupancy
        hist /= (self._occupancy/np.nansum(self._occupancy))
        hist[~np.isfinite(hist)] = 0.0

        # note: if pos_bin_delta is not one, this will not sum to 1
        hist /= (np.sum(hist) * self._pos_bin_struct.pos_bin_delta)

        # print("")
        # print(hist)
        # print("")

        return EncoderJointProbEstimate(
            np.sum(in_range), weights, positions, hist
        )

    def update_position(self, position, update_occupancy:bool):
        self._position = position

        if update_occupancy:

            bin_idx = self._pos_bin_struct.get_bin(self._position)
            self._occupancy[bin_idx] += 1
            utils.apply_no_anim_boundary(
                self._pos_bins, self._arm_coords, self._occupancy, np.nan)

            self._occupancy_ct += 1

            if self._occupancy_ct % self.p['num_occupancy_points'] == 0:
                print(f"Number of encoder occupancy points: {self._occupancy_ct}")

    def save(self):
        filename = os.path.join(
            self._config['files']['output_dir'],
            f"{self._config['files']['prefix']}_" +
            f"trode_{self._trode}.encoder.npz"
        )
        np.savez(
            filename,
            marks=self._marks,
            positions=self._positions,
            mark_idx=np.atleast_1d(self._mark_idx),
            occupancy=self._occupancy,
            occupancy_ct=np.atleast_1d(self._occupancy_ct)
        )
        self.class_log.info(f"Saved encoding model to {filename}")

class EncoderManager(base.BinaryRecordBase, base.MessageHandler):

    def __init__(self, rank, config, send_interface, spikes_interface,
        pos_interface, pos_mapper
    ):

        n_bins = config['encoder']['position']['num_bins']
        dig = len(str(n_bins))

        n_mark_dims = config['encoder']['mark_dim']

        super().__init__(
            rank=rank,
            rec_ids=[
                binary_record.RecordIDs.ENCODER_QUERY,
                binary_record.RecordIDs.ENCODER_OUTPUT,
                #############################################################################################################################
                # Only for testing, remove when finalized
                binary_record.RecordIDs.POS_INFO
                #############################################################################################################################
            ],
            rec_labels=[
                ['timestamp',
                'elec_grp_id',
                'weight',
                'position'],
                ['timestamp', 'elec_grp_id','position', 'velocity',
                'encode_spike', 'cred_int', 'decoder_rank',
                'nearby_spikes', 'sent_to_decoder',
                'vel_thresh', 'frozen_model', 'task_state'] +
                [f'mark_dim_{dim}' for dim in range(n_mark_dims)] +
                [f'x{v:0{dig}d}' for v in range(n_bins)],
                ['timestamp', 'x', 'y', 'x2', 'y2', 'segment', 'position', 'smooth_x', 'smooth_y', 'vel', 'mapped_pos']
            ],
            rec_formats=[
                'qidd',
                'qidd?qqq?d?i'+'d'*n_mark_dims+'d'*n_bins,
                'qddddiddddd'
            ],
            send_interface=send_interface,
            manager_label='state'
        )

        self._config = config

        self._spikes_interface = spikes_interface
        self._pos_interface = pos_interface

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

        self._spike_msg = np.zeros(
            (1, ),
            dtype=messages.get_dtype(
                "SpikePosJointProb", config=config
            )
        )

        # key for these dictionaries is elec_grp_id
        self._spk_counters = {}
        self._encoders = {}
        self._dead_channels = {}
        self._decoder_map = {} # map elec grp id to decoder rank
        self._times = {}
        self._times_ind = {}

        self._task_state = 1
        self._save_early = True

        self._pos_counter = 0
        self._current_pos = 0
        self._current_vel = 0
        self._pos_timestamp = -1

        self._init_params()

    def handle_message(self, msg, mpi_status):

        if isinstance(msg, messages.TrodeSelection):
            self._set_up_trodes(msg.trodes)
        elif isinstance(msg, messages.BinaryRecordCreate):
            self.set_record_writer_from_message(msg)
        elif isinstance(msg, messages.StartRecordMessage):
            self.class_log.info("Starting records")
            self.start_record_writing()
        elif isinstance(msg, messages.ActivateDataStreams):
            self.class_log.info("Activating datastreams")
            self._spikes_interface.activate()
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

    def next_iter(self):

        spike_msg = self._spikes_interface.__next__()
        if spike_msg is not None:
            self._process_spike(spike_msg)

        pos_msg = self._pos_interface.__next__()
        if pos_msg is not None:
            self._process_pos(pos_msg)

    def _init_params(self):

        self.p = {}
        self.p['taskstate_file'] = self._config.get('trodes').get('taskstate_file')
        self.p['num_bins'] = self._config['encoder']['position']['num_bins']
        self.p['spk_amp'] = self._config['encoder']['spk_amp']
        self.p['preloaded_model'] = self._config['preloaded_model']
        self.p['frozen_model'] = self._config['frozen_model']
        self.p['smooth_x'] = self._config['kinematics']['smooth_x']
        self.p['smooth_y'] = self._config['kinematics']['smooth_y']
        self.p['smooth_speed'] = self._config['kinematics']['smooth_speed']
        self.p['vel_thresh'] = self._config['encoder']['vel_thresh']
        self.p['cred_interval'] = self._config['cred_interval']['val']
        self.p['timings_bufsize'] = self._config['encoder']['timings_bufsize']
        self.p['num_encoding_disp'] = self._config['display']['encoder']['encoding_spikes']
        self.p['num_total_disp'] = self._config['display']['encoder']['total_spikes']
        self.p['num_pos_disp'] = self._config['display']['encoder']['position']
        self.p['num_pos_points'] = self._config['encoder']['num_pos_points']

    def _update_gui_params(self, gui_msg):
        self.class_log.info("Updating GUI encoder parameters")
        self.p['vel_thresh'] = gui_msg.encoding_velocity_threshold
        self.p['frozen_model'] = gui_msg.freeze_model

    def _init_timings(self, trode):
        dt = np.dtype([
            ('elec_grp_id', '=i4'),
            ('timestamp', '=i8'),
            ('t_send_data', '=i8'),
            ('t_recv_data', '=i8'),
            ('t_start_kde', '=i8'),
            ('t_end_kde', '=i8'),
            ('t_start_enc_send', '=i8'),
            ('t_end_enc_send', '=i8')
        ])
        self._times[trode] = np.zeros(
            self.p['timings_bufsize'],
            dtype=dt
        )
        self._times_ind[trode] = 0

    def _process_spike(self, spike_msg):

        spike_timestamp = spike_msg.timestamp
        elec_grp_id = spike_msg.elec_grp_id

        # zero out dead channels
        if elec_grp_id in self._dead_channels:
            dch  = self._dead_channels[elec_grp_id]
            spike_msg.data[dch] = 0 # mutates data

        mark_vec = self._compute_mark(spike_msg)

        if max(mark_vec) > self.p['spk_amp']:

            t_start_kde = time.time_ns()
            joint_prob_obj = self._encoders[elec_grp_id].get_joint_prob(
                mark_vec
            )
            t_end_kde = time.time_ns()

            # determine if encoding spike
            encoding_spike = self._is_training_epoch()

            # determine decoder
            decoder_rank = self._decoder_map[elec_grp_id]

            if joint_prob_obj is not None:

                # compute credible interval
                spxx = np.sort(joint_prob_obj.hist)[::-1]
                cred_int = np.searchsorted(np.cumsum(spxx), self.p['cred_interval']) + 1

                # send decoded spike message
                self._spike_msg[0]['timestamp'] = spike_timestamp
                self._spike_msg[0]['elec_grp_id'] = elec_grp_id
                self._spike_msg[0]['current_pos'] = self._current_pos
                self._spike_msg[0]['cred_int'] = cred_int
                self._spike_msg[0]['hist'] = joint_prob_obj.hist
                self._spike_msg[0]['send_time'] = time.time_ns()
                t_start_enc_send = self._spike_msg['send_time']
                self.send_interface.send_joint_prob(decoder_rank, self._spike_msg)
                t_end_enc_send = time.time_ns()

                self._record_timings(
                    elec_grp_id, spike_timestamp,
                    spike_msg.t_send_data, spike_msg.t_recv_data,
                    t_start_kde, t_end_kde,
                    t_start_enc_send, t_end_enc_send
                )

                # record result
                self.write_record(
                    binary_record.RecordIDs.ENCODER_OUTPUT,
                    spike_timestamp, elec_grp_id,
                    self._current_pos, self._current_vel,
                    encoding_spike, cred_int,
                    decoder_rank, True,
                    self.p['vel_thresh'], self.p['frozen_model'],
                    self._task_state,
                    joint_prob_obj.nearby_spikes,
                    *mark_vec, *joint_prob_obj.hist
                )

            # either first spike or not enough neighboring spikes
            # (assuming filter is on). still record result
            else:
                self.write_record(
                    binary_record.RecordIDs.ENCODER_OUTPUT,
                    spike_timestamp, elec_grp_id,
                    self._current_pos, self._current_vel,
                    encoding_spike, -1, # since didn't compute credible interval
                    decoder_rank, False,
                    self.p['vel_thresh'], self.p['frozen_model'],
                    self._task_state, -1,
                    *mark_vec, *np.zeros(self.p['num_bins'])
                )

            # now that we've estimated the spike/pos joint probability,
            # we need to decide whether to add it to the encoding model
            # or not
            if encoding_spike:
                self._encoders[elec_grp_id].add_new_mark(mark_vec)
                self._spk_counters[elec_grp_id]['encoding'] += 1
                if self._spk_counters[elec_grp_id]['encoding'] % self.p['num_encoding_disp'] == 0:
                    self.class_log.info(
                        f"Added {self._spk_counters[elec_grp_id]['encoding']} "
                        "spikes to encoding model"
                    )

        self._spk_counters[elec_grp_id]['total'] += 1
        if self._spk_counters[elec_grp_id]['total'] % self.p['num_total_disp'] == 0:
            self.class_log.info(
                f"Received {self._spk_counters[elec_grp_id]['total']} "
                f"total spikes from ntrode {elec_grp_id}"
            )

    def _process_pos(self, pos_msg):

        if pos_msg.timestamp <= self._pos_timestamp:
            self.class_log.warning(
                f"Duplicate or backwards timestamp. New timestamp: {pos_msg.timestamp}, "
                f"Most recent timestamp: {self._pos_timestamp}"
            )
            return

        self._pos_timestamp = pos_msg.timestamp

        if (
            self._pos_counter % self.p['num_pos_points'] == 0 and
            self.p["taskstate_file"] is not None
        ):

            self._task_state = utils.get_last_num(self.p['taskstate_file'])

        #################################################################################################################
        # debugging, remove when done
        if pos_msg.x == 0:
            self.class_log.info(f"{pos_msg.timestamp} got a 0 xloc, {pos_msg.x}, {pos_msg.y}, {pos_msg.x2}, {pos_msg.y2}")
        ##################################################################################################################

        # calculate velocity using the midpoints
        xmid = (pos_msg.x + pos_msg.x2)/2
        ymid = (pos_msg.y + pos_msg.y2)/2
        # we don't care about x and y returned by compute_kinematics(),
        # as we are using the position mapper to get the appropriate
        # linear coordinates
        _1, _2, self._current_vel = self._kinestimator.compute_kinematics(
            xmid, ymid,
            smooth_x=self.p['smooth_x'],
            smooth_y=self.p['smooth_y'],
            smooth_speed=self.p['smooth_speed']
        )

        # map position to linear coordinates
        self._current_pos = self._pos_mapper.map_position(pos_msg)

        #####################################################################################################
        # For testing, remove when finalized
        # self.write_record(
        #     binary_record.RecordIDs.POS_INFO, pos_msg.timestamp,
        #     pos_msg.x, pos_msg.y, pos_msg.x2, pos_msg.y2,
        #     pos_msg.segment, pos_msg.position, _1, _2,
        #     self._current_vel, self._current_pos
        # )
        # self.class_log.info(f"{pos_msg.timestamp/30000}, {pos_msg.x}, {pos_msg.y}, {pos_msg.y}, {pos_msg.y2}")
        #####################################################################################################

        update_occupancy = self._is_training_epoch()
        for encoder in self._encoders.values():
            encoder.update_position(self._current_pos, update_occupancy)
            if self._task_state != 1 and self._save_early:
                # we also save encoder models at the end of the program,
                # but we do it here as well just to be safe
                encoder.save()
                self._save_early = False

        self._pos_counter += 1
        if self._pos_counter % self.p['num_pos_disp'] == 0:
            self.class_log.debug(f"Received {self._pos_counter} pos points")

    def _compute_mark(self, datapoint):
        spike_data = np.atleast_2d(datapoint.data)
        channel_peaks = np.max(spike_data, axis=1)
        peak_channel_ind = np.argmax(channel_peaks)
        t_ind = np.argmax(spike_data[peak_channel_ind])
        amp_mark = spike_data[:, t_ind]

        return amp_mark

    def _is_training_epoch(self):

        res = (
            abs(self._current_vel) >= self.p['vel_thresh'] and
            self._task_state == 1 and
            not self.p['frozen_model']
        )
        return res

    def _record_timings(
        self, trode, timestamp,
        t_send_data, t_recv_data,
        t_start_kde, t_end_kde,
        t_start_enc_send, t_end_enc_send
    ):

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
        tarr[ind]['t_send_data'] = t_send_data
        tarr[ind]['t_recv_data'] = t_recv_data
        tarr[ind]['t_start_kde'] = t_start_kde
        tarr[ind]['t_end_kde'] = t_end_kde
        tarr[ind]['t_start_enc_send'] = t_start_enc_send
        tarr[ind]['t_end_enc_send'] = t_end_enc_send
        self._times_ind[trode] += 1

    def _save_timings(self):
        for trode in self._times:
            filename = os.path.join(
                self._config['files']['output_dir'],
                f"{self._config['files']['prefix']}_encoder_trode_{trode}." +
                f"{self._config['files']['timing_postfix']}.npz"
            )
            data = self._times[trode]
            ind = self._times_ind[trode]
            np.savez(filename, timings=data[:ind])
            self.class_log.info(
                f"Wrote timings file for trode {trode} to {filename}")

    def _set_up_trodes(self, trodes:List[int]):

        for trode in trodes:
            self._spikes_interface.register_datatype_channel(trode)

            self._encoders[trode] = Encoder(
                self._config,
                trode,
                position.PositionBinStruct(
                    self._config['encoder']['position']['lower'],
                    self._config['encoder']['position']['upper'],
                    self._config['encoder']['position']['num_bins']
                )
            )

            self._spk_counters[trode] = {}
            self._spk_counters[trode]['total'] = 0
            self._spk_counters[trode]['encoding'] = 0

            try:
                dch = self._config['encoder']['dead_channels'][trode]
                self._dead_channels[trode] = dch
                self.class_log.info(f"Set dead channels for trode {trode}")
            except KeyError:
                pass

            for dec_rank, dec_trodes in self._config['decoder_assignment'].items():
                if trode in dec_trodes:
                    self._decoder_map[trode] = dec_rank

            self._init_timings(trode)

    def finalize(self):
        for key in self._spk_counters:
            self.class_log.info(
                f"Got {self._spk_counters[key]} spikes for electrode "
                f"group {key}"
            )
            self._encoders[key].save()
        self._save_timings()
        self._spikes_interface.deactivate()
        self._pos_interface.deactivate()
        self.stop_record_writing()

####################################################################################
# Processes
####################################################################################

class EncoderProcess(base.RealtimeProcess):

    def __init__(
        self, comm, rank, config, spikes_interface, pos_interface, pos_mapper
    ):
        super().__init__(comm, rank, config)

        try:
            self._encoder_manager = EncoderManager(
                rank, config, EncoderMPISendInterface(comm, rank, config),
                spikes_interface, pos_interface, pos_mapper
            )
        except:
            self.class_log.exception("Exception in init!")

        self._mpi_recv = base.StandardMPIRecvInterface(
            comm, rank, config, messages.MPIMessageTag.COMMAND_MESSAGE,
            self._encoder_manager
        )

        self._gui_recv = base.StandardMPIRecvInterface(
            comm, rank, config, messages.MPIMessageTag.GUI_PARAMETERS,
            self._encoder_manager
        )

    def main_loop(self):

        try:
            self._encoder_manager.setup_mpi()
            while True:
                self._mpi_recv.receive()
                self._gui_recv.receive()
                self._encoder_manager.next_iter()

        except StopIteration as ex:
            self.class_log.info("Exiting normally")
        except Exception as e:
            self.class_log.exception(
                "Encoder process exception occurred!"
            )

        self._encoder_manager.finalize()
        self.class_log.info("Exited main loop")
