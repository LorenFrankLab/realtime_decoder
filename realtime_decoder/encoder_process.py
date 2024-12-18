import os
import glob
import time
import fcntl
import numpy as np

from mpi4py import MPI
from typing import Sequence, List

from realtime_decoder import (
    base, utils, position, datatypes, messages, binary_record, taskstate
)

####################################################################################
# Data classes
####################################################################################

class EncoderJointProbEstimate(object):
    """Data object containing infomration about joint probability
    over marks and position"""

    def __init__(self, nearby_spikes, weights, positions, hist):
        self.nearby_spikes = nearby_spikes
        self.weights = weights
        self.positions = positions
        self.hist = hist

####################################################################################
# Interfaces
####################################################################################

class EncoderMPISendInterface(base.StandardMPISendInterface):
    """Sending interface object for encoder_process"""

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    def send_joint_prob(self, dest, msg):
        """Send mark-position joint probability data"""

        self.comm.Send(
            buf=msg.tobytes(),
            dest=dest,
            tag=messages.MPIMessageTag.SPIKE_DECODE_DATA
        )

####################################################################################
# Data handlers/managers
####################################################################################

class Encoder(base.LoggingClass):
    """Represents an encoding model. Note: this class only handles
    1D position currently"""

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
            self._temp_idx = 0 # NOTE(DS): so that mark_idx does not increase but still write down in the mark vec

        self._init_params()
    def _load_model(self):

        fname = os.path.join(
                self._config['files']['saved_model_dir'],
                f"{self._config['files']['saved_model_prefix']}*trode_{self._trode}.encoder.npz"
            )
        print(f"encoder model fname: {fname}")
        
        files = glob.glob(fname)

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

        self._temp_idx = 0 

    def _init_params(self):
        """Initialize parameters used for the encoding model"""

        self.p = {}
        self.p['mark_dim'] = self._config['encoder']['mark_dim']
        self.p['use_channel_dist_from_max_amp'] = self._config['encoder']['use_channel_dist_from_max_amp']
        self.p['use_filter'] = self._config['encoder']['mark_kernel']['use_filter']
        self.p['filter_std'] = self._config['encoder']['mark_kernel']['std']
        self.p['filter_n_std'] = self._config['encoder']['mark_kernel']['n_std']
        self.p['n_marks_min'] = self._config['encoder']['mark_kernel']['n_marks_min']
        self.p['num_occupancy_points'] = self._config['display']['encoder']['occupancy']

    def add_new_mark(self, mark):
        '''
        # this is where the mark_size increases over time 
        self._marks[self._mark_idx%self._marks.shape[0]] = mark
        self._positions[self._mark_idx%self._marks.shape[0]] = self._position
        self._mark_idx += 1
        '''
        
        # NOTE(DS): Having only the most recent spikes bias the encoding 
        if self._mark_idx < self._marks.shape[0]:
            self._marks[self._mark_idx%self._marks.shape[0]] = mark
            self._positions[self._mark_idx%self._marks.shape[0]] = self._position
            self._mark_idx += 1
        else:
            if self._mark_idx%2 == 0:
                self._marks[self._mark_idx%self._marks.shape[0]] = mark
                self._positions[self._mark_idx%self._marks.shape[0]] = self._position
            self._mark_idx += 3

        '''
        if self._mark_idx < self._marks.shape[0]:
            self._marks[self._mark_idx] = mark
            self._positions[self._mark_idx] = self._position
            self._mark_idx += 1

        else:
            self._marks[self._temp_idx%self._marks.shape[0]] = mark
            self._positions[self._temp_idx%self._marks.shape[0]] = self._position
            self._temp_idx += 2
            if self._temp_idx%2000 == 0:
                self.class_log.info(
                f"mark buffer is full. substitutes every other markvec {self._temp_idx/2}"
                )
        '''

        ''' # NOTE(DS): This make buf_size meaningless
        self._marks = np.vstack((
            self._marks,
            np.zeros_like(self._marks)
        ))
        self._positions = np.hstack((
            self._positions,
            np.zeros_like(self._positions)
        ))
        '''





    def get_joint_prob(self, mark):
        """Get a estimate of the joint mark-position probability,
        given an observed mark"""

        # on the very first spike, there are no marks with which to evaluate
        # the kernel. therefore, return immediately
        if self._mark_idx == 0:
            return None

        if self._mark_idx >= self._marks.shape[0]:
            mark_idx = self._marks.shape[0]
        else:
            mark_idx = self._mark_idx


        #print(mark)

        in_range = np.ones(mark_idx, dtype=bool)
        if self.p['use_filter']:
            std = self.p['filter_std']
            n_std = self.p['filter_n_std']
            for ii in range(self._marks.shape[1]):
                in_range = np.logical_and(
                    np.logical_and(
                        self._marks[:mark_idx, ii] > mark[ii] - n_std * std,
                        self._marks[:mark_idx, ii] < mark[ii] + n_std * std
                    ),
                    in_range
                )

            # not enough spikes within n-cube
            if np.sum(in_range) < self.p['n_marks_min']:
                return None

        # evaluate Gaussian kernel on distance in mark space
        squared_distance = np.sum(
            np.square(self._marks[:mark_idx] - mark),
            axis=1
        )
        weights = self._k1 * np.exp(squared_distance * self._k2)
        positions = self._positions[:mark_idx]

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
        """Update the current position of the encoding model"""

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
        """Save the encoding model to disk"""

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
    """Manager class that handles MPI messsages and delegates training
    of the encoding model, among other functions"""

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
        self._task_state_handler = taskstate.TaskStateHandler(self._config)
        self._save_early = True

        self._pos_counter = 0
        self._current_pos = 0
        self._current_vel = 0
        self._pos_timestamp = -1

        self._init_params()

    def handle_message(self, msg, mpi_status):
        """Process a (non neural data) received MPI message"""

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
        """Run one iteration processing any available neural data"""

        spike_msg = self._spikes_interface.__next__()
        if spike_msg is not None:
            self._process_spike(spike_msg)

        pos_msg = self._pos_interface.__next__()
        if pos_msg is not None:
            self._process_pos(pos_msg)

    def _init_params(self):
        """Initialize parameters used by this object"""

        self.p = {}
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
        self.p['use_channel_dist_from_max_amp'] = self._config['encoder']['use_channel_dist_from_max_amp']

    def _update_gui_params(self, gui_msg):
        """Update parameters that can be changed by the GUI"""

        self.class_log.info("Updating GUI encoder parameters")
        self.p['vel_thresh'] = gui_msg.encoding_velocity_threshold
        self.p['frozen_model'] = gui_msg.freeze_model

    def _init_timings(self, trode):
        """Initialize objects that are used for keeping track of
        timing information"""

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
        """Process a spike event"""

        spike_timestamp = spike_msg.timestamp
        elec_grp_id = spike_msg.elec_grp_id

        # zero out dead channels
        if elec_grp_id in self._dead_channels:
            dch  = self._dead_channels[elec_grp_id]
            spike_msg.data[dch] = 0 # mutates data

        mark_vec = self._compute_mark(spike_msg)


        #print('this is mark vec')
        #print(mark_vec)
        #print(mark_vec.shape) # NOTE(DS): for debug

        
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
                if self._encoders[elec_grp_id]._mark_idx%1000 == 0:
                    self.class_log.info(
                        f"num spikes in {elec_grp_id} is {self._encoders[elec_grp_id]._mark_idx}"
                        )
                self._spk_counters[elec_grp_id]['encoding'] += 1
                if self._spk_counters[elec_grp_id]['encoding'] % self.p['num_encoding_disp'] == 0:
                    self.class_log.info(
                        f"Added {self._spk_counters[elec_grp_id]['encoding']} "
                        f"spikes to encoding model of nTrode {elec_grp_id}"
                    )

        self._spk_counters[elec_grp_id]['total'] += 1
        if self._spk_counters[elec_grp_id]['total'] % self.p['num_total_disp'] == 0:
            self.class_log.info(
                f"Received {self._spk_counters[elec_grp_id]['total']} "
                f"total spikes from ntrode {elec_grp_id}"
            )

    def _process_pos(self, pos_msg):
        """Process a new position data point"""

        if pos_msg.timestamp <= self._pos_timestamp:
            self.class_log.warning(
                f"Duplicate or backwards timestamp. New timestamp: {pos_msg.timestamp}, "
                f"Most recent timestamp: {self._pos_timestamp}"
            )
            return

        self._pos_timestamp = pos_msg.timestamp

        if self._pos_counter % self.p['num_pos_points'] == 0:

            self._task_state = self._task_state_handler.get_task_state(
                self._pos_timestamp
            )

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

    def _get_peak_amplitude_relevant_channels(self,
            features: np.ndarray,
            printbit: bool = False
    )-> np.ndarray:
        '''
        (DS)get the output of _get_peak_amplitude and make the values distance away from the peak zero
        features: np.ndarray, shape (n_spikes, n_channels) -- output of _get_peak_amplitude
        distance: int -- number of channels away from the peak to keep ; if 2, then 5 channels will be kept (peak and 2 on each side),
            default value of 2 was chosen based on quantification of decoding error study by DS.
        '''
        distance = self.p['use_channel_dist_from_max_amp']

        if printbit:
            print("features.shape", features.shape)

        
        modified_features = np.zeros(features.shape)
        if len(features.shape) == 1:
            max_abs_index = np.argmax(np.abs(features))
            start_index = max(0, max_abs_index - distance)
            end_index = min(features.shape[0], max_abs_index + (distance+1))
            modified_features[start_index:end_index] = features[start_index:end_index]

        elif len(features.shape) == 2:
            for i in range(features.shape[0]):
                max_abs_index = np.argmax(np.abs(features[i]))
                start_index = max(0, max_abs_index - distance)
                end_index = min(features.shape[1], max_abs_index + (distance+1))
                modified_features[i, start_index:end_index] = features[i, start_index:end_index]


        return modified_features


    def _compute_mark(self, datapoint):
        """Compute mark vector given an object containing spike waveform
        data"""

        # Make sure format is (n_channels, n_waveform_points)
        spike_data = np.atleast_2d(datapoint.data)

        # Determine the peak value for each channel
        channel_peaks = np.max(spike_data, axis=1)

        # Find out which of the channels has the highest peak value
        peak_channel_ind = np.argmax(channel_peaks)

        # Determine at which index the peak value was observed, given
        # the channel computed immediately above
        t_ind = np.argmax(spike_data[peak_channel_ind])

        # Find the spike waveform values for each channel (i.e. a vector)
        # given the index computed immediately above
        amp_mark = spike_data[:, t_ind]

        if amp_mark.shape[0] > 2*self.p['use_channel_dist_from_max_amp'] + 1: #if nTrode sortgroup is larger (2*dist + 1) -- where this is meaningful
            amp_mark = self._get_peak_amplitude_relevant_channels(features = amp_mark)
        return amp_mark

    def _is_training_epoch(self):
        """Whether or not the encoding model is in the training phase"""

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
        """Record timing information for a processed spike event"""

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
        """Save timing data"""

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
        """Set up data objects given a list of electrode groups
        this object will be handling/managing"""

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
        """Final method called before exiting the main data processing loop"""

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
    """Top level object for encoder_process"""

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
        """Main data processing loop"""

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
