import numpy as np
import ghostipy as gsp
import time
import os
import scipy.signal as sig

from mpi4py import MPI
from collections import OrderedDict
from typing import List

from realtime_decoder import (
    base, utils, datatypes, messages, binary_record,
    position
)

####################################################################################
# Interfaces
####################################################################################

class RippleMPISendInterface(base.StandardMPISendInterface):

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    def send_lfp_timestamp(self, timestamp):
        for r in self.config['rank']['decoders']:
            self.comm.send(
                timestamp, dest=r,
                tag=messages.MPIMessageTag.LFP_TIMESTAMP
            )

    def send_ripple(self, dest, msg):
        self.comm.Send(
            buf=msg.tobytes(),
            dest=dest,
            tag=messages.MPIMessageTag.RIPPLE_DETECTION
        )
####################################################################################
# Data handlers/managers
####################################################################################
    
class EnvelopeEstimator(base.LoggingClass):

    def __init__(self, config):
        super().__init__()
        self._config = config

    def initialize_filters(self, num_signals:int):

        # set up ripple band filter
        filt = self._config['ripples']['filter']['type']
        if filt == 'fir':
            self._b_ripple = gsp.firdesign(
                self._config['ripples']['filter']['num_taps'],
                self._config['ripples']['filter']['band_edges'],
                self._config['ripples']['filter']['desired'],
                fs=self._config['sampling_rate']['lfp']
            )[:, None]
            self._a_ripple = None
            self._x_ripple = np.zeros((self._b_ripple.shape[0], num_signals))
            self._y_ripple = None
        elif filt == 'iir':
            # Use second-order sections for numeric stability,
            # add extra axis for broadcasting to the number
            # of channels dimension
            sos = sig.iirfilter(
                self._config['ripples']['filter']['order'],
                self._config['ripples']['filter']['crit_freqs'],
                output='sos',
                fs=self._config['sampling_rate']['lfp'],
                **self._config['ripples']['filter']['kwargs']
            )[:, :, None]

            ns = sos.shape[0]
            self._b_ripple = sos[:, :3] # (ns, 3, num_signals)
            self._a_ripple = sos[:, 3:] # (ns, 3, num_signals)

            self._x_ripple = np.zeros((ns, 3, num_signals))
            self._y_ripple = np.zeros((ns, 3, num_signals))
        else:
            raise ValueError(f"Invalid filter type {filt}")

        # set up envelope filter
        self._b_env = gsp.firdesign(
            self._config["ripples"]["smoothing_filter"]["num_taps"],
            self._config["ripples"]["smoothing_filter"]["band_edges"],
            self._config["ripples"]["smoothing_filter"]["desired"],
            fs=self._config["sampling_rate"]["lfp"]
        )[:, None]
        self._x_env = np.zeros((self._b_env.shape[0], num_signals))

    def add_new_data(self, data):
        
        # filter to ripple band
        if self._a_ripple is not None: # IIR
            ns = self._a_ripple.shape[0]
            for ii in range(ns):

                self._x_ripple[ii, 1:] = self._x_ripple[ii, :-1]
                if ii == 0: # new input is incoming data
                    self._x_ripple[ii, 0] = data
                else: # new input is IIR output of previous stage
                    self._x_ripple[ii, 0] = self._y_ripple[ii - 1, 0]

                self._y_ripple[ii, 1:] = self._y_ripple[ii, :-1]
                ripple_data = (
                    np.sum(self._b_ripple[ii] * self._x_ripple[ii], axis=0) -
                    np.sum(self._a_ripple[ii, 1:] * self._y_ripple[ii, 1:], axis=0)
                )
                self._y_ripple[ii, 0] = ripple_data
        else: # FIR
            self._x_ripple[1:] = self._x_ripple[:-1]
            self._x_ripple[0] = data
            ripple_data = np.sum(
                self._b_ripple * self._x_ripple, axis=0
            )

        # estimate envelope
        self._x_env[1:] = self._x_env[:-1]
        self._x_env[0] = ripple_data**2
        env = np.sqrt(np.sum(self._b_env * self._x_env, axis=0))

        return ripple_data, env

class RippleManager(base.BinaryRecordBase, base.MessageHandler):

    def __init__(
        self, rank, config, send_interface, lfp_interface,
        pos_interface
    ):

        super().__init__(
            rank=rank,
            rec_ids=[
                binary_record.RecordIDs.RIPPLE_STATE,
                binary_record.RecordIDs.RIPPLE_DETECTED,
                binary_record.RecordIDs.RIPPLE_END,
                binary_record.RecordIDs.RIPPLE_EVENT
            ],
            rec_labels=[
                ['timestamp',
                'elec_grp_id',
                'content_rip_threshold',
                'conditioning_rip_threshold',
                'thresh_crossed',
                'lockout',
                'custom_mean',
                'custom_std',
                'lfp_data',
                'rd',
                'current_val'],
                ['t_send_data', 't_recv_data', 't_sys', 'timestamp',
                 'elec_grp_id', 'ripple_type', 'env_mean', 'env_std',
                 'threshold_sigma', 'vel_thresh', 'stats_frozen', 'is_consensus'],
                ['t_send_data', 't_recv_data', 't_sys', 'timestamp',
                 'elec_grp_id', 'ripple_type', 'normal_end', 'threshold_sigma',
                 'stats_frozen', 'is_consensus'],
                ['elec_grp_id', 'timestamp_start', 'timestamp_end',
                't_send_data_start', 't_recv_data_start', 't_sys_start',
                't_send_data_end', 't_recv_data_end', 't_sys_end',
                'ripple_type', 'env_mean', 'env_std', 'threshold_sigma_start',
                'threshold_sigma_end', 'normal_end', 'stats_frozen', 'is_consensus']
            ],
            rec_formats=[
                'Iidd??ddddd', 'qqqqi10sdddd??',
                'qqqqi10s?d??', 'iqqqqqqqq10sdddd???'
            ],
            send_interface=send_interface,
            manager_label='state'
        )

        self._config = config
        self._lfp_interface = lfp_interface
        self._pos_interface = pos_interface
        self._envelope_estimator = EnvelopeEstimator(config)
        self._kinestimator = position.KinematicsEstimator(
            scale_factor=config['kinematics']['scale_factor'],
            dt=1/config['sampling_rate']['position'],
            xfilter=config['kinematics']['smoothing_filter'],
            yfilter=config['kinematics']['smoothing_filter'],
            speedfilter=config['kinematics']['smoothing_filter'],
        )

        self._ripple_msg = np.zeros(
            (1, ), dtype=messages.get_dtype("Ripples")
        )

        # map trode to index. doesn't have to be ordered
        self._trode_ind_map = {}

        self._lockout_sample = OrderedDict()

        self._rtimes = {}
        self._in_ripple = {}

        self._lfp_count = 0
        self._curr_timestamp = -1

        self._current_vel = 0

        self._init_params()
        self._init_timings()


    def handle_message(self, msg, mpi_status):
        
        if isinstance(msg, messages.TrodeSelection):
            trodes = msg.trodes
            self.class_log.debug(f"Registering continuous channels: {trodes}")
            for ii, trode in enumerate(trodes):
                if trode == -1:
                    raise ValueError(
                        "Invalid trode ID. An ID of -1 is reserved "
                        "for the consensus trace"
                    )

                self._lfp_interface.register_datatype_channel(trode)
                self._trode_ind_map[trode] = ii
            
            # initialize filters too!
            self._envelope_estimator.initialize_filters(len(trodes))
            self._reset_data(trodes)
            self._seed_stats(trodes)
        elif isinstance(msg, messages.BinaryRecordCreate):
            self.set_record_writer_from_message(msg)
        elif isinstance(msg, messages.StartRecordMessage):
            self.class_log.info("Starting records")
            self.start_record_writing()
        elif isinstance(msg, messages.ActivateDataStreams):
            self.class_log.info("Activating datastreams")
            self._lfp_interface.activate()
            self._pos_interface.activate()
        elif isinstance(msg, messages.TerminateSignal):
            self._lfp_interface.deactivate()
            self._pos_interface.deactivate()
            raise StopIteration()
        elif isinstance(msg, messages.VerifyStillAlive):
            self.send_interface.send_alive_message()
        elif isinstance(msg, messages.GuiRippleParameters):
            self._update_gui_params(msg)
        else:
            self._class_log.warning(
                f"Received message of type {type(msg)} "
                f"from source: {mpi_status.source}, "
                f" tag: {mpi_status.tag}, ignoring"
            )

    def next_iter(self):
        lfp_msg = self._lfp_interface.__next__()
        if lfp_msg is not None:
            t0 = time.time_ns()
            self._process_lfp(lfp_msg)
            t1 = time.time_ns()
            if (t1 - t0)/1e6 > 50:
                self.class_log.warning("Timestamp {lfp_msg.timestamp}, high latency {lat_ms}")
            self._record_timings(
                lfp_msg.timestamp,
                lfp_msg.t_send_data, lfp_msg.t_recv_data,
                t0, t1, len(self._in_ripple.keys())
            )

        pos_msg = self._pos_interface.__next__()
        if pos_msg is not None:
            self._process_pos(pos_msg)

    def _process_lfp(self, msg):

        msg_data = msg.data
        msg_timestamp = msg.timestamp
        t_send_data = msg.t_send_data
        t_recv_data = msg.t_recv_data

        if msg_timestamp != self._curr_timestamp:
            self._curr_timestamp = msg_timestamp
            self._lfp_count += 1

        # get envelope
        filtered_data, env = self._envelope_estimator.add_new_data(msg_data)
        cons_env = np.mean(env)

        # updates stats
        if not self.p['freeze_stats']:
            self._means, self._M2, self._counts = utils.estimate_new_stats(
                env, self._means, self._M2, self._counts
            )
            self._sigmas = np.sqrt(self._M2 / self._counts)

            self._cons_mean, self._cons_M2, self._cons_count = utils.estimate_new_stats(
                cons_env, self._cons_mean, self._cons_M2, self._cons_count
            )
            self._cons_sigma = np.sqrt(self._cons_M2 / self._cons_count)

        # detect start/end of ripples
        for ii, trode in enumerate(self._lockout_sample.keys()):

            if trode == -1: # use consensus data
                data = cons_env
                mean = self._cons_mean
                sigma = self._cons_sigma
                is_consensus = True
            else:
                data = env[ii]
                mean = self._means[ii]
                sigma = self._sigmas[ii]
                is_consensus = False

            (
                self._lockout_sample[trode],
                self._in_ripple[trode]['standard'],
                self._in_ripple[trode]['cond'],
                self._in_ripple[trode]['content']
            ) = self._detect_ripple_bounds(
                t_send_data, t_recv_data, msg_timestamp,
                trode, data, mean, sigma,
                self._lockout_sample[trode], is_consensus
            )

        # send timestamp to decoder for timekeeping
        if (
            self.p['send_lfp_timestamp'] and
            self._lfp_count % self.p['lfp_samples_per_time_bin'] == 0
        ):
            # self.class_log.info(f"Sending timestamp {self._curr_timestamp}")
            self.send_interface.send_lfp_timestamp(self._curr_timestamp)

        if self._lfp_count % self.p['num_lfp_disp'] == 0:
            self.class_log.debug(f"Received {self._lfp_count} lfp points")

    def _process_pos(self, pos_msg):

        # calculate velocity using the midpoints
        xmid = (pos_msg.x + pos_msg.x2)/2
        ymid = (pos_msg.y + pos_msg.y2)/2

        # we don't care about x and y returned by compute_kinematics(),
        # as we are using the position mapper to get the appropriate
        # linear coordinates
        _, _, self._current_vel = self._kinestimator.compute_kinematics(
            xmid, ymid,
            smooth_x=self.p['smooth_x'],
            smooth_y=self.p['smooth_y'],
            smooth_speed=self.p['smooth_speed']
        )

    def _detect_ripple_bounds(
        self, t_send_data, t_recv_data, timestamp,
        trode, datapoint, mean, sigma,
        lockout_sample, is_consensus_trace
    ):

        in_standard_ripple = self._in_ripple[trode]['standard']
        in_cond_ripple = self._in_ripple[trode]['cond']
        in_content_ripple = self._in_ripple[trode]['content']


        if lockout_sample == 0: # ok to detect ripples
            # note that velocity threshold is used only when detecting the
            # start of a standard ripple. all other detections (i.e. other
            # ripple types and end of a ripple) are not dependent on the
            # velocity threshold
            if (
                datapoint > mean + self.p['standard_thresh']*sigma and
                self._current_vel < self.p['vel_thresh'] and
                not in_standard_ripple

            ):

                # sending message gets first priority
                self._send_ripple_message(
                    timestamp, trode, "standard", is_consensus_trace
                )

                self._rtimes[trode]['standard']['timestamp_start'] = timestamp
                self._rtimes[trode]['standard']['t_send_data_start'] = t_send_data
                self._rtimes[trode]['standard']['t_recv_data_start'] = t_recv_data
                self._rtimes[trode]['standard']['t_sys_start'] = time.time_ns()

                lockout_sample += 1
                in_standard_ripple = True
        
        else: # we are currently in a ripple

            lockout_sample += 1

            # while in a ripple, see if other thresholds exceeded
            # second condition is to prevent detecting ripple multiple times
            if (datapoint > mean + self.p['cond_thresh']*sigma and not in_cond_ripple):

                # sending message gets first priority
                self._send_ripple_message(
                    timestamp, trode, "cond", is_consensus_trace
                )

                self._rtimes[trode]['cond']['timestamp_start'] = timestamp
                self._rtimes[trode]['cond']['t_send_data_start'] = t_send_data
                self._rtimes[trode]['cond']['t_recv_data_start'] = t_recv_data
                self._rtimes[trode]['cond']['t_sys_start'] = time.time_ns()

                in_cond_ripple = True

            if (datapoint > mean + self.p['content_thresh']*sigma and not in_content_ripple):

                # sending message gets first priority
                self._send_ripple_message(
                    timestamp, trode, "content", is_consensus_trace
                )

                self._rtimes[trode]['content']['timestamp_start'] = timestamp
                self._rtimes[trode]['content']['t_send_data_start'] = t_send_data
                self._rtimes[trode]['content']['t_recv_data_start'] = t_recv_data
                self._rtimes[trode]['content']['t_sys_start'] = time.time_ns()

                in_content_ripple = True

            # test for end-of-ripple conditions
            tup = (
                (in_standard_ripple, 'standard'),
                (in_cond_ripple, 'cond'),
                (in_content_ripple, 'content')
            )
            if datapoint <= mean + self.p['end_thresh']*sigma:

                (
                    lockout_sample,
                    in_standard_ripple,
                    in_cond_ripple,
                    in_content_ripple
                ) = self._handle_ripple_end(
                    timestamp, trode, mean, sigma,
                    t_send_data, t_recv_data, time.time_ns(),
                    True, is_consensus_trace, tup
                ) 

            if lockout_sample >= self.p['max_ripple_samples']:

                (
                    lockout_sample,
                    in_standard_ripple,
                    in_cond_ripple,
                    in_content_ripple

                ) = self._handle_ripple_end(
                    timestamp, trode, mean, sigma,
                    t_send_data, t_recv_data, time.time_ns(),
                    False, is_consensus_trace, tup
                ) 

        return lockout_sample, in_standard_ripple, in_cond_ripple, in_content_ripple

    def _send_ripple_message(
        self, timestamp, elec_grp_id, ripple_type,
        is_consensus
    ):
        self._ripple_msg[0]['timestamp'] = timestamp
        self._ripple_msg[0]['elec_grp_id'] = elec_grp_id
        self._ripple_msg[0]['ripple_type'] = ripple_type
        self._ripple_msg[0]['is_consensus'] = is_consensus

        self.send_interface.send_ripple(
            self._config['rank']['supervisor'][0],
            self._ripple_msg
        )

    def _handle_ripple_end(
        self, timestamp, trode, mean, sigma,
        t_send_data_end, t_recv_data_end,
        t_sys, is_normal_end, is_consensus_trace,
        condition_tuple
    ):

        # sending message gets first priority
        self._send_ripple_message(
            timestamp, trode, "end", is_consensus_trace
        )

        for in_ripple, rtype in condition_tuple:

            if in_ripple:

                if rtype == 'standard':
                    nsigma1 = self.p['standard_thresh']
                elif rtype == 'cond':
                    nsigma1 = self.p['cond_thresh']
                elif rtype == 'content':
                    nsigma1 = self.p['content_thresh']

                self.write_record(
                    binary_record.RecordIDs.RIPPLE_EVENT,
                    trode, self._rtimes[trode][rtype]['timestamp_start'],
                    timestamp, self._rtimes[trode][rtype]['t_send_data_start'],
                    self._rtimes[trode][rtype]['t_recv_data_start'],
                    self._rtimes[trode][rtype]['t_sys_start'],
                    t_send_data_end, t_recv_data_end, t_sys,
                    bytes(rtype, 'utf-8'), mean, sigma, nsigma1,
                    self.p['end_thresh'], is_normal_end,
                    self.p['freeze_stats'], is_consensus_trace
                )

        return 0, False, False, False

    def freeze_stats(self):
        self.class_log.info("Updating stats disabled")
        self.p['freeze_stats'] = True

    def unfreeze_stats(self):
        self.p['freeze_stats'] = False

    def _reset_data(self, trodes:List):

        self._reset_stats(trodes)

        # all trodes plus a -1 for the consensus trace
        alltrodes = trodes + [-1]

        for trode in alltrodes:
            self._rtimes[trode] = {}
            self._in_ripple[trode] = {}

            for rtype in ('standard', 'cond', 'content'):
                self._rtimes[trode][rtype] = {}
                self._rtimes[trode][rtype]['timestamp_start'] = 0
                self._rtimes[trode][rtype]['t_send_data_start'] = 0
                self._rtimes[trode][rtype]['t_recv_data_start'] = 0
                self._rtimes[trode][rtype]['t_sys_start'] = 0

                self._in_ripple[trode][rtype] = False

            self._lockout_sample[trode] = 0

    def _reset_stats(self, trodes:List):

        num_signals = len(trodes)
        # stats for individual traces
        self._means = np.zeros(num_signals)
        self._M2 = np.zeros(num_signals)
        self._sigmas = np.zeros(num_signals)
        self._counts = np.zeros(num_signals)

        # stats for consensus trace
        self._cons_mean = 0
        self._cons_M2 = 0
        self._cons_sigma = 0
        self._cons_count = 0

    def _seed_stats(self, trodes:List):

        for trode in trodes:

            ind = self._trode_ind_map[trode]

            try:
                self._means[ind] = self._config['ripples']['custom_mean'][trode]
                self._sigmas[ind] = self._config['ripples']['custom_std'][trode]
                self._counts[ind] = 1
                self._M2[ind] = self._sigmas[ind]**2 * self._counts[ind]
                self.class_log.debug(f"Seeded stats for trode {trode}")
            except KeyError:
                pass

        try:
            self._cons_mean = self._config['ripples']['custom_mean']['consensus']
            self._cons_sigma = self._config['ripples']['custom_std']['consensus']
            self._cons_count = 1
            self._cons_M2 = self._cons_sigma**2 * self._cons_count
            self.class_log.debug("Seeded stats for consensus trace")
        except KeyError:
            pass

    def _compute_lfp_send_interval(self):

        ds, rem = divmod(
            self._config['sampling_rate']['spikes'],
            self._config['sampling_rate']['lfp']
        )
        if rem != 0:
            raise ValueError(
                "LFP/Spike downsampling factor is not an integer!"
            )

        n_lfp_samples, rem = divmod(
            self._config['decoder']['time_bin']['samples'],
            (self._config['sampling_rate']['spikes'] /
            self._config['sampling_rate']['lfp']
            )
        )
        if rem != 0:
            raise ValueError(
                "Number of LFP samples per time bin is not an integer!"
            )

        return n_lfp_samples

    def _init_timings(self):
        dt = np.dtype([
            ('ripple_rank', '=i4'),
            ('timestamp', '=i8'),
            ('t_send_data', '=i8'),
            ('t_recv_data', '=i8'),
            ('t_start_processing', '=i8'),
            ('t_end_processing', '=i8'),
            ('num_traces', '=i4'),
        ])
        self._times = np.zeros(
            self.p['timings_bufsize'],
            dtype=dt
        )
        self._times_ind = 0

    def _record_timings(
        self, timestamp, t_send_data, t_recv_data,
        t_start, t_end, num_traces
    ):

        ind = self._times_ind

        if ind == len(self._times):
            self._times = np.hstack((
                self._times,
                np.zeros(
                    self.p['timings_bufsize'],
                    dtype=self._times.dtype
                )
            ))

        self._times[ind]['ripple_rank'] = self.rank
        self._times[ind]['timestamp'] = timestamp
        self._times[ind]['t_send_data'] = t_send_data
        self._times[ind]['t_recv_data'] = t_recv_data
        self._times[ind]['t_start_processing'] = t_start
        self._times[ind]['t_end_processing'] = t_end
        self._times[ind]['num_traces'] = num_traces
        self._times_ind += 1

    def _init_params(self):
        self.p = {}

        self.p['smooth_x'] = self._config['kinematics']['smooth_x']
        self.p['smooth_y'] = self._config['kinematics']['smooth_y']
        self.p['smooth_speed'] = self._config['kinematics']['smooth_speed']

        self.p['timings_bufsize'] = self._config['ripples']['timings_bufsize']
        self.p['send_lfp_timestamp'] = (self.rank == self._config['rank']['ripples'][0])
        self.p['lfp_samples_per_time_bin'] = self._compute_lfp_send_interval()
        self.p['max_ripple_samples'] = self._config["ripples"]["max_ripple_samples"]
        self.p['num_lfp_disp'] = self._config['display']['ripples']['lfp']
        self.p['vel_thresh'] = self._config['ripples']['vel_thresh']
        self.p['standard_thresh'] = self._config['ripples']['threshold']['standard']
        self.p['cond_thresh'] = self._config['ripples']['threshold']['conditioning']
        self.p['content_thresh'] = self._config['ripples']['threshold']['content']
        self.p['end_thresh'] = self._config['ripples']['threshold']['end']
        self.p['freeze_stats'] = self._config['ripples']['freeze_stats']

        if (
            self.p['standard_thresh'] >= self.p['cond_thresh'] or
            self.p['standard_thresh'] >= self.p['content_thresh']
        ):
            raise ValueError(
                f"Ripple detection threshold {self.p['standard_thresh']} "
                "is not lower than the conditioning and content thresholds"
            )

    def _update_gui_params(self, gui_msg):
        self.class_log.info("Updating GUI ripple parameters")
        self.p['vel_thresh'] = gui_msg.velocity_threshold
        self.p['standard_thresh'] = gui_msg.ripple_threshold
        self.p['cond_thresh'] = gui_msg.conditioning_ripple_threshold
        self.p['content_thresh'] = gui_msg.content_ripple_threshold
        self.p['end_thresh'] = gui_msg.end_ripple_threshold
        self.p['freeze_stats'] = gui_msg.freeze_stats

    def finalize(self):
        filename = os.path.join(
            self._config['files']['output_dir'],
            f"{self._config['files']['prefix']}_ripples_rank_{self.rank}." +
            f"{self._config['files']['timing_postfix']}.npz"
        )
        np.savez(filename, timings=self._times[:self._times_ind])
        self.class_log.info(
            f"Wrote timings file for ripple rank {self.rank} to {filename}")

####################################################################################
# Processes
####################################################################################

class RippleProcess(base.RealtimeProcess):

    def __init__(self, comm, rank, config, lfp_interface, pos_interface):
        super().__init__(comm, rank, config)

        try:
            self._ripple_manager = RippleManager(
                rank, config, RippleMPISendInterface(comm, rank, config),
                lfp_interface, pos_interface)
        except:
            self.class_log.exception("Exception in init!")

        self._mpi_recv = base.StandardMPIRecvInterface(
            comm, rank, config, messages.MPIMessageTag.COMMAND_MESSAGE,
            self._ripple_manager
        )

        self._gui_recv = base.StandardMPIRecvInterface(
            comm, rank, config, messages.MPIMessageTag.GUI_PARAMETERS,
            self._ripple_manager
        )

    def main_loop(self):

        try:
            self._ripple_manager.setup_mpi()
            t0 = time.time()
            freeze = True
            while True:
                self._mpi_recv.receive()
                self._gui_recv.receive()
                self._ripple_manager.next_iter()
                # if time.time() - t0 > (5.5*60):
                #     if freeze:
                #         self._ripple_manager.freeze_stats()
                #         freeze = False

        except StopIteration as ex:
            self.class_log.info("Exiting normally")
        except Exception as e:
            self.class_log.exception(
                "Ripple process exception occurred!"
            )

        self._ripple_manager.finalize()
        self.class_log.info("Exited main loop")