import numpy as np
import ghostipy as gsp
import time
import scipy.signal as sig

from mpi4py import MPI
from collections import OrderedDict
from typing import Sequence

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
                binary_record.RecordIDs.RIPPLE_END
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
                 'stats_frozen', 'is_consensus'] 
            ],
            rec_formats=['Iidd??ddddd', 'qqqqi10sdddd??', 'qqqqi10s?d??'],
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
        
        self._lockout_sample = OrderedDict()
        self._cons_lockout_sample = 0

        self._in_standard_ripple = {}
        self._in_cond_ripple = {}
        self._in_content_ripple = {}

        self._lfp_count = 0
        self._curr_timestamp = -1

        self._current_vel = 0

        self._init_params()


    def handle_message(self, msg, mpi_status):
        
        if isinstance(msg, messages.TrodeSelection):
            trodes = msg.trodes
            self.class_log.debug(f"Registering continuous channels: {trodes}")
            for trode in trodes:
                self._lfp_interface.register_datatype_channel(trode)
            
            # initialize filters too!
            self._envelope_estimator.initialize_filters(len(trodes))
            self._reset_stats(trodes)
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
            self._process_lfp(lfp_msg)

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
        filtered_data, env = self._envelope_estimator.add_new_data(
            msg_data
        )
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
            data = env[ii]
            mean = self._means[ii]
            sigma = self._sigmas[ii]

            (self._lockout_sample[trode],
             self._in_standard_ripple[trode],
             self._in_cond_ripple[trode],
             self._in_content_ripple[trode]
            ) = self._detect_ripple_bounds(
                t_send_data, t_recv_data, msg_timestamp,
                trode, data, mean, sigma,
                self._lockout_sample[trode], False
            )

        # detect start/end of ripples for consensus trace
        (self._cons_lockout_sample,
         self._in_standard_ripple_cons,
         self._in_cond_ripple_cons,
         self._in_content_ripple_cons
        ) = self._detect_ripple_bounds(
            t_send_data, t_recv_data, msg_timestamp,
            -1, cons_env, self._cons_mean, self._cons_sigma,
            self._cons_lockout_sample, True
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

        if is_consensus_trace:
            in_standard_ripple = self._in_standard_ripple_cons
            in_cond_ripple = self._in_cond_ripple_cons
            in_content_ripple = self._in_content_ripple_cons
        else:
            in_standard_ripple = self._in_standard_ripple[trode]
            in_cond_ripple = self._in_cond_ripple[trode]
            in_content_ripple = self._in_content_ripple[trode]


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
                self.write_record(
                    binary_record.RecordIDs.RIPPLE_DETECTED,
                    t_send_data, t_recv_data, time.time_ns(),
                    timestamp, trode, bytes('standard', 'utf-8'),
                    mean, sigma, self.p['standard_thresh'],
                    self.p['vel_thresh'], self.p['freeze_stats'],
                    is_consensus_trace
                )
                # self.class_log.info(f"Standard ripple detected {timestamp}, {is_consensus_trace}")
                lockout_sample += 1
                in_standard_ripple = True
                # log ripple detected
        
        else: # we are currently in a ripple

            lockout_sample += 1
            
            # while in a ripple, see if other thresholds exceeded
            # second condition is to prevent detecting ripple multiple times
            if (datapoint > mean + self.p['cond_thresh']*sigma and not in_cond_ripple):
                # sending message gets first priority
                self._send_ripple_message(
                    timestamp, trode, "cond", is_consensus_trace
                )
                self.write_record(
                    binary_record.RecordIDs.RIPPLE_DETECTED,
                    t_send_data, t_recv_data, time.time_ns(),
                    timestamp, trode, bytes('cond', 'utf-8'),
                    mean, sigma, self.p['cond_thresh'],
                    self.p['vel_thresh'], self.p['freeze_stats'],
                    is_consensus_trace
                )
                # self.class_log.info(f"Cond ripple detected {timestamp}, {is_consensus_trace}")
                # log conditioning ripple
                in_cond_ripple = True

            if (datapoint > mean + self.p['content_thresh']*sigma and not in_content_ripple):
                # sending message gets first priority
                self._send_ripple_message(
                    timestamp, trode, "content", is_consensus_trace
                )
                self.write_record(
                    binary_record.RecordIDs.RIPPLE_DETECTED,
                    t_send_data, t_recv_data, time.time_ns(),
                    timestamp, trode, bytes('content', 'utf-8'),
                    mean, sigma, self.p['content_thresh'],
                    self.p['vel_thresh'], self.p['freeze_stats'],
                    is_consensus_trace
                )
                # self.class_log.info(f"Content ripple detected {timestamp}, {is_consensus_trace}")
                # log content ripple
                in_content_ripple = True

            # test for end-of-ripple conditions
            tup = (
                (in_standard_ripple, 'standard'),
                (in_cond_ripple, 'cond'),
                (in_content_ripple, 'content')
            )
            if datapoint <= mean + self.p['end_thresh']*sigma:

                # sending message gets first priority
                self._send_ripple_message(
                    timestamp, trode, "end", is_consensus_trace
                )
                for test_condition, rtype in tup:
                    if test_condition:
                        self.write_record(
                            binary_record.RecordIDs.RIPPLE_END,
                            t_send_data, t_recv_data, time.time_ns(),
                            timestamp, trode, bytes(rtype, 'utf-8'),
                            True, self.p['end_thresh'],
                            self.p['freeze_stats'], is_consensus_trace
                        )

                # self.class_log.info(f"Ripple ended {timestamp}, {is_consensus_trace}")
                lockout_sample = 0
                in_standard_ripple = False
                in_cond_ripple = False
                in_content_ripple = False
                # log ripple ended

            if lockout_sample >= self.p['max_ripple_samples']:

                # sending message gets first priority
                self._send_ripple_message(
                    timestamp, trode, "end", is_consensus_trace
                )
                for test_condition, rtype in tup:
                    if test_condition:
                        self.write_record(
                            binary_record.RecordIDs.RIPPLE_END,
                            t_send_data, t_recv_data, time.time_ns(),
                            timestamp, trode, bytes(rtype, 'utf-8'),
                            False, self.p['end_thresh'],
                            self.p['freeze_stats'], is_consensus_trace
                        )

                # self.class_log.info(f"Ripple ended {timestamp}, {is_consensus_trace}")
                lockout_sample = 0
                in_standard_ripple = False
                in_cond_ripple = False
                in_content_ripple = False
                # log ripple ended

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

    def freeze_stats(self):
        self.class_log.info("Updating stats disabled")
        self.p['freeze_stats'] = True

    def unfreeze_stats(self):
        self.p['freeze_stats'] = False

    def _reset_stats(self, trodes:Sequence):

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

        # for individual trace
        for trode in trodes:
            self._lockout_sample[trode] = 0
            self._in_standard_ripple[trode] = False
            self._in_cond_ripple[trode] = False
            self._in_content_ripple[trode] = False

        self._in_standard_ripple_cons = False
        self._in_cond_ripple_cons = False
        self._in_content_ripple_cons = False

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

    def _init_params(self):
        self.p = {}

        self.p['smooth_x'] = self._config['kinematics']['smooth_x']
        self.p['smooth_y'] = self._config['kinematics']['smooth_y']
        self.p['smooth_speed'] = self._config['kinematics']['smooth_speed']

        self.p['send_lfp_timestamp'] = (self.rank == self._config['rank']['ripples'][0])
        self.p['lfp_samples_per_time_bin'] = self._compute_lfp_send_interval()
        self.p['max_ripple_samples'] = self._config["ripples"]["max_ripple_samples"]
        self.p['num_lfp_disp'] = self._config['display']['ripples']['lfp']
        self.p['vel_thresh'] = self._config['ripples']['vel_thresh']
        self.p['standard_thresh'] = self._config['ripples']['threshold']['standard']
        self.p['cond_thresh'] = self._config['ripples']['threshold']['conditioning']
        self.p['content_thresh'] = self._config['ripples']['threshold']['content']
        self.p['end_thresh'] = self._config['ripples']['threshold']['end']
        self.p['freeze_stats'] = False

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

        self.class_log.info("Exited main loop")