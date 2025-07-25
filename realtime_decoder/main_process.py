#

import time
import numpy as np

from typing import Sequence, List
from mpi4py import MPI

from realtime_decoder import (
    base, trodes_file_sim, utils, messages, binary_record
)

####################################################################################
# Interfaces
####################################################################################

class GenericMainRecvInterface(base.MPIRecvInterface):
    """Generic receiving interface object for main_process"""

    def __init__(
        self, comm, rank, config,
        msg_dtype, msg_tag, msg_handler
    ):
        super().__init__(comm, rank, config)
        self._msg_dtype = msg_dtype
        self._msg_tag = msg_tag
        self._msg_handler = msg_handler

        self._msg_buffer = bytearray(self._msg_dtype.itemsize)
        self._mpi_status = MPI.Status()
        self._req = self.comm.Irecv(
            buf=self._msg_buffer,
            tag=self._msg_tag
        )

    def receive(self):
        """Test for a message and if available, process it"""

        rdy = self._req.Test(status=self._mpi_status)
        if rdy:
            msg = np.frombuffer(self._msg_buffer, dtype=self._msg_dtype)
            self._msg_handler.handle_message(msg, self._mpi_status)
            self._req = self.comm.Irecv(
                buf=self._msg_buffer,
                tag=self._msg_tag
            )

class MainMPISendInterface(base.MPISendInterface):
    """Sending interface object for main_process"""

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    def send_record_register_messages(self):
        """Raise an error since this method is not supposed to be called
        for this particular object"""

        raise NotImplementedError(
            "This interface does not send records to the main process to be registered")

    def send_trode_selection(self, rank:int, trodes:Sequence[int]):
        """Send a message notifying another process which electrode
        groups it should be handling"""

        self.comm.send(
            obj=messages.TrodeSelection(trodes),
            dest=rank, tag=messages.MPIMessageTag.COMMAND_MESSAGE
        )

    def send_activate_datastream(self, ranks:Sequence[int]):
        """Send a message telling another process to activate its
        data receivers"""

        for rank in ranks:
            self.comm.send(
                obj=messages.ActivateDataStreams(),
                dest=rank, tag=messages.MPIMessageTag.COMMAND_MESSAGE
            )

    def send_setup_complete(self):
        """Notify GUI that setup is complete for all processes
        and the decoder system is ready to start processing neural
        data"""

        self.class_log.debug("Sending SetupComplete message")

        # very hacky but sometimes the GUI fails to receive a SetupComplete
        # messages. here we send it multiple times in the hopes that the
        # GUI will pick up at least one of them
        for ii in range(self.config['num_setup_messages']):
            self.comm.send(
                obj=messages.SetupComplete(data=ii),
                dest=self.config['rank']['gui'][0],  # need to check config?
                tag=messages.MPIMessageTag.COMMAND_MESSAGE
            )

    def send_termination_signal(self, ranks:Sequence[int], *, exit_code=0):
        """Send a list of ranks a signal to terminate"""

        obj = messages.TerminateSignal(exit_code=exit_code)
        for rank in ranks:
            self.comm.send(
                obj=obj, dest=rank,
                tag=messages.MPIMessageTag.COMMAND_MESSAGE
            )

    def send_new_writer_message(self, ranks:Sequence[int], new_writer_message):
        """Send a list of ranks a message to instantiate a writer
        object for recording their data"""

        for rank in ranks:
            self.comm.send(
                obj=new_writer_message, dest=rank,
                tag=messages.MPIMessageTag.COMMAND_MESSAGE)

    def send_start_rec_message(self, ranks:Sequence[int]):
        """Send a list of ranks a message to enable recording
        their data"""

        for rank in ranks:
            self.comm.send(
                obj=messages.StartRecordMessage(), dest=rank,
                tag=messages.MPIMessageTag.COMMAND_MESSAGE)

    def send_decoder_started(self):
        """Notify GUI that the decoder system has begun processing
        neural data"""

        self.comm.send(
            obj=messages.DecoderStarted(),
            dest=self.config['rank']['gui'][0],  # need to check config?
            tag=messages.MPIMessageTag.COMMAND_MESSAGE
        )

    def send_synchronize_datastreams(
        self, ranks:Sequence[int], start_time:float,
        first_timestamp:int, last_timestamp:int):

        for rank in ranks:
            self.comm.send(
                obj=messages.SynchronizeDatastreams(
                    start_time, first_timestamp, last_timestamp),
                dest=rank,
                tag=messages.MPIMessageTag.COMMAND_MESSAGE
            )

####################################################################################
# Managers
####################################################################################

class MainManager(base.MessageHandler):
    """A manager class that handles MPI, among other functions"""

    def __init__(
        self, rank, num_ranks, config, send_interface, stim_decider, *,
        manager_label='state'
    ):

        super().__init__()

        self._rank = rank
        self._config = config
        self._send_interface = send_interface
        self._stim_decider = stim_decider

        self._rec_manager = binary_record.BinaryRecordsManager(
            manager_label,
            len(str(num_ranks)),
            self._config['files']['output_dir'],
            self._config['files']['prefix'],
            self._config['files']['rec_postfix']
        )

        self._ranks_to_monitor = []
        for proc_type in self._config['rank']:
            rank_list = self._config['rank'][proc_type]
            for rank in rank_list:
                if self._rank != rank:
                    self._ranks_to_monitor.append(rank)
        self._ranks_to_monitor = sorted(self._ranks_to_monitor)

        self._ranks_sending_recs = sorted(
            self._config['rank_settings']['enable_rec']
        )
        try:
            self._ranks_sending_recs.remove(self._rank)
        except:
            pass

        self._set_up_ranks = []
        self._all_ranks_set_up = False

        # stim decider bypasses the normal record registration message sending
        for message in stim_decider.get_records():
            self._rec_manager.register_rec_type_message(message)

        self._started = False

    def send_setup_complete(self):
        """Send message that setup is complete"""

        self._send_interface.send_setup_complete()

    def handle_message(self, msg, mpi_status):
        """Process message"""

        source_rank = mpi_status.source

        if isinstance(msg, messages.StartupSignal):
            self.class_log.info(
                f"Received StartupSignal from rank {source_rank}"
            )
            self.startup()

        elif isinstance(msg, messages.TerminateSignal):
            self.class_log.info(
                f"Received TerminateSignal from rank {source_rank}, now terminating all"
            )
            self.trigger_termination()

        elif isinstance(msg, messages.BinaryRecordType):
            self.class_log.debug(
                f"BinaryRecordType received for rec id {msg.rec_id} from rank {source_rank}"
            )
            self._rec_manager.register_rec_type_message(msg)

        elif isinstance(msg, messages.BinaryRecordSendComplete):
            self._update_all_rank_setup_status(source_rank)

        else:
            self.class_log.warning(
                f"Unrecognized command message of type {type(msg)}, ignoring")

    def handle_gui_message(self, msg, mpi_status):
        """Raise error because this object is not designed to handle
        messages sent by the GUI"""

        raise NotImplementedError(
            "MainManager does not handle GUI messages"
        )

    def startup(self):
        """Initiate startup sequence"""

        if not self._started:
            self.class_log.info("Starting up")

            # startup encoder ranks
            self._startup_encoders()

            # startup ripple ranks
            self._startup_ripples()

            # Update binary_record file writers before starting datastream
            self._startup_rec_writers()
            self._activate_datastreams()

            if self._config['datasource'] == 'trodes_file_simulator':
                # For this special case, we need to synchronize the file
                # readers to a global time
                self._sync_file_readers()

            self._send_decoder_started()

            self._started = True
        else:
            self.class_log.warning(
                "Already started up, ignoring startup request"
            )

    def trigger_termination(self, *, raise_stop_iteration=True):
        """Tell all ranks to terminate"""

        self._send_interface.send_termination_signal(self._ranks_to_monitor)

        if raise_stop_iteration:
            raise StopIteration()

    def terminate_remaining_processes(self, ranks):
        """Terminate ranks that are still alive and then
        terminate this process as well"""

        self._send_interface.send_termination_signal(
            sorted(ranks), exit_code=1
        )
        raise StopIteration()

    def _startup_ripples(self):
        """Notify ripple processes what electrode groups they
        should be handling"""

        trodes = self._config["trode_selection"]["ripples"]

        # Round robin allocation of channels to ripple ranks
        enable_count = 0
        num_ranks = len(self._config["rank"]["ripples"])
        trode_assignments = [[] for _ in self._config["rank"]["ripples"]]
        for trode in trodes:
            trode_assignments[enable_count % num_ranks].append(trode)
            enable_count += 1

        for rank_ind, rank in enumerate(self._config["rank"]["ripples"]):
            self.class_log.info(
                f"Assigned to rank {rank} "
                f"ripple tetrodes: {trode_assignments[rank_ind]}"
            )
            self._send_interface.send_trode_selection(
                rank, trode_assignments[rank_ind]
            )

    def _startup_encoders(self):
        """Notify encoder processes what electrode groups they
        should be handling"""

        trodes = self._config["trode_selection"]["decoding"]

        # Round robin allocation of channels to encoder ranks
        enable_count = 0
        num_ranks = len(self._config["rank"]["encoders"])
        trode_assignments = [[] for _ in self._config["rank"]["encoders"]]
        for trode in trodes:
            trode_assignments[enable_count % num_ranks].append(trode)
            enable_count += 1

        for rank_ind, rank in enumerate(self._config["rank"]["encoders"]):
            self.class_log.info(
                f"Assigned to rank {rank} "
                f"decoding tetrodes {trode_assignments[rank_ind]}"
            )
            self._send_interface.send_trode_selection(
                rank, trode_assignments[rank_ind]
            )

    def _activate_datastreams(self):
        """Tell all relevant processes to activate their
        neural data receivers"""

        self._send_interface.send_activate_datastream(
            self._config['rank']['ripples']
        )

        self._send_interface.send_activate_datastream(
            self._config['rank']['encoders']
        )

        self._send_interface.send_activate_datastream(
            self._config['rank']['decoders']
        )

    def _startup_rec_writers(self):
        """Notify processes to begin recording their data, if recording
        is enabled"""

        rs = [r for r in self._config['rank_settings']['enable_rec'] if r is not self._rank]

        self._send_interface.send_new_writer_message(
            rs, self._rec_manager.get_new_writer_message()
        )
        self._send_interface.send_start_rec_message(rs)

        # create stim decider writer and start its record writing locally
        if self._rank in self._config['rank_settings']['enable_rec']:
            self._stim_decider.set_record_writer_from_message(
                self._rec_manager.get_new_writer_message()
            )
        self._stim_decider.start_record_writing()

    def _sync_file_readers(self):
        """Synchronize other processes to a common time and tell them
        the valid range of data"""

        datasource = self._config['datasource']
        pos_data, lin_pos_data = trodes_file_sim.load_pos(
            self._config[datasource]['raw_dir'])

        # Give ample amount of time for other processes to synchronize
        start_time = time.time_ns() + 5e9
        first_timestamp = pos_data['time'][0]
        last_timestamp = pos_data['time'][-1]
        self._send_interface.send_synchronize_datastreams(
            self._config['rank']['ripples'],
            start_time,
            first_timestamp,
            last_timestamp
        )
        self._send_interface.send_synchronize_datastreams(
            self._config['rank']['encoders'],
            start_time,
            first_timestamp,
            last_timestamp
        )
        self._send_interface.send_synchronize_datastreams(
            self._config['rank']['decoders'],
            start_time,
            first_timestamp,
            last_timestamp
        )


    def _send_decoder_started(self):
        """Send a message signifying that the decoder system has started
        processing neural data"""

        self._send_interface.send_decoder_started()

    def _update_all_rank_setup_status(self, source_rank):
        """Update list keeping track of which ranks have registered
        the type of data they can record to disk"""

        self.class_log.debug(f"Record registration complete for rank {source_rank}")
        self._set_up_ranks.append(source_rank)
        if sorted(self._set_up_ranks) == self._ranks_sending_recs:
            self._all_ranks_set_up = True
            self.class_log.debug(
                f"Received from {self._set_up_ranks}, expected {self._ranks_sending_recs}")

    def finalize(self):
        """Final method called before exiting the main data processing loop"""

        self._stim_decider.stop_record_writing()

    @property
    def all_ranks_set_up(self):
        """Query whether all ranks have been set up"""

        return self._all_ranks_set_up


####################################################################################
# Processes
####################################################################################

class MainProcess(base.RealtimeProcess):
    """Top level object in main_process"""

    def __init__(
        self, comm, rank, config, stim_decider, network_client,
        main_manager, mpi_recv, ripple_recv, vel_pos_recv, posterior_recv,
        gui_params_recv
    ):

        super().__init__(comm, rank, config)

        if not isinstance(stim_decider, base.MessageHandler):
            raise TypeError(
                f"Custom 'stim_decider' expected to be of type "
                "realtime_decoder.base.MessageHandler, but got "
                f"type {type(stim_decider)}")

        self._network_client = network_client

        self._main_manager = main_manager
        
        self._mpi_recv = mpi_recv
        self._ripple_recv = ripple_recv
        self._vel_pos_recv = vel_pos_recv
        self._posterior_recv = posterior_recv
        self._gui_params_recv = gui_params_recv

        self._ranks_to_check = [r for r in range(comm.Get_size())]
        self._ranks_to_check.remove(self.rank)

        self.p = {}
        self.p['process_monitor'] = {}
        self.p['process_monitor']['interval'] = config['process_monitor']['interval']
        self.p['process_monitor']['timeout'] = config['process_monitor']['timeout']

    def main_loop(self):
        """Main data processing loop"""

        check_user_input = True

        try:
            last_time = time.time()
            while True:

                if (
                    self.p['process_monitor']['interval'] > 0 and
                    time.time() - last_time >= self.p['process_monitor']['interval']
                ):

                    is_ok, alive_ranks = self._check_processes()
                    # GUI case is a bit unique. Even if there is an error, the window
                    # won't be closed due to the Qt error handling. So we make sure to
                    # add it to the list of ranks that should be considered alive.
                    # is_ok is still "correct" though.
                    if not self.config['rank']['gui'][0] in alive_ranks:
                        alive_ranks.append(self.config['rank']['gui'][0])
                        alive_ranks = sorted(alive_ranks)
                    if not is_ok:
                        self.class_log.error(
                            f"Expected {self._ranks_to_check} to be alive but only "
                            f"{sorted(alive_ranks)} are, shutting down")
                        self._main_manager.terminate_remaining_processes(alive_ranks)

                    last_time = time.time()

                self._mpi_recv.receive()
                self._network_client.receive()
                self._ripple_recv.receive()
                self._vel_pos_recv.receive()
                self._posterior_recv.receive()
                self._gui_params_recv.receive()

                if check_user_input and self._main_manager.all_ranks_set_up:
                    print("***************************************", flush=True)
                    print("   All ranks are set up, ok to start   ", flush=True)
                    print("***************************************", flush=True)
                    self._main_manager.send_setup_complete()
                    self.class_log.debug("Notified GUI that setup was complete")
                    check_user_input = False


        except StopIteration as ex:
            self.class_log.info("Exiting normally")
        except Exception as e:
            self.class_log.exception(
                f"Main process exception occurred!"
            )
            self._main_manager.trigger_termination(
                raise_stop_iteration=False
            )

        self._main_manager.finalize()
        self.class_log.info("Exited main loop")

    def startup(self):
        """Initiate startup sequence for processing neural data"""

        self._main_manager.startup()

    def trigger_termination(self):
        """Initiate termination of all processes"""

        self._main_manager.trigger_termination()

    def _check_processes(self):
        """Check whether the other processes are still alive"""

        for rank in self._ranks_to_check:
            self.comm.send(
                obj=messages.VerifyStillAlive(),
                tag=messages.MPIMessageTag.COMMAND_MESSAGE,
                dest=rank
            )

        # we are only checking the tag; therefore it doesn't matter
        # what data we are receiving
        status = MPI.Status()
        req = self.comm.irecv(
            tag=messages.MPIMessageTag.PROCESS_IS_ALIVE
        )

        is_ok = False
        alive_ranks = []
        t = time.time()

        while time.time() - t < self.p['process_monitor']['timeout']:
            rdy, msg = req.test(status=status)
            if rdy:
                alive_ranks.append(status.source)
                if sorted(alive_ranks) == self._ranks_to_check:
                    is_ok = True
                    break # no need to wait until timeout
                req = self.comm.irecv(
                    tag=messages.MPIMessageTag.PROCESS_IS_ALIVE
                )

        return is_ok, alive_ranks


