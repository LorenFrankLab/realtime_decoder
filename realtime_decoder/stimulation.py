from realtime_decoder import base, utils, messages

class TrodesStimDecider(base.BinaryRecordBase, base.MessageHandler):

    def __init__(self, rank, config, trodes_client):
        super().__init__(
            rank=rank
        )
        self._trodes_client = trodes_client

    def handle_message(self, msg, mpi_status):

        # feedback, velocity/position, posterior
        if isinstance(msg, messages.GuiMainParameters):
            self._update_gui_params(msg)
        elif mpi_status.tag == messages.MPIMessageTag.RIPPLE_DETECTION:
            self._update_ripples(msg)
        elif mpi_status.tag == messages.MPIMessageTag.VEL_POS:
            self._update_velocity_position(msg)
        elif mpi_status.tag == messages.MPIMessageTag.POSTERIOR:
            self._update_posterior(msg)
        else:
            self._class_log.warning(
                f"Received message of type {type(msg)} "
                f"from source: {mpi_status.source}, "
                f" tag: {mpi_status.tag}, ignoring"
            )

    def _update_gui_params(self, gui_msg):
        self.class_log.info("Updating GUI main parameters")

    def _update_ripples(self, msg):
        pass

    def _update_velocity_position(self, msg):
        pass

    def _update_posterior(self, msg):
        pass