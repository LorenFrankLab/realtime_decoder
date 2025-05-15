from realtime_decoder import (
    base, messages, main_process, ripple_process, encoder_process,
    decoder_process, gui_process, trodesnet, stimulation, datatypes,
    position, taskstate)


"""Creates different decoder objects. Note: By default these
are designed to work with Trodes"""

def create_main_process(comm, rank, config):

    # Set up the trodes client
    trodes_client = trodesnet.TrodesClient(config)

    # Set up the stim decider
    stim_decider = stimulation.TwoArmTrodesStimDecider(
        comm, rank, config, trodes_client)

    # The manager can function as a message handler
    main_manager = main_process.MainManager(
        rank, comm.Get_size(), config,
        main_process.MainMPISendInterface(comm, rank, config),
        stim_decider, manager_label='state')

    # Set up the interfaces
    mpi_recv = base.StandardMPIRecvInterface(
        comm, rank, config,
        messages.MPIMessageTag.COMMAND_MESSAGE,
        main_manager)

    ripple_recv = main_process.GenericMainRecvInterface(
        comm, rank, config,
        messages.get_dtype("Ripples"),
        messages.MPIMessageTag.RIPPLE_DETECTION,
        stim_decider)

    vel_pos_recv = main_process.GenericMainRecvInterface(
        comm, rank, config,
        messages.get_dtype("VelocityPosition"),
        messages.MPIMessageTag.VEL_POS,
        stim_decider)

    posterior_recv = main_process.GenericMainRecvInterface(
        comm, rank, config,
        messages.get_dtype("Posterior", config=config),
        messages.MPIMessageTag.POSTERIOR,
        stim_decider)

    gui_params_recv = base.StandardMPIRecvInterface(
        comm, rank, config,
        messages.MPIMessageTag.GUI_PARAMETERS,
        stim_decider)

    # Set up the main process
    process = main_process.MainProcess(
        comm, rank, config, stim_decider, trodes_client, main_manager,
        mpi_recv, ripple_recv, vel_pos_recv, posterior_recv, gui_params_recv)

    trodes_client.set_startup_callback(process.startup)
    trodes_client.set_termination_callback(process.trigger_termination)

    return process


def create_ripple_process(comm, rank, config):

    lfp_interface = trodesnet.TrodesDataReceiver(
        comm, rank, config, datatypes.Datatypes.LFP)

    pos_interface = trodesnet.TrodesDataReceiver(
        comm, rank, config, datatypes.Datatypes.LINEAR_POSITION)

    # The manager can function as a message handler
    ripple_manager = ripple_process.RippleManager(
        rank, config,
        ripple_process.RippleMPISendInterface(comm, rank, config),
        lfp_interface, pos_interface)

    mpi_recv = base.StandardMPIRecvInterface(
        comm, rank, config, messages.MPIMessageTag.COMMAND_MESSAGE,
        ripple_manager)

    gui_recv = base.StandardMPIRecvInterface(
        comm, rank, config, messages.MPIMessageTag.GUI_PARAMETERS,
        ripple_manager)

    process = ripple_process.RippleProcess(
        comm, rank, config, ripple_manager, mpi_recv, gui_recv)

    return process


def create_encoder_process(comm, rank, config):

    spikes_interface = trodesnet.TrodesDataReceiver(
        comm, rank, config, datatypes.Datatypes.SPIKES)

    pos_interface = trodesnet.TrodesDataReceiver(
        comm, rank, config, datatypes.Datatypes.LINEAR_POSITION)

    pos_mapper = position.TrodesPositionMapper(
        config['encoder']['position']['arm_ids'],
        config['encoder']['position']['arm_coords'])

    encoder_manager = encoder_process.EncoderManager(
        rank, config,
        encoder_process.EncoderMPISendInterface(comm, rank, config),
        spikes_interface, pos_interface, pos_mapper)

    mpi_recv = base.StandardMPIRecvInterface(
        comm, rank, config,
        messages.MPIMessageTag.COMMAND_MESSAGE,
        encoder_manager)

    gui_recv = base.StandardMPIRecvInterface(
        comm, rank, config,
        messages.MPIMessageTag.GUI_PARAMETERS,
        encoder_manager)

    process = encoder_process.EncoderProcess(
        comm, rank, config, encoder_manager, mpi_recv, gui_recv)

    return process

def create_decoder_process(comm, rank, config):

    pos_interface = trodesnet.TrodesDataReceiver(
        comm, rank, config, datatypes.Datatypes.LINEAR_POSITION)

    pos_mapper = position.TrodesPositionMapper(
        config['encoder']['position']['arm_ids'],
        config['encoder']['position']['arm_coords'])

    decoder_manager = decoder_process.DecoderManager(
        rank, config,
        decoder_process.DecoderMPISendInterface(comm, rank, config),
        decoder_process.SpikeRecvInterface(comm, rank, config), pos_interface,
        decoder_process.LFPTimeInterface(comm, rank, config), pos_mapper)

    mpi_recv = base.StandardMPIRecvInterface(
        comm, rank, config,
        messages.MPIMessageTag.COMMAND_MESSAGE,
        decoder_manager)

    gui_recv = base.StandardMPIRecvInterface(
        comm, rank, config,
        messages.MPIMessageTag.GUI_PARAMETERS,
        decoder_manager)

    process = decoder_process.DecoderProcess(
        comm, rank, config, decoder_manager, mpi_recv, gui_recv)

    return process

def create_gui_process(comm, rank, config):

    return gui_process.GuiProcess(comm, rank, config)