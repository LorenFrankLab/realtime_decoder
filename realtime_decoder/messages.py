# Messages passed around between different processes
import numpy as np

from enum import IntEnum
from typing import Sequence, Dict

class MPIMessageTag(IntEnum):
    """Tags for messages passed between MPI processes
    """
    
    COMMAND_MESSAGE = 1
    FEEDBACK_DATA = 2
    TIMING_MESSAGE = 3
    PROCESS_IS_ALIVE = 4
    GUI_COMMAND_MESSAGE = 5
    GUI_PARAMETERS = 6

    SIMULATOR_LFP_DATA = 10
    SIMULATOR_SPK_DATA = 11
    SIMULATOR_POS_DATA = 12
    SIMULATOR_LINPOS_DATA = 13

    SPIKE_DECODE_DATA = 20
    STIM_DECISION = 21
    POSTERIOR = 22
    VEL_POS = 23
    LFP_TIMESTAMP = 24
    RIPPLE_DETECTION = 25

    ARM_EVENTS = 30
    REWARDS = 31
    DROPPED_SPIKES = 32

class PrintableClass(object):
    """Contains repr to print out object so its attributes can be seen easily
    """
    
    def __repr__(self):
        return f'<{self.__class__.__name__} at {hex(id(self))}>: {self.__dict__}'

class BinaryRecordCreate(PrintableClass):
    """Message used by BinaryRecordsManager to notify other classes
    of relevant information that will be used for writing binary
    records
    """
    
    def __init__(self, manager_label, file_id, save_dir, file_prefix,
        file_postfix, rec_label_dict, rec_format_dict, num_digits):

        self.manager_label = manager_label
        self.file_id = file_id
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.file_postfix = file_postfix
        self.rec_label_dict = rec_label_dict
        self.rec_format_dict = rec_format_dict
        self.num_digits = num_digits


class BinaryRecordType(PrintableClass):
    """A class containing relevant information for a particular
    binary record type
    """
    
    def __init__(self, manager_label, rec_id, rec_labels, rec_struct_fmt):
        self.manager_label = manager_label
        self.rec_id = rec_id
        self.rec_labels = rec_labels
        self.rec_struct_fmt = rec_struct_fmt

class BinaryRecordSendComplete(PrintableClass):
    """Signifies to the main process that all binary record
    registration messages have been sent
    """
    
    def __init__(self):
        pass

class TrodeSelection(PrintableClass):
    """Tells a process which trodes should be managed

    Init parameters
    ---------------
    trodes : Sequence of type int
        The trodes that will be managed
    """
    
    def __init__(self, trodes:Sequence[int]):
        self.trodes = trodes

class StartRecordMessage(PrintableClass):
    """Tells a process to start record writing
    """

    def __init__(self):
        pass

class ActivateDataStreams(PrintableClass):
    """Message used to tell data sources to be activated
    """
    
    def __init__(self):
        pass

class StartupSignal(PrintableClass):
    """Control signal sent by GUI to initiate startup
    """

    def __init__(self):
        pass

class TerminateSignal(PrintableClass):
    """Communicates that a process should be terminated
    """
    
    def __init__(self):
        pass

class SetupComplete(PrintableClass):
    """Communicates that setup is complete for all processes
    """
    
    def __init__(self, *, data=None):
        self.data = data

class DecoderStarted(PrintableClass):
    """Tells GUI that decoder has successfully started up
    """

    def __init__(self):
        pass

class VerifyStillAlive(PrintableClass):
    """Message sent out by the main process to other processes
    to check whether they are still aliive (not in an error state)
    """

    def __init__(self):
        pass

class GuiMainParameters(PrintableClass):
    
    def __init__(
        self, posterior_threshold:float, max_center_well_distance:float,
        num_above_threshold:int, min_duration:float, well_angle_range:float,
        within_angle_range:float, shortcut_message_on:bool
    ):

        self.posterior_threshold = posterior_threshold
        self.max_center_well_distance = max_center_well_distance
        self.num_above_threshold = num_above_threshold
        self.min_duration = min_duration
        self.well_angle_range = well_angle_range
        self.within_angle_range = within_angle_range
        self.shortcut_message_on = shortcut_message_on

class GuiRippleParameters(PrintableClass):
    
    def __init__(
        self, velocity_threshold:float, ripple_threshold:float,
        conditioning_ripple_threshold:float,
        content_ripple_threshold:float, end_ripple_threshold:float,
        freeze_stats:bool
    ):
        self.velocity_threshold = velocity_threshold
        self.ripple_threshold = ripple_threshold
        self.conditioning_ripple_threshold = conditioning_ripple_threshold
        self.content_ripple_threshold = content_ripple_threshold
        self.end_ripple_threshold = end_ripple_threshold
        self.freeze_stats = freeze_stats

class GuiEncodingModelParameters(PrintableClass):
    
    def __init__(self, encoding_velocity_threshold:float, freeze_model:bool):
        self.encoding_velocity_threshold = encoding_velocity_threshold
        self.freeze_model = freeze_model


def get_dtype(msg_type:str, *, config:Dict={}):

    # Note: there might be an issue if we're passing data
    # between machines with different native byte order
    
    if msg_type == "Ripples":
        dt = np.dtype([
            ('timestamp', '=i8'),
            ('elec_grp_id', '=i4'),
            ('ripple_type', '=U10'),
            ('is_consensus', '=?')
        ])
    elif msg_type == "SpikePosJointProb":
        num_bins = config['encoder']['position']['num_bins']
        dt = np.dtype([
            ('timestamp', '=i8'),
            ('elec_grp_id', '=i4'),
            ('current_pos', '=f8'),
            ('cred_int', '=i8'),
            ('send_time', '=i8'),
            ('hist', '=f8', (num_bins, ))
        ])
    elif msg_type == "Posterior":
        num_bins = config['encoder']['position']['num_bins']
        algorithm = config['algorithm']
        num_states = len(config[algorithm]['state_labels'])
        num_buff = config['decoder']['cred_int_bufsize']
        dt = np.dtype([
            ('rank', '=i4'),
            ('lfp_timestamp', '=i8'),
            ('bin_timestamp_l', '=i8'),
            ('bin_timestamp_r', '=i8'),
            ('posterior', '=f8', (num_states, num_bins)),
            ('likelihood', '=f8', (num_bins, )),
            ('velocity', '=f8'),
            ('cred_int_post', '=i8'),
            ('cred_int_lk', '=i8'),
            ('enc_cred_intervals', '=f8', (num_buff, )),
            ('enc_argmaxes', '=f8', (num_buff, )),
            ('spike_count', '=i8')
        ])
    elif msg_type == "VelocityPosition":
        dt = np.dtype([
            ('rank', '=i4'),
            ('timestamp', '=i8'),
            ('segment', '=i4'),
            ('raw_x', '=f8'),
            ('raw_y', '=f8'),
            ('raw_x2', '=f8'),
            ('raw_y2', '=f8'),
            ('mapped_pos', '=f8'),
            ('velocity', '=f8')
        ])
    elif msg_type == "DroppedSpikes":
        dt = np.dtype([
            ('rank', '=i4'),
            ('pct', '=f4')
        ])
    else:
        raise ValueError(f"Unknown message type {msg_type}")

    return dt