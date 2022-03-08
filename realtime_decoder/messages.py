# Messages passed around between different processes
from enum import IntEnum
from typing import Sequence

import realtime_decoder.base as base

class MPIMessageTag(IntEnum):
    """Tags for messages passed between MPI processes
    """
    
    COMMAND_MESSAGE = 1
    FEEDBACK_DATA = 2
    TIMING_MESSAGE = 3

    SIMULATOR_LFP_DATA = 10
    SIMULATOR_SPK_DATA = 11
    SIMULATOR_POS_DATA = 12
    SIMULATOR_LINPOS_DATA = 13

    SPIKE_DECODE_DATA = 20
    STIM_DECISION = 21
    POSTERIOR = 22
    VEL_POS = 23

    GUI_POSTERIOR = 30
    GUI_ARM_EVENTS = 31
    GUI_REWARDS = 32
    GUI_DROPPED_SPIKES = 33
    GUI_COMMAND_MESSAGE = 34

class PrintableClass(object):
    """Contains repr to print out object so its attributes can be seen easily
    """
    
    def __repr__(self):
        return f'<{self.__class__.__name__}> at {hex(id(self))}>: {self.__dict__}'

class BinaryRecordCreate(PrintableClass):
    """Message used by BinaryRecordsManager to notify other classes
    of relevant information that will be used for writing binary
    records
    """
    
    def __init__(self, manager_label, file_id, save_dir, file_prefix,
        file_postfix, rec_label_dict, rec_format_dict):

        self.manager_label = manager_label
        self.file_id = file_id
        self.save_dir = save_dir
        self.file_prefix = file_prefix
        self.file_postfix = file_postfix
        self.rec_label_dict = rec_label_dict
        self.rec_format_dict = rec_format_dict


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

class ActivateDataStreams(PrintableClass):
    """Message used to tell data sources to be activated
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
    
    def __init__(self):
        pass

# class SpikePosJointProbMessage(PrintableMessage):
#     def __init__(
#         self, timestamp, elec_grp_id, current_pos, cred_int,
#         pos_hist, acq_send_time, send_time
#     ):
        
#         self.timestamp = timestamp
#         self.elec_grp_id = elec_grp_id
#         self.current_pos = current_pos
#         self.cred_int = cred_int
#         self.pos_hist = pos_hist
#         self.acq_send_time = acq_send_time
#         self.send_time = send_time

# class PosteriorMessage(PrintableMessage):

#     def __init__(
#         self, bin_timestamp, spike_timestamp, target, offtarget,
#         box, spike_count, crit_ind, posterior_max, rank, arm_posteriors,
#         tetrodes, lk_argmaxes
#     ):

#         self.bin_timestamp = bin_timestamp
#         self.spike_timestamp = spike_timestamp
#         self.target = target
#         self.offtarget = offtarget
#         self.box = box
#         self.spike_count = spike_count
#         self.crit_ind = crit_ind
#         self.posterior_max = posterior_max
#         self.rank = rank
#         self.arm_posteriors = arm_posteriors
#         self.tetrodes = tetrodes
#         self.lk_argmaxes = lk_argmaxes

# class VelocityPositionMessage(PrintableMessage):

#     def __init__(
#         self, bin_timestamp, raw_x, raw_y, raw_x2, raw_y2, angle,
#         angle_well_1, angle_well_2, pos, vel, rank
#     ):

#         self.bin_timestamp = bin_timestamp
#         self.raw_x = raw_x
#         self.raw_y = raw_y
#         self.raw_x2 = raw_x2
#         self.raw_y2 = raw_y2
#         self.angle = angle
#         self.angle_well_1 = angle_well_1
#         self.angle_well_2 = angle_well_2
#         self.pos = pos
#         self.vel = vel
#         self.rank = rank






# class RippleParameterMessage(rt_logging.PrintableMessage):
#     def __init__(self, rip_coeff1=1.2, rip_coeff2=0.2, ripple_threshold=5, baseline_window_timestamp=10000, n_above_thresh=1,
#                  lockout_time=7500, ripple_conditioning_lockout_time = 7500, posterior_lockout_time = 7500,
#                  detect_no_ripple_time=60000, dio_gate_port=None, detect_no_ripples=False,
#                  dio_gate=False, enabled=False, use_custom_baseline=False, update_custom_baseline=False):
#         self.rip_coeff1 = rip_coeff1
#         self.rip_coeff2 = rip_coeff2
#         self.ripple_threshold = ripple_threshold
#         self.baseline_window_timestamp = baseline_window_timestamp
#         self.n_above_thresh = n_above_thresh
#         self.lockout_time = lockout_time
#         self.ripple_conditioning_lockout_time = ripple_conditioning_lockout_time
#         self.posterior_lockout_time = posterior_lockout_time
#         self.detect_no_ripple_time = detect_no_ripple_time
#         self.dio_gate_port = dio_gate_port
#         self.detect_no_ripples = detect_no_ripples
#         self.dio_gate = dio_gate
#         self.enabled = enabled
#         self.use_custom_baseline = use_custom_baseline
#         self.update_custom_baseline = update_custom_baseline


# class CustomRippleBaselineMeanMessage(rt_logging.PrintableMessage):
#     def __init__(self, mean_dict):
#         self.mean_dict = mean_dict


# class CustomRippleBaselineStdMessage(rt_logging.PrintableMessage):
#     def __init__(self, std_dict):
#         self.std_dict = std_dict


# class RippleStatusDictListMessage(rt_logging.PrintableMessage):
#     def __init__(self, ripple_rank, status_dict_list):
#         self.ripple_rank = ripple_rank
#         self.status_dict_list = status_dict_list


# class RippleThresholdState(rt_logging.PrintableMessage):
#     """"Message containing whether or not at a given timestamp a ntrode's ripple filter threshold is crossed.

#     This message has helper serializer/deserializer functions to be used to speed transmission.
#     """
#     # MEC: in order to have different conditioning and regular ripple thresholds, add new conditioning state here
#     _byte_format = 'Iiii'

#     def __init__(self, timestamp, elec_grp_id, threshold_state, conditioning_thresh_state):
#         self.timestamp = timestamp
#         self.elec_grp_id = elec_grp_id
#         self.threshold_state = threshold_state
#         self.conditioning_thresh_state = conditioning_thresh_state

#     def pack(self):
#         return struct.pack(self._byte_format, self.timestamp, self.elec_grp_id,
#                            self.threshold_state, self.conditioning_thresh_state)

#     @classmethod
#     def unpack(cls, message_bytes):
#         timestamp, elec_grp_id, threshold_state, conditioning_thresh_state = struct.unpack(cls._byte_format, message_bytes)
#         return cls(timestamp=timestamp, elec_grp_id=elec_grp_id,
#                    threshold_state=threshold_state, conditioning_thresh_state=conditioning_thresh_state)


# class SpikeDecodeResultsMessage(realtime_logging.PrintableMessage):

#     _header_byte_fmt = '=qidqqi'
#     _header_byte_len = struct.calcsize(_header_byte_fmt)

#     def __init__(self, timestamp, elec_grp_id, current_pos, cred_int, pos_hist, send_time):
#         self.timestamp = timestamp
#         self.elec_grp_id = elec_grp_id
#         self.current_pos = current_pos
#         self.cred_int = cred_int
#         self.pos_hist = pos_hist
#         self.send_time = send_time

#     def pack(self):
#         pos_hist_len = len(self.pos_hist)
#         pos_hist_byte_len = pos_hist_len * struct.calcsize('=d')


#         message_bytes = struct.pack(self._header_byte_fmt,
#                                     self.timestamp,
#                                     self.elec_grp_id,
#                                     self.current_pos,
#                                     self.cred_int,
#                                     self.send_time,
#                                     pos_hist_byte_len)

#         message_bytes = message_bytes + self.pos_hist.tobytes()

#         return message_bytes

#     @classmethod
#     def unpack(cls, message_bytes):
#         timestamp, elec_grp_id, current_pos, cred_int, send_time, pos_hist_len = struct.unpack(cls._header_byte_fmt,
#                                                                     message_bytes[0:cls._header_byte_len])

#         pos_hist = np.frombuffer(message_bytes[cls._header_byte_len:cls._header_byte_len+pos_hist_len])

#         return cls(timestamp=timestamp, elec_grp_id=elec_grp_id,
#                    current_pos=current_pos, cred_int=cred_int, pos_hist=pos_hist, send_time=send_time)

# class PosteriorSum(rt_logging.PrintableMessage):
#     """"Message containing summed posterior from decoder_process.

#     This message has helper serializer/deserializer functions to be used to speed transmission.
#     """
#     _byte_format = 'IIdddddddddddiiiiiiiiiiiiiiiiiiiiiiii'

#     def __init__(
#         self, bin_timestamp, spike_timestamp, target, offtarget, box,
#         arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, spike_count,
#         crit_ind, posterior_max, rank,tet1,tet2,tet3,tet4,tet5,
#         tet6, tet7, tet8, tet9, tet10,
#         lk_argmax1, lk_argmax2, lk_argmax3, lk_argmax4, lk_argmax5,
#         lk_argmax6, lk_argmax7, lk_argmax8, lk_argmax9, lk_argmax10):
#         self.bin_timestamp = bin_timestamp
#         self.spike_timestamp = spike_timestamp
#         self.target = target
#         self.offtarget = offtarget
#         self.box = box
#         self.arm1 = arm1
#         self.arm2 = arm2
#         self.arm3 = arm3
#         self.arm4 = arm4
#         self.arm5 = arm5
#         self.arm6 = arm6
#         self.arm7 = arm7
#         self.arm8 = arm8
#         self.spike_count = spike_count
#         self.crit_ind = crit_ind
#         self.posterior_max = posterior_max
#         self.rank = rank
#         self.tet1 = tet1
#         self.tet2 = tet2
#         self.tet3 = tet3
#         self.tet4 = tet4
#         self.tet5 = tet5
#         self.tet6 = tet6
#         self.tet7 = tet7
#         self.tet8 = tet8
#         self.tet9 = tet9
#         self.tet10 = tet10
#         self.lk_argmax1 = lk_argmax1
#         self.lk_argmax2 = lk_argmax2
#         self.lk_argmax3 = lk_argmax3
#         self.lk_argmax4 = lk_argmax4
#         self.lk_argmax5 = lk_argmax5
#         self.lk_argmax6 = lk_argmax6
#         self.lk_argmax7 = lk_argmax7
#         self.lk_argmax8 = lk_argmax8
#         self.lk_argmax9 = lk_argmax9
#         self.lk_argmax10 = lk_argmax10


#     def pack(self):
#         return struct.pack(
#             self._byte_format, self.bin_timestamp, self.spike_timestamp, self.target, self.offtarget,
#             self.box, self.arm1, self.arm2, self.arm3, self.arm4, self.arm5, self.arm6, self.arm7,
#             self.arm8, self.spike_count, self.crit_ind, self.posterior_max, self.rank,
#             self.tet1,self.tet2,self.tet3,self.tet4,self.tet5,self.tet6, self.tet7, self.tet8, self.tet9,self.tet10,
#             self.lk_argmax1, self.lk_argmax2, self.lk_argmax3, self.lk_argmax4, self.lk_argmax5,
#             self.lk_argmax6, self.lk_argmax7, self.lk_argmax8, self.lk_argmax9, self.lk_argmax10)

#     @classmethod
#     def unpack(cls, message_bytes):
#         (bin_timestamp, spike_timestamp, target, offtarget, box,
#         arm1, arm2, arm3, arm4, arm5, arm6, arm7, arm8, spike_count, crit_ind,
#         posterior_max, rank,tet1,tet2,tet3,tet4,tet5,
#         tet6,tet7,tet8,tet9,tet10,
#         lk_argmax1, lk_argmax2, lk_argmax3, lk_argmax4, lk_argmax5,
#         lk_argmax6, lk_argmax7, lk_argmax8, lk_argmax9, lk_argmax10) = struct.unpack(cls._byte_format, message_bytes)

#         return cls(bin_timestamp=bin_timestamp, spike_timestamp=spike_timestamp, target=target, offtarget=offtarget,
#                 box=box, arm1=arm1, arm2=arm2, arm3=arm3, arm4=arm4, arm5=arm5, arm6=arm6, arm7=arm7, arm8=arm8, 
#                 spike_count=spike_count,crit_ind=crit_ind, posterior_max=posterior_max, rank=rank,
#                 tet1=tet1,tet2=tet2,tet3=tet3,tet4=tet4,tet5=tet5,
#                 tet6=tet6,tet7=tet7,tet8=tet8,tet9=tet9,tet10=tet10,
#                 lk_argmax1=lk_argmax1, lk_argmax2=lk_argmax2, lk_argmax3=lk_argmax3,
#                 lk_argmax4=lk_argmax4, lk_argmax5=lk_argmax5, lk_argmax6=lk_argmax6,
#                 lk_argmax7=lk_argmax7, lk_argmax8=lk_argmax8, lk_argmax9=lk_argmax9,
#                 lk_argmax10=lk_argmax10)


# class VelocityPosition(rt_logging.PrintableMessage):
#     """"Message containing velocity and linearized position from decoder_process.

#     This message has helper serializer/deserializer functions to be used to speed transmission.
#     """
#     _byte_format = 'Iiiiidddidi'

#     def __init__(self, bin_timestamp, raw_x, raw_y, raw_x2, raw_y2, angle, angle_well_1,
#                  angle_well_2, pos, vel, rank):
#         self.bin_timestamp = bin_timestamp
#         self.raw_x = raw_x
#         self.raw_y = raw_y
#         self.raw_x2 = raw_x2
#         self.raw_y2 = raw_y2
#         self.angle = angle
#         self.angle_well_1 = angle_well_1
#         self.angle_well_2 = angle_well_2
#         self.pos = pos
#         self.vel = vel
#         self.rank = rank

#     def pack(self):
#         return struct.pack(self._byte_format, self.bin_timestamp, self.raw_x, self.raw_y,
#                            self.raw_x2, self.raw_y2, self.angle, self.angle_well_1,
#                            self.angle_well_2, self.pos, self.vel,self.rank)

#     @classmethod
#     def unpack(cls, message_bytes):
#         bin_timestamp, raw_x, raw_y, raw_x2, raw_y2, angle, angle_well_1, angle_well_2, pos, vel, rank = struct.unpack(
#             cls._byte_format, message_bytes)
#         return cls(bin_timestamp=bin_timestamp, raw_x=raw_x, raw_y=raw_y,
#                    raw_x2=raw_x2, raw_y2=raw_y2, angle=angle, angle_well_1=angle_well_1,
#                    angle_well_2=angle_well_2, pos=pos, vel=vel, rank=rank)