
import struct
from enum import IntEnum

from realtime_decoder import messages

"""Describes the type of external data (typically coming from data acquisition)
that can be processed by the system"""

class Datatypes(IntEnum):
    '''Numerical "tag" for external data that can be processed by the system'''
    LFP = 1
    SPIKES = 2
    POSITION = 3
    LINEAR_POSITION = 4 # not currently used

# Data returned by DataSourceReceivers
class SpikePoint(messages.PrintableClass):
    """Object describing a single spike event"""

    def __init__(self, timestamp, elec_grp_id, data, t_send_data, t_recv_data):

        self.timestamp = timestamp
        self.elec_grp_id = elec_grp_id
        self.data = data
        self.t_send_data = t_send_data
        self.t_recv_data = t_recv_data

class LFPPoint(messages.PrintableClass):
    """Object describing a single LFP data sample"""

    def __init__(self, timestamp, elec_grp_ids, data, t_send_data, t_recv_data):
        
        self.timestamp = timestamp
        self.elec_grp_ids = elec_grp_ids
        self.data = data
        self.t_send_data = t_send_data
        self.t_recv_data = t_recv_data

class CameraModulePoint(messages.PrintableClass):
    """Object describing a single data sample coming from a camera"""

    def __init__(
        self, timestamp, segment, position,
        x, y, x2, y2, t_recv_data
    ):

        self.timestamp = timestamp
        self.segment = segment
        self.position = position
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.t_recv_data = t_recv_data

##########################################################################
# These are not currently being used
class RawPosPoint(messages.PrintableClass):
    def __init__(self, timestamp, x1, y1, x2, y2, camera_id):
        self.timestamp = timestamp
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.camera_id = camera_id


class PosPoint(messages.PrintableClass):
    def __init__(self, timestamp, x, y, camera_id):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.camera_id = camera_id


class DigIOStateChange(messages.PrintableClass):
    def __init__(self, timestamp, port, io_dir, state):
        self.timestamp = timestamp
        self.port = port
        self.io_dir = io_dir        # 1 - input, 0 - output
        self.state = state


class SystemTimePoint(messages.PrintableClass):
    def __init__(self, timestamp, tv_sec, tv_nsec):
        self.timestamp = timestamp
        self.tv_sec = tv_sec
        self.tv_nsec = tv_nsec
##########################################################################