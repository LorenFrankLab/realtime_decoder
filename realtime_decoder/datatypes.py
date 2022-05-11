
import struct
from enum import IntEnum

from realtime_decoder import messages

class Datatypes(IntEnum):
    LFP = 1
    SPIKES = 2
    POSITION = 3
    LINEAR_POSITION = 4

# Data returned by DataSourceReceivers
class SpikePoint(messages.PrintableClass):

    def __init__(self, timestamp, elec_grp_id, data, systime, t_recv_data):

        self.timestamp = timestamp
        self.elec_grp_id = elec_grp_id
        self.data = data
        self.systime = systime
        self.t_recv_data = t_recv_data

class LFPPoint(messages.PrintableClass):

    def __init__(self, timestamp, elec_grp_ids, data, systime, t_recv_data):
        
        self.timestamp = timestamp
        self.elec_grp_ids = elec_grp_ids
        self.data = data
        self.systime = systime
        self.t_recv_data = t_recv_data

class CameraModulePoint(messages.PrintableClass):

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