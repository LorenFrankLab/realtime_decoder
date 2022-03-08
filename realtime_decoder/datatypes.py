
import struct
from enum import IntEnum

import realtime_decoder.messages as messages


class Datatypes(IntEnum):
    LFP = 1
    SPIKES = 2
    POSITION = 3
    LINEAR_POSITION = 4


class SpikePoint(messages.PrintableClass):
    """
    Spike data message.
    """
    _byte_format = '=qi40h40h40h40h'

    def __init__(self, timestamp, elec_grp_id, data):
        self.timestamp = timestamp
        self.elec_grp_id = elec_grp_id
        self.data = data

    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.elec_grp_id,
                           *self.data[0], *self.data[1], *self.data[2], *self.data[3])

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, elec_grp_id, *raw_data = struct.unpack(cls._byte_format, message_bytes)
        return cls(timestamp=timestamp, elec_grp_id=elec_grp_id, data=[raw_data[0:40],
                                                                                     raw_data[40:80],
                                                                                     raw_data[80:120],
                                                                                     raw_data[120:160]])

    @classmethod
    def packed_message_size(cls):
        return struct.calcsize(cls._byte_format)


class LFPPoint(messages.PrintableClass):
    _byte_format = '=qiii'

    def __init__(self, timestamp, ntrode_index, elec_grp_id, data):
        self.timestamp = timestamp
        self.ntrode_index = ntrode_index
        self.elec_grp_id = elec_grp_id
        self.data = data

    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.ntrode_index, self.elec_grp_id, self.data)

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, ntrode_index, elec_grp_id, data = struct.unpack(cls._byte_format, message_bytes)
        message = cls(timestamp=timestamp, ntrode_index=elec_grp_id,
                      elec_grp_id=elec_grp_id, data=data)
        return message

    @classmethod
    def packed_message_size(cls):
        return struct.calcsize(cls._byte_format)


class LinearPosPoint(messages.PrintableClass):
    _byte_format = '=qff'

    def __init__(self, timestamp, x, vel):
        self.timestamp = timestamp
        self.x = x
        self.vel = vel

    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.x, self.vel)

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, x, vel = struct.unpack(cls._byte_format, message_bytes)
        message = cls(timestamp=timestamp, x=x, vel=vel)
        return message

    @classmethod
    def packed_message_size(cls):
        return struct.calcsize(cls._byte_format)

class CameraModulePoint(messages.PrintableClass):
    _byte_format = '=qidii'

    def __init__(self, timestamp, segment, position, x, y, x2, y2):
        self.timestamp = timestamp
        self.segment = segment
        self.position = position
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2

    # we never actually use the pack and unpack methods. Remove in the next version of the decoder
    def pack(self):
        return struct.pack(self._byte_format, self.timestamp, self.segment, self.position, self.x, self.y)

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, segment, position, x, y = struct.unpack(cls._byte_format, message_bytes)
        message = cls(timestamp=timestamp, segment=segment, position=position, x=x, y=y)
        return message

    @classmethod
    def packed_message_size(cls):
        return struct.calcsize(cls._byte_format)




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
