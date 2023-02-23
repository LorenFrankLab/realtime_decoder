import time
import numpy as np

from typing import Callable
from zmq import ZMQError

from realtime_decoder import utils
from realtime_decoder.datatypes import Datatypes
from realtime_decoder.base import DataSourceReceiver
from realtime_decoder.datatypes import LFPPoint, SpikePoint, CameraModulePoint
from trodesnetwork.socket import SourceSubscriber
from trodesnetwork.trodes import TrodesAcquisitionSubscriber, TrodesHardware


class TrodesDataReceiver(DataSourceReceiver):

    def __init__(self, comm, rank, config, datatype):
        if not datatype in (
            Datatypes.LFP,
            Datatypes.SPIKES,
            Datatypes.LINEAR_POSITION
        ):
            raise TypeError(f"Invalid datatype {datatype}")
        super().__init__(comm, rank, config, datatype)

        self.sub_obj = None
        
        self.start = False
        self.stop = False

        self.ntrode_ids = [] # only applicable for spikes and LFP
        self.inds_to_extract = [] # only applicable for LFP
        self.sfact = self.config["trodes"]["voltage_scaling_factor"]

        self.temp_data = None

    def __next__(self):

        if not self.start:
            return None
        
        try:
            self.temp_data = self.sub_obj.receive(noblock=True)
            
            if self.datatype == Datatypes.LFP:
                
                datapoint = LFPPoint(
                    self.temp_data['localTimestamp'],
                    self.ntrode_ids,
                    (
                        np.array(self.temp_data['lfpData']) * self.sfact
                    )[self.inds_to_extract],
                    self.temp_data['systemTimestamp'],
                    time.time_ns())
                
                return datapoint

            elif self.datatype == Datatypes.SPIKES:
                
                ntid = self.temp_data['nTrodeId']
                if ntid in self.ntrode_ids:
                    
                    datapoint = SpikePoint(
                        self.temp_data['localTimestamp'],
                        ntid,
                        np.array(self.temp_data['samples']) * self.sfact,
                        self.temp_data['systemTimestamp'],
                        time.time_ns())
                    
                    return datapoint

                return None
            
            else:

                datapoint = CameraModulePoint(
                    self.temp_data['timestamp'],
                    self.temp_data['lineSegment'],
                    self.temp_data['posOnSegment'],
                    self.temp_data['x'],
                    self.temp_data['y'],
                    self.temp_data['x2'],
                    self.temp_data['y2'],
                    time.time_ns())
                
                return datapoint

        except ZMQError:
            return None

    def register_datatype_channel(self, channel):

        ntrode_id = channel
        if self.datatype in (Datatypes.LFP, Datatypes.SPIKES):
            if not ntrode_id in self.ntrode_ids:
                self.ntrode_ids.append(ntrode_id)
            else:
                self.class_log.debug(f"Already streaming from ntrode id {ntrode_id}")
        else:
            self.class_log.debug("Already set up to stream position, doing nothing")
            return
        
        if self.datatype == Datatypes.LFP:
            self.inds_to_extract = utils.get_ntrode_inds(
                self.config['trodes']['config_file'],
                self.ntrode_ids
            )

        self.class_log.debug(
            f"Set up to stream from ntrode ids {self.ntrode_ids}"
        )

    def activate(self):
        
        if self.datatype == Datatypes.LFP:
            name = 'source.lfp'
        elif self.datatype == Datatypes.SPIKES:
            name = 'source.waveforms'
        else:
            name = 'source.position'

        server_address = utils.get_network_address(
            self.config['trodes']['config_file']
        )
        if server_address is None:
            self.sub_obj = SourceSubscriber(name)
        else:
            self.sub_obj = SourceSubscriber(name, server_address=server_address)

        self.start = True
        self.class_log.debug(f"Datastream {name} activated")

    def deactivate(self):
        self.start = False

    def stop_iterator(self):
        raise StopIteration()


class TrodesClient(object):
    def __init__(self, config):
        self._startup_callback = utils.nop
        self._termination_callback = utils.nop

        server_address = utils.get_network_address(config['trodes']['config_file'])
        self._acq_sub = TrodesAcquisitionSubscriber(server_address=server_address)
        self._trodes_hardware = TrodesHardware(server_address=server_address)

    def send_statescript_shortcut_message(self, val):
        self._trodes_hardware.ecu_shortcut_message(val)

    def receive(self):
        try:
            data = self._acq_sub.receive(noblock=True)

            if ('play' in data['command'] or 'record' in data['command']):
                self._startup_callback()
            if 'stop' in data['command']: # 'stop' for playback, 'stoprecord' for recording
                self._termination_callback()
        except ZMQError:
            pass

    def set_startup_callback(self, callback:Callable):
        self._startup_callback = callback

    def set_termination_callback(self, callback:Callable):
        self._termination_callback = callback

