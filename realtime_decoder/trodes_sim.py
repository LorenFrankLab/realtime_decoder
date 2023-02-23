import math
import time
import pathlib
import os
import glob
import zmq
import msgpack
import numpy as np

from operator import attrgetter
from abc import ABCMeta
from zmq import ZMQError

from realtime_decoder import base, logging_base, trodes, utils, messages
from realtime_decoder.datatypes import (
    Datatypes, LFPPoint, SpikePoint, CameraModulePoint
)

class TrodesSpikeSample(messages.PrintableClass):

    def __init__(self, elec_grp_id, timestamp, data):
        self.elec_grp_id = elec_grp_id
        self.timestamp = timestamp
        self.data = data

class TrodesSimSubscriber(logging_base.LoggingClass):

    def __init__(self, address):
        self._conn_address = address
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.connect(address)
        self._sock.setsockopt(zmq.SUBSCRIBE, b'') # all topics

    def receive(self, *, noblock=False):
        flags = zmq.NOBLOCK if noblock else zmq.NULL
        data_msg = self._sock.recv(flags=flags)
        return msgpack.unpackb(data_msg, raw=False)

    def __del__(self):
        self._sock.disconnect(self._conn_address)


class TrodesSimReceiver(base.DataSourceReceiver):

    # based off of TrodesDataReceiver in trodesnet.py
    # should probably refactor into factory method
    # for the sub_obj member variable

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
            recfile = utils.find_unique_file(
                os.path.join(
                    self.config['trodes_simulator']['raw_dir'], 
                    "*.rec"
                ),
                "rec"
            )
            self.inds_to_extract = utils.get_ntrode_inds(recfile, self.ntrode_ids)

        self.class_log.debug(
            f"Set up to stream from ntrode ids {self.ntrode_ids}"
        )

    def activate(self):
        
        address = "tcp://127.0.0.1:"
        if self.datatype == Datatypes.LFP:
            name = 'LFP sim'
            address += f"{self.config['trodes_simulator']['lfp_port']}"
        elif self.datatype == Datatypes.SPIKES:
            name = 'spikes sim'
            address += f"{self.config['trodes_simulator']['spikes_port']}"
        else:
            name = 'pos sim'
            address += f"{self.config['trodes_simulator']['pos_port']}"

        self.class_log.debug(f"Connecting to {address}")
        self.sub_obj = TrodesSimSubscriber(address)

        self.start = True
        self.class_log.debug(f"Datastream {name} activated")

    def deactivate(self):
        self.start = False

    def stop_iterator(self):
        raise StopIteration()


class TrodesSimServer(logging_base.LoggingClass):

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._spike_data = {}
        self._lfp_data = {}
        self._pos_data = {}
        self._setup_sockets()

    def _setup_sockets(self):
        self._ctx = zmq.Context.instance()

        host = "tcp://127.0.0.1"

        port = self._config['trodes_simulator']['spikes_port'] 
        self._spike_sock = self._ctx.socket(zmq.PUB)
        self._spike_sock.bind(f"{host}:{port}")
        self.class_log.debug(
            f"Spikes socket connected to "
            f"{self._spike_sock.get_string(zmq.LAST_ENDPOINT)}"
        )

        port = self._config['trodes_simulator']['lfp_port']
        self._lfp_sock = self._ctx.socket(zmq.PUB)
        self._lfp_sock.bind(f"{host}:{port}")
        self.class_log.debug(
            f"LFP socket connected to "
            f"{self._lfp_sock.get_string(zmq.LAST_ENDPOINT)}"
        )
        
        port = self._config['trodes_simulator']['pos_port']
        self._pos_sock = self._ctx.socket(zmq.PUB)
        self._pos_sock.bind(f"{host}:{port}")
        self.class_log.debug(
            f"Pos socket connected to "
            f"{self._pos_sock.get_string(zmq.LAST_ENDPOINT)}"
        )

    def send_spike_data(self, spike_obj:TrodesSpikeSample):
        # make sure all quantities are converted into Python
        # native types
        self._spike_data['localTimestamp'] = int(spike_obj.timestamp)
        self._spike_data['systemTimestamp'] = time.time_ns()
        self._spike_data['nTrodeId'] = int(spike_obj.elec_grp_id)
        self._spike_data['samples'] = spike_obj.data.tolist()

        print(f"Sending spike ntrode {int(spike_obj.elec_grp_id)}, timestamp {int(spike_obj.timestamp)}")

        self._spike_sock.send(msgpack.packb(self._spike_data))

    def send_lfp_data(self, lfp_arr:np.ndarray):
        self._lfp_data['localTimestamp'] = int(lfp_arr['time'])
        self._lfp_data['systemTimestamp'] = time.time_ns()
        self._lfp_data['lfpData'] = lfp_arr['lfp'].tolist()

        self._lfp_sock.send(msgpack.packb(self._lfp_data))

    def send_pos_data(self, pos_arr:np.ndarray):

        self._pos_data['timestamp'] = int(pos_arr['time'])
        self._pos_data['lineSegment'] = int(pos_arr['segment'])
        self._pos_data['posOnSegment'] = int(pos_arr['segment_pos'])
        self._pos_data['x'] = int(pos_arr['xloc'])
        self._pos_data['y'] = int(pos_arr['yloc'])
        self._pos_data['x2'] = int(pos_arr['xloc2'])
        self._pos_data['y2'] = int(pos_arr['yloc2'])

        self._pos_sock.send(msgpack.packb(self._pos_data))


class TrodesClientStub(object):

    def __init__(self, config):
        pass

    def send_statescript_shortcut_message(self, val):
        pass

    def receive(self):
        pass


class TrodesFileReader(logging_base.LoggingClass, metaclass=ABCMeta):

    def __init__(self, config):
        super().__init__()
        self.config = config


class TrodesPosReader(TrodesFileReader):
    
    def __init__(self, config):
        super().__init__(config)
        self._data = None # when populated, will be numpy record array
        self._curr_ind = 0
        self._pos_arr = None

    def _load_pos(self):

        self.class_log.info("Loading position data")

        pos_file = utils.find_unique_file(
            os.path.join(
                self.config['trodes_simulator']['raw_dir'],
                "*.videoPositionTracking"
            ),
            "position tracking"
        )
        pos_data, _ = trodes.load_dat_file(
            pos_file
        )

        linear_tracking_file = utils.find_unique_file(
            os.path.join(
                self.config['trodes_simulator']['raw_dir'],
                "*.videoLinearTracking"
            ),
            "linear tracking"
        )
        linear_pos_data, _ = trodes.load_dat_file(
            linear_tracking_file,
            dtype_fields=['time', 'LineSegment', 'RelativeLinearPos']
        )

        self._data = self._combine_pos_data(pos_data, linear_pos_data)

    def _combine_pos_data(self, pos_data, linear_pos_data):

        ts, pos_inds, linear_pos_inds = np.intersect1d(
            pos_data['time'], linear_pos_data['time'],
            return_indices=True
        )

        dt = np.dtype([
            ('time', pos_data['time'].dtype),
            ('segment', linear_pos_data['LineSegment'].dtype),
            ('segment_pos', linear_pos_data['RelativeLinearPos'].dtype),
            ('xloc', pos_data['xloc'].dtype),
            ('yloc', pos_data['yloc'].dtype),
            ('xloc2', pos_data['xloc2'].dtype),
            ('yloc2', pos_data['yloc2'].dtype)
        ])

        combined_data = np.zeros(len(ts), dtype=dt)
        combined_data['time'] = ts
        combined_data['segment'] = linear_pos_data['LineSegment'][linear_pos_inds]
        combined_data['segment_pos'] = linear_pos_data['RelativeLinearPos'][linear_pos_inds]
        combined_data['xloc'] = pos_data['xloc'][pos_inds]
        combined_data['yloc'] = pos_data['yloc'][pos_inds]
        combined_data['xloc2'] = pos_data['xloc2'][pos_inds]
        combined_data['yloc2'] = pos_data['yloc2'][pos_inds]

        return combined_data

    def load_data(self):
        self._load_pos()

    def get_first_timestamp(self):

        return self._data['time'][0]

    def set_start_point(self, timestamp):

        self._curr_ind = np.argwhere(
            self._data['time'] >= timestamp
        ).squeeze()[0]

    def get_data_at_timestamp(self, ts):

        try:
            if self._data[self._curr_ind]['time'] == ts:
                self._pos_arr = self._data[self._curr_ind]
                self._curr_ind += 1
                return self._pos_arr
            else:
                return None
        except IndexError:
            raise StopIteration()


class TrodesSpikesReader(TrodesFileReader):

    def __init__(self, config):
        super().__init__(config)
        self._curr_ind = 0
        self._spike_datas = [] # list of TrodesSpikeSample objects
        self._spike_obj = None

    def _load_spikes(self):

        self.class_log.info(
            "Loading spikes from ntrodes "
            f"{self.config['trode_selection']['decoding']}..."
        )

        # the reason we store spikes as a list of objects is because
        # we don't know the dimensionality of each tetrode
        spike_datas = []
        for ntid in self.config['trode_selection']['decoding']:
            spikes_file = utils.find_unique_file(
                os.path.join(
                    self.config['trodes_simulator']['spikes_dir'],
                    f"*nt{ntid}.dat"
                ),
                "spikes"
            )

            # since we don't know how many channels each ntrode has, we only check
            # for the first channel (the smallest an ntrode can be)
            data, _ = trodes.load_dat_file(
                spikes_file, dtype_fields=['time', 'waveformCh1']
            )

            # reshape data into (n_samples, n_channels, n_points_per_channel)
            reshaped_spike_data = self._reshape_spikes(data)
            assert len(data['time']) == reshaped_spike_data.shape[0]
            self.class_log.debug(f"ntrode {ntid}: {reshaped_spike_data.shape}")

            # unclear how bad performance will be with say, 100s of
            # millions of spikes. if it becomes an issue, should look into a
            # better way of combining data to be subsequently sorted by
            # timestamp
            for ts, snippet in zip(data['time'], reshaped_spike_data):
                spike_datas.append(TrodesSpikeSample(ntid, ts, snippet))

        spike_datas = sorted(spike_datas, key=attrgetter('timestamp'))
        return spike_datas

    def _reshape_spikes(self, spike_data):

        fields = spike_data.dtype.fields
        fields_to_use = [f for f in fields if 'waveform' in f]

        num_samples = len(spike_data['time'])
        num_channels = len(fields_to_use)
        num_points_per_channel = spike_data['waveformCh1'].shape[1]

        output = np.zeros(
            (num_samples, num_channels, num_points_per_channel),
            dtype=spike_data['waveformCh1'].dtype
        )

        for ii, name in enumerate(fields_to_use):
            output[:, ii, :] = spike_data[name]
        # output *= self.config['trodes']['voltage_scaling_factor']

        return output

    def load_data(self):
        self._spike_datas = self._load_spikes()
        self.class_log.info(
            f"Total number of spikes: {len(self._spike_datas)}"
        )

    def set_start_point(self, timestamp):

        # we assume timestamp is relatively close to the first
        # timestamp in our list of spike objects. otherwise
        # this is a horrible way of setting the index and
        # this method could take a long time to complete
        ind = 0
        for ii, obj in enumerate(self._spike_datas):
            if obj.timestamp >= timestamp:
                self._curr_ind = ii
                break

    def get_data_at_timestamp(self, ts):

        try:
            if self._spike_datas[self._curr_ind].timestamp == ts:
                self._spike_obj = self._spike_datas[self._curr_ind]
                self._curr_ind += 1
                return self._spike_obj
            else:
                return None
        except IndexError:
            return None


class TrodesLFPReader(TrodesFileReader):

    def __init__(self, config):
        super().__init__(config)
        self._data = None # when populated, will be a numpy record array
        self._curr_ind = 0
        self._lfp_arr = None

    def _load_timestamps(self):
        ts_file = utils.find_unique_file(
            os.path.join(
                self.config['trodes_simulator']['lfp_dir'],
                '*timestamps.dat'
            ),
            "lfp timestamps"
        )
        data, _ = trodes.load_dat_file(ts_file, dtype_fields=['time'])

        return data['time']

    def _load_lfp(self, timestamps):

        recfile = utils.find_unique_file(
            os.path.join(
                self.config['trodes_simulator']['raw_dir'], 
                "*.rec"
            ),
            "rec"
        )
        root = utils.get_xml_root(recfile)
        num_ntrodes = len(
            root.find("SpikeConfiguration").findall("SpikeNTrode")
        )
        num_samples = len(timestamps)

        # array of floats, need this datatype since will multiply
        # int16's by the decimal conversion factor to get uV
        # lfp_datas = np.zeros((num_samples, num_ntrodes))

        # dt = np.dtype([('timestamp', ts.dtype), ('lfp', )])

        # self._data_dict['lfp'] = np.zeros(
        #     (len(self._data_dict['timestamps'], num_ntrodes))
        # )

        for ii, ntrode in enumerate(root.iter("SpikeNTrode")):
            ntid = int(ntrode.get("id"))
            self.class_log.debug(f"Loading LFP from ntrode {ntid}")
            lfp_file = utils.find_unique_file(
                os.path.join(
                    self.config['trodes_simulator']['lfp_dir'],
                    f'*nt{ntid}ch*'
                ),
                "lfp"
            )

            data, _ = trodes.load_dat_file(
                lfp_file, dtype_fields=['voltage']
            )

            # now we know what the datatype should be, so we can define
            # and pre-allocate lfp_datas properly
            if ii == 0:
                lfp_datas = np.zeros(
                    (num_samples, num_ntrodes),
                    dtype=data['voltage'].dtype
                )

            assert lfp_datas.dtype == data['voltage'].dtype
            lfp_datas[:, ii] = data['voltage']

        # lfp_datas *= self.config['trodes']['voltage_scaling_factor']

        return lfp_datas

    def load_data(self):
        ts = self._load_timestamps()
        lfp_datas = self._load_lfp(ts)

        assert ts.shape[0] == lfp_datas.shape[0]

        dt = np.dtype([
            ('time', ts.dtype),
            ('lfp', lfp_datas.dtype, (lfp_datas.shape[1], ))
        ])

        self._data = np.zeros(len(ts), dtype=dt)
        self._data['time'] = ts
        self._data['lfp'] = lfp_datas

    def set_start_point(self, timestamp):

        self._curr_ind = np.argwhere(
            self._data['time'] >= timestamp
        ).squeeze()[0]

    def get_data_at_timestamp(self, ts):

        try:
            if self._data[self._curr_ind]['time'] == ts:
                self._lfp_arr = self._data[self._curr_ind]
                self._curr_ind += 1
                return self._lfp_arr
            else:
                return None
        except IndexError:
            return None


class TrodesSimManager(base.MessageHandler):

    def __init__(self, rank, config, send_interface, sim_server):
        super().__init__()
        self._config = config
        self._send_interface = send_interface
        self._sim_server = sim_server

        self._spikes_reader = TrodesSpikesReader(config)
        self._lfp_reader = TrodesLFPReader(config)
        self._pos_reader = TrodesPosReader(config)

        self._sample_rate = self._config['trodes_simulator']['desired_sampling_rate']
        self._offset = 0
        
        self._sample_num = 0
        self._num_samples_processed = 0

        self._start_time = 0
        self._curr_time = 0
        self._switch_time = 0

        self._send_raw_data = False

        # delete when ready
        self._t0 = 0

    def handle_message(self, msg, mpi_status):

        source_rank = mpi_status.source
        if isinstance(msg, messages.StartupSignal):
            self.class_log.info(
                f"Received StartupSignal from rank {source_rank}"
            )
            self._startup()
        elif isinstance(msg, messages.VerifyStillAlive):
            self._send_interface.send_alive_message()
        elif isinstance(msg, messages.TerminateSignal):
            self.class_log.info(f"Got terminate signal from rank {source_rank}")
            raise StopIteration()
        else:
            self.class_log.warning(
                f"Unrecognized command message of type {type(msg)}, ignoring")

    def _startup(self):

        self._spikes_reader.load_data()
        self._lfp_reader.load_data()
        self._pos_reader.load_data()

        self._offset = self._pos_reader.get_first_timestamp()
        self._switch_time = self._get_task_file_info()
        self._set_task_file(self._offset, self._switch_time)

        self._spikes_reader.set_start_point(self._offset)
        self._lfp_reader.set_start_point(self._offset)

        self._start_time = time.time_ns()
        self._curr_time = time.time_ns()

        self._send_raw_data = True

        # delete when ready
        self._t0 = self._start_time

    def _get_task_file_info(self):

        statescript_log_file = utils.find_unique_file(
            os.path.join(
                self._config['trodes_simulator']['raw_dir'], 
                "*.stateScriptLog"
            ),
            "state script log"
        )
        t = utils.get_switch_time(statescript_log_file)

        recfile = utils.find_unique_file(
            os.path.join(
                self._config['trodes_simulator']['raw_dir'], 
                "*.rec"
            ),
            "rec"
        )
        root = utils.get_xml_root(recfile)
        sr = int(root.find("HardwareConfiguration").attrib['samplingRate'])

        # convert milliseconds to sample number
        return int(t / 1000 * sr)

    def _set_task_file(self, offset, switch_time):

        taskfile = self._config['trodes']['taskstate_file']

        if offset < switch_time:
            val = 1
        else:
            val = 2
            self.class_log.warning(
                "Task state is already 2. Are you sure the configuration/data "
                "are correct?"
            )

        utils.write_text_file(taskfile, val)

    def next_iter(self):

        if self._send_raw_data:
            self._curr_time = time.time_ns()

            # round down fractional samples
            num_expected = math.floor(
                (self._curr_time - self._start_time)/1e9 * self._sample_rate
            )

            num_to_process = num_expected - self._num_samples_processed
            for n in range(num_to_process):
                simulated_ts = self._num_samples_processed + n + self._offset
                self._send_data_if_ready(simulated_ts)
                self._switch_task_if_ready(simulated_ts)

            self._num_samples_processed += num_to_process

            # delete when ready
            # if self._num_samples_processed % 30000 == 0:
            #     self.class_log.info(
            #         f"Processed {self._num_samples_processed}, "
            #         f"elapsed time: {(time.time_ns() - self._t0)/1e9} seconds"
            #     )

    def _send_data_if_ready(self, simulated_ts):

        pos_obj = self._pos_reader.get_data_at_timestamp(
            simulated_ts
        )
        if pos_obj is not None:
            self._sim_server.send_pos_data(pos_obj)

        spikes_obj = self._spikes_reader.get_data_at_timestamp(
            simulated_ts
        )
        if spikes_obj is not None:
            self._sim_server.send_spike_data(spikes_obj)

        lfp_obj = self._lfp_reader.get_data_at_timestamp(
            simulated_ts
        )
        if lfp_obj is not None:
            self._sim_server.send_lfp_data(lfp_obj)

    def _switch_task_if_ready(self, simulated_ts):

        if self._switch_time == simulated_ts:
            taskfile = self._config['trodes']['taskstate_file']
            utils.write_text_file(taskfile, 2)
            self.class_log.info("Setting task state to 2")


class TrodesSimProcess(base.RealtimeProcess):

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

        self._trodes_sim_manager = TrodesSimManager(
            rank, config, base.StandardMPISendInterface(comm, rank, config),
            TrodesSimServer(config)
        )

        self._mpi_recv = base.StandardMPIRecvInterface(
            comm, rank, config, messages.MPIMessageTag.COMMAND_MESSAGE,
            self._trodes_sim_manager
        )

    def main_loop(self):

        # delete when ready
        t0 = time.time()

        try:
            while True:
                self._mpi_recv.receive()
                self._trodes_sim_manager.next_iter()

                # delete when ready
                # if time.time() - t0 > 5:
                #     break

        except StopIteration as ex:
            self.class_log.info("Exiting normally")
        except Exception as e:
            self.class_log.exception(
                "Trodes sim process exception occurred!"
            )

        self.class_log.info("Exited main loop")