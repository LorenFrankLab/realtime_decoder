"""Contains objects needed for running in simulation mode,
in which one data receivers read from Trodes-extracted files"""


import os
import glob
import time
import numpy as np

from realtime_decoder import trodes, messages

from realtime_decoder.datatypes import (
    Datatypes, LFPPoint, SpikePoint, CameraModulePoint)
from realtime_decoder.base import DataSourceReceiver
from realtime_decoder.ripple_process import RippleManager
from realtime_decoder.encoder_process import EncoderManager
from realtime_decoder.decoder_process import DecoderManager

def load_lfp(lfp_dir, ntrode_ids):

    ts_files = glob.glob(f"{lfp_dir}/*.timestamps*")
    if len(ts_files) == 0:
        raise ValueError(f"Unable to find timestamp file in LFP directory {lfp_dir}")
    elif len(ts_files) != 1:
        raise ValueError(
            f"Multiple timestamp files found in LFP directory {lfp_dir}: {ts_files}")

    timestamps, _ = trodes.load_data_file(ts_files[0])
    lfp_timestamps = timestamps['time']

    lfp_data = np.zeros( (len(lfp_timestamps), len(ntrode_ids)) )

    for ind, ntid in enumerate(ntrode_ids):
        lfp_files = glob.glob(f"{lfp_dir}/*nt{ntid}ch*.dat")
        if len(lfp_files) == 0:
            raise ValueError(f"Unable to find LFP file for ntrode ID {ntid}")
        elif len(lfp_files) != 1:
            raise ValueError(
                f"Multiple LFP files with ntrode ID {ntid}: {lfp_files}")

        data, _ = trodes.load_data_file(lfp_files[0])
        lfp_data[:, ind] = data['voltage']

    # Data type is the same as contained in file
    return lfp_timestamps, lfp_data.astype(data['voltage'].dtype)

def load_spikes(spike_dir, ntrode_id):

    spike_files = glob.glob(f"{spike_dir}/*nt{ntrode_id}.dat")
    if len(spike_files) == 0:
        raise ValueError(
            f"Unable to find spike file for ntrode ID {ntrode_id} in {spike_dir}")
    elif len(spike_files) != 1:
        raise ValueError(
            f"Multiple spike files found in spikes directory {spike_dir}: {spike_files}")

    data, _ = trodes.load_data_file(spike_files[0])

    # Combine the spikes into a 3D array of shape (n_spikes, n_channels, n_points)
    # where n_points is the number of samples per spike
    n_spikes = len(data['time'])

    n_chan = 0
    for field in data.dtype.fields:
        if 'waveform' in field:
            n_chan += 1

    # An ntrode will always have one channel, so we expect this key to exist.
    # But if not, raise an error
    try:
        n_points = data['waveformCh1'].shape[1]
    except KeyError:
        raise KeyError(
            f"Could not find field 'waveformCh1', fields are {data.dtype.fields}")

    spike_data = np.zeros(
        (n_spikes, n_chan, n_points), dtype=data['waveformCh1'].dtype)

    for ind in range(n_chan):
        channel = ind + 1
        spike_data[:, ind, :] = data[f'waveformCh{channel}']

    return data['time'], spike_data

def load_pos(raw_dir):

    lin_pos_files = glob.glob(f"{raw_dir}/*videoLinearTracking")
    if len(lin_pos_files) == 0:
        raise ValueError(
            f"Could not find linear tracking file in raw directory {raw_dir}")
    elif len(lin_pos_files) > 1:
        raise ValueError(
            f"Multiple linear tracking files found in raw directory {raw_dir}: {lin_pos_files}")

    pos_files = glob.glob(f"{raw_dir}/*videoPositionTracking")
    if len(pos_files) == 0:
        raise ValueError(
            f"Could not find position tracking file in raw directory {raw_dir}")
    elif len(pos_files) > 1:
        raise ValueError(
            f"Multiple position tracking files found in raw directory {raw_dir}: {pos_files}")

    lin_pos_data, _ = trodes.load_data_file(lin_pos_files[0])
    pos_data, _ = trodes.load_data_file(pos_files[0])

    # For some reason, the timestamps are sometimes not the same length, handle this
    common_timestamps = np.intersect1d(pos_data['time'], lin_pos_data['time'])

    pos_mask = np.logical_and(
        pos_data['time'] >= common_timestamps[0],
        pos_data['time'] <= common_timestamps[-1]
    )
    pos_data = pos_data[pos_mask]

    lin_pos_mask = np.logical_and(
        lin_pos_data['time'] >= common_timestamps[0],
        lin_pos_data['time'] <= common_timestamps[-1]
    )
    lin_pos_data = lin_pos_data[lin_pos_mask]

    return pos_data, lin_pos_data

class LfpDataExtractor(object):

    def __init__(self, lfp_timestamps, lfp_data, ntrode_ids, scale_factor):

        self.lfp_ts = lfp_timestamps
        self.lfp_data = lfp_data
        self.ntrode_ids = ntrode_ids
        self.sfact = scale_factor

        self.curr_ind = 0

    def set_data_range(self, first_timestamp, last_timestamp):
        mask = np.logical_and(
            self.lfp_ts >= first_timestamp,
            self.lfp_ts <= last_timestamp)
        self.lfp_ts = self.lfp_ts[mask]
        self.lfp_data = self.lfp_data[mask]

    def get_data_at(self, ts_start, ts_end):

        # We are at the end of data
        if self.curr_ind >= len(self.lfp_ts):
            return [], True

        datas = []
        while True:
            # No data in requested range
            if (self.curr_ind >= len(self.lfp_ts) or
                self.lfp_ts[self.curr_ind] < ts_start or
                self.lfp_ts[self.curr_ind] >= ts_end
            ):
                break
            
            curr_time = time.time_ns()
            #print(f"***********Got lfp at timestamp {self.lfp_ts[self.curr_ind]}!***********")
            datas.append(
                LFPPoint(
                    self.lfp_ts[self.curr_ind],
                    self.ntrode_ids,
                    self.lfp_data[self.curr_ind],
                    curr_time,
                    curr_time)
            )
            self.curr_ind += 1

        return datas, False

class SpikeDataExtractor(object):

    def __init__(self, spike_timestamps, spike_data, ntrode_id, scale_factor):

        self.spike_ts = spike_timestamps
        self.spike_data = spike_data
        self.ntrode_id = ntrode_id
        self.sfact = scale_factor

        self.curr_ind = 0

    def set_data_range(self, first_timestamp, last_timestamp):
        mask = np.logical_and(
            self.spike_ts >= first_timestamp,
            self.spike_ts <= last_timestamp)
        self.spike_ts = self.spike_ts[mask]
        self.spike_data = self.spike_data[mask]

    def get_data_at(self, ts_start, ts_end, datas):

        # We are at the end of data
        if self.curr_ind >= len(self.spike_ts):
            return [], True

        while True:
            # No data in requested range
            if (self.curr_ind >= len(self.spike_ts) or
                self.spike_ts[self.curr_ind] < ts_start or
                self.spike_ts[self.curr_ind] >= ts_end
            ):
                break
            
            #print(f"***********Got spike at timestamp {self.spike_ts[self.curr_ind]}!***********")
            curr_time = time.time_ns()
            datas.append(
                SpikePoint(
                    self.spike_ts[self.curr_ind],
                    self.ntrode_id,
                    self.spike_data[self.curr_ind],
                    curr_time,
                    curr_time)
            )
            self.curr_ind += 1

        return datas, False

class PosDataExtractor(object):

    def __init__(self, pos_data, lin_pos_data):

        if not np.allclose(pos_data['time'], lin_pos_data['time']):
            raise ValueError(
                f"Position and linear position data timestamps are not identical."
                "Data cannot be combined")

        self.pos_data = pos_data
        self.lin_pos_data = lin_pos_data

        self.curr_ind = 0

    def set_data_range(self, first_timestamp, last_timestamp):
        # Doesn't matter which array we use
        mask = np.logical_and(
            self.pos_data['time'] >= first_timestamp,
            self.pos_data['time'] <= last_timestamp)
        self.pos_data = self.pos_data[mask]
        self.lin_pos_data = self.lin_pos_data[mask]

    def get_data_at(self, ts_start, ts_end):

        # We are at the end of data
        if self.curr_ind >= len(self.pos_data):
            return [], True

        datas = []
        while True:
            # It doesn't matter if we use self.pos_data['time] or self.lin_pos_data['time].
            # We already checked in the constructor that position and linear position
            # timestamps are identical.
            # No data in requested range
            if (self.curr_ind >= len(self.pos_data['time']) or
                self.pos_data['time'][self.curr_ind] < ts_start or
                self.pos_data['time'][self.curr_ind] >= ts_end
            ):
                break

            # print(f"***********Got pos at timestamp {self.pos_data['time'][self.curr_ind]}!***********")
            curr_time = time.time_ns()
            datas.append(
                CameraModulePoint(
                    self.pos_data['time'][self.curr_ind],
                    self.lin_pos_data['LineSegment'][self.curr_ind],
                    self.lin_pos_data['RelativeLinearPos'][self.curr_ind],
                    self.pos_data['xloc'][self.curr_ind],
                    self.pos_data['yloc'][self.curr_ind],
                    self.pos_data['xloc2'][self.curr_ind],
                    self.pos_data['yloc2'][self.curr_ind],
                    curr_time)
            )
            self.curr_ind += 1

        return datas, False

class TrodesFileDataReader(DataSourceReceiver):

    def __init__(self, comm, rank, config, datatype):
        if not datatype in (
            Datatypes.LFP,
            Datatypes.SPIKES,
            Datatypes.LINEAR_POSITION
        ):
            raise TypeError(f"Invalid datatype {datatype}")
        super().__init__(comm, rank, config, datatype)

        self.stream = False

        self.ntrode_ids = [] # only applicable for spikes and LFP

        self.source = self.config['datasource']
        self.sfact = self.config[self.source]["voltage_scaling_factor"]

        # For Trodes, spikes are sampled at the same rate as the data
        # acquisition system. So we can just reuse this config setting
        self.sampling_rate = self.config['sampling_rate']["spikes"]

        self.extractor = None
        self.spike_extractors = []
        self.samples_processed = 0

        self.t0 = 0
        self.ts_offset = 0
        self.curr_ts = 0

    def __next__(self):

        if not self.stream:
            return [], False

        # Figure out how many samples are expected based on sampling rate
        # and elapsed time from the start
        dt = (time.time_ns() - self.t0) / 1e9
        samples_due = int(dt * self.sampling_rate) - self.samples_processed

        if self.datatype == Datatypes.SPIKES:
            # Provide the output object
            datas = []
            end_of_data = False
            for extractor in self.spike_extractors:
                result = extractor.get_data_at(
                    self.curr_ts, self.curr_ts + samples_due, datas)
                end_of_data = end_of_data or result
        else:
            datas, end_of_data = self.extractor.get_data_at(
                self.curr_ts, self.curr_ts + samples_due)

        self.curr_ts += samples_due
        self.samples_processed += samples_due

        return datas, end_of_data

    def register_datatype_channel(self, channel):
        """Sets up streaming from the given channel, or more accurately
        an ntrode id. This method can be called repeatedly such that multiple
        channels can be streamed from one FileDataReceiver instance"""

        ntrode_id = channel
        if self.datatype in (Datatypes.LFP, Datatypes.SPIKES):
            if not ntrode_id in self.ntrode_ids:
                self.ntrode_ids.append(ntrode_id)
            else:
                self.class_log.debug(f"Already streaming from ntrode id {ntrode_id}")
        else:
            self.class_log.debug("Already set up to stream position, doing nothing")
            return

        self.class_log.debug(
            f"Set up to stream from ntrode ids {self.ntrode_ids}"
        )

    def activate(self):
        """Enable streaming"""

        if self.datatype == Datatypes.LFP:
            name = 'filesource.lfp'

            lfp_ts, lfp_data = load_lfp(
                self.config[self.source]['lfp_dir'], self.ntrode_ids)
            # Multiplication is expensive and we want to be as fast as possible. Therefore
            # do it here
            lfp_data = lfp_data * self.sfact
            self.extractor = LfpDataExtractor(lfp_ts, lfp_data, self.ntrode_ids, self.sfact)

        elif self.datatype == Datatypes.SPIKES:
            name = 'filesource.spikes'

            for ntrode_id in self.ntrode_ids:
                spike_ts, spike_data = load_spikes(
                    self.config[self.source]['spike_dir'], ntrode_id)
                spike_data = spike_data * self.sfact
                self.spike_extractors.append(
                    SpikeDataExtractor(spike_ts, spike_data, ntrode_id, self.sfact))

        else:
            name = 'filesource.position'

            pos_data, lin_pos_data = load_pos(
                self.config[self.source]['raw_dir'])

            self.extractor = PosDataExtractor(pos_data, lin_pos_data)

        self.class_log.debug(f"Datastream {name} activated")

    def deactivate(self):
        """Deactivate streaming. The __next__() method can still be called
        but no data will be returned"""

        self.stream = False

    def stop_iterator(self):
        """Stop streaming entirely"""

        raise StopIteration()

    def sync_data_time(self, start_time, first_timestamp, last_timestamp):

        self.class_log.info(
            f"Setting start time: {start_time}, first timestamp: {first_timestamp}, "
            f"last timestamp: {last_timestamp}"
        )

        # Physical units of time
        self.t0 = start_time

        # Software timestamps
        self.ts_offset = first_timestamp
        self.curr_ts = first_timestamp

        if self.datatype == Datatypes.SPIKES:
            for extractor in self.spike_extractors:
                extractor.set_data_range(first_timestamp, last_timestamp)
        else:
            self.extractor.set_data_range(first_timestamp, last_timestamp)

        self.stream = True

def validate_input(config):
    """Sanity check configuration for a file simulation"""

    datasource = config['datasource']
    if datasource != 'trodes_file_simulator':
        raise NotImplementedError(
            f'Datasource "{datasource}" not currently supported')

    spike_dir = config[datasource]['spike_dir']
    if not os.path.isdir(spike_dir):
        raise ValueError(f"Spikes directory {spike_dir} does not exist")

    lfp_dir = config[datasource]['lfp_dir']
    if not os.path.isdir(lfp_dir):
        raise ValueError(f"LFP directory {lfp_dir} does not exist")


class SimulatorRippleManager(RippleManager):

    def __init__(
        self, rank, config, send_interface, lfp_interface,
        pos_interface
    ):

        super().__init__(
            rank, config, send_interface, lfp_interface, pos_interface)

        self._termination_request_sent = False

    def next_iter(self):
        lfp_msgs, end_of_data = self._lfp_interface.__next__()

        if (
            end_of_data and
            self.rank == self._config['rank']['ripples'][0] and
            not self._termination_request_sent
        ):
            # send message to supervisor requesting termination
            self.class_log.info("End of LFP data, requesting termination")
            self.send_interface.comm.send(
                obj=messages.TerminateSignal(),
                dest=self._config['rank']['supervisor'][0],
                tag=messages.MPIMessageTag.COMMAND_MESSAGE
            )
            self._termination_request_sent = True

        for lfp_msg in lfp_msgs:
            t0 = time.time_ns()
            self._process_lfp(lfp_msg)
            t1 = time.time_ns()
            self._record_timings(
                lfp_msg.timestamp,
                lfp_msg.t_send_data, lfp_msg.t_recv_data,
                t0, t1, len(self._in_ripple.keys())
            )

        pos_msgs, _ = self._pos_interface.__next__()
        for pos_msg in pos_msgs:
            self._process_pos(pos_msg)


class SimulatorEncoderManager(EncoderManager):

    def __init__(self, rank, config, send_interface, spikes_interface,
        pos_interface, pos_mapper
    ):

        super().__init__(
            rank, config, send_interface, spikes_interface,
            pos_interface, pos_mapper)

    def next_iter(self):
        spike_msgs, _ = self._spikes_interface.__next__()
        for spike_msg in spike_msgs:
            self._process_spike(spike_msg)

        pos_msgs, _ = self._pos_interface.__next__()
        for pos_msg in pos_msgs:
            self._process_pos(pos_msg)


class SimulatorDecoderManager(DecoderManager):

    def __init__(
        self, rank, config, send_interface, spike_interface,
        pos_interface, lfp_interface, pos_mapper
    ):

        super().__init__(
            rank, config, send_interface, spike_interface,
            pos_interface, lfp_interface, pos_mapper)

    def next_iter(self):
        spike_msg = self._spike_interface.receive()
        if spike_msg is not None:
            self._process_spike(spike_msg)

        timestamp = self._lfp_interface.receive()
        if timestamp is not None:
            self._process_lfp_timestamp(timestamp)

        pos_msgs, _ = self._pos_interface.__next__()
        for pos_msg in pos_msgs:
            self._process_pos(pos_msg)

class TrodesClientStub(object):

    def __init__(self, config):
        pass

    def receive(self):
        pass

    def send_statescript_shortcut_message(self, val):
        pass

def main():

    import argparse
    import oyaml as yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to config file")
    parser.add_argument(
        '-p', '--processes', type=int, default=4, help="Number of processes to run")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    validate_input(config)

    receiver = TrodesFileDataReader(None, 0, config, Datatypes.LFP)
    for ch in range(1):
        receiver.register_datatype_channel(ch + 1)
    receiver.activate()

    x = TrodesFileDataReader(None, 0, config, Datatypes.LINEAR_POSITION)

    num = 1
    for ch in range(num):
        x.register_datatype_channel(ch + 1)
    x.activate()


    t0 = time.time()

    x.sync_data_time(
        time.time_ns(),
        receiver.extractor.lfp_ts[0],
        receiver.extractor.lfp_ts[0] + 30000*5)

    last_time = time.time_ns()
    curr_time = time.time_ns()
    while time.time() - t0 < 5:
        curr_time = time.time_ns()
        elapsed_time = (curr_time - last_time)/1e3
        if (elapsed_time > 25):
            print(f"Elapsed time us: {elapsed_time}")
        result = x.__next__()
        last_time = curr_time

    print(f"Samples processed: {x.samples_processed}")
    print(f"First lfp timestamp: {x.extractor.lfp_ts[0]}")
    print(f"Current lfp timestamp: {x.extractor.lfp_ts[x.extractor.curr_ind]}")

    # Do a test load to make sure everything is ok before starting up
    # processes
    datasource = config['datasource']
    lfp_data, lfp_timestamps = load_lfp(config[datasource]['lfp_dir'])
    pos_data, lin_pos_data = load_pos(config[datasource]['raw_dir'])

    # We need position data to be available to use spikes and lfp
    first_ts = lfp_timestamps[0]


if __name__ == "__main__":
    main()