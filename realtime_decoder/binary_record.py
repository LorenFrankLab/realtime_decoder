import os
import time
import json
import struct
import numpy as np
import pandas as pd

from collections import OrderedDict
from enum import IntEnum
from typing import List

from realtime_decoder import logging_base, messages

class RecordIDs(IntEnum):
    """The numeric ID for each record type
    """

    RIPPLE_STATE = 1
    ENCODER_QUERY = 2
    ENCODER_OUTPUT = 3
    DECODER_OUTPUT = 4
    DECODER_MISSED_SPIKES = 5
    LIKELIHOOD_OUTPUT = 6
    OCCUPANCY = 7
    #####################################################################################################################################
    # This one is only temporary for testing. Remove when finalized
    POS_INFO = 8
    #####################################################################################################################################

    STIM_STATE = 10
    STIM_LOCKOUT = 11
    STIM_MESSAGE = 12
    STIM_HEAD_DIRECTION = 13
    STIM_RIPPLE_DETECTED = 14
    STIM_RIPPLE_END = 15

    RIPPLE_DETECTED = 20 
    RIPPLE_END = 21

    TIMING = 100

class BinaryRecordsError(Exception):
    """An exception raised when an error occurs in handling binary
    record data
    """
    def __init__(self, value, **kwargs):
        self.value = value
        self.data = kwargs

    def __str__(self):
        return repr(self.value) + '\n' + repr(self.data)


class BinaryRecordsManager(object):
    """
    Keeps track of binary records. Can handle records from multiple processes

    Init parameters
    ---------------
    manager_label: str
        Label describing this manager
    save_dir : str, optional
        Directory to save files to. Default is current working directory
    file_prefix : str
        Prefix for written files. Default is nothing
    file_postfix : str
        Postfix for written files. Default is nothing
    """
    def __init__(
        self, manager_label:str, num_digits:int=2,
        save_dir:str='', file_prefix:str='',
        file_postfix:str=''
    ):

        self._manager_label = manager_label
        self._save_dir = save_dir
        self._file_prefix = file_prefix
        self._file_postfix = file_postfix
        self._num_digits = num_digits
        self._rec_format_dict = {}
        self._rec_label_dict = {}
        self._next_file_index = 1

    def register_rec_type_message(
        self, rec_type_message:messages.BinaryRecordType
    ) -> None:

        """Register a binary record type message
        """
        if rec_type_message.manager_label != self._manager_label:
            raise BinaryRecordsError(
                "Trying to register record type with wrong manager: "
                f"record manager_label={rec_type_message.manager_label}, "
                f"manager manager_label={self._manager_label}"
            )
        self._register_rec_type(
            rec_type_message.rec_id,
            rec_type_message.rec_labels,
            rec_type_message.rec_struct_fmt
        )

    def _register_rec_type(
        self, rec_id:int, rec_labels:List[str], rec_struct_fmt:List[str]
    ) -> None:
        
        """Register binary record information
        """

        if self._next_file_index > 1:
            raise BinaryRecordsError(
                "Cannot add more record types after manager has created a file. A programming error "
                f" must have occured. Tried to add {rec_id}, {rec_labels}, {rec_struct_fmt}"
            )
        else:
            if rec_id in self._rec_format_dict:
                if (
                    (rec_labels != self._rec_label_dict[rec_id]) or 
                    (rec_struct_fmt != self._rec_format_dict[rec_id])
                ):
                    raise BinaryRecordsError(
                        "Record ID already exists and id or format does not match. "
                        f"old rec: ({rec_id}, {self._rec_label_dict[rec_id]}, {self._rec_format_dict[rec_id]} "
                        f"new rec: ({rec_id}, {rec_labels}, {rec_struct_fmt}"
                    )
            self._rec_format_dict.update({rec_id: rec_struct_fmt})
            self._rec_label_dict.update({rec_id: rec_labels})

    def get_new_writer_message(self):
        message = messages.BinaryRecordCreate(
            self._manager_label, self._next_file_index, self._save_dir, self._file_prefix,
            self._file_postfix, self._rec_label_dict, self._rec_format_dict, self._num_digits
        )
        self._next_file_index += 1

        return message

def _get_absolute_path(
    save_dir:str, file_prefix:str, rank:int, num_digits:int,
    manager_label:str, file_postfix:str
) -> str:
    
    """Gets absolute path

    Parameters
    ----------
    save_dir : str, optional
        Directory to save files to. Default is current working directory
    file_prefix : str
        Prefix for written files. Default is nothing
    rank : int
        MPI rank of this writer
    manager_label: str
        Label describing this manager
    file_postfix : str
        Postfix for written files. Default is nothing

    Returns
    -------
    Absolute path, of type str
    """
    return os.path.join(
        save_dir,
        f"{file_prefix}.{rank:0{num_digits}d}.{manager_label}.{file_postfix}"
    )


class BinaryRecordsFileWriter(logging_base.LoggingClass):
    """
    File handler for a single Binary Records file.

    Current can only be created through a BinaryRecordCreateMessage (primarily for remote file
    creation).

    The file name will be defined by the BinaryRecordCreateMessage's attributes and the mpi_rank
    parameter if specified: <file_prefix>.<manager_label>.<mpi_rank>.<file_postfix>

    A Binary Records file consists of a JSON header prepended to the file that must define the
    following entries:
        file_prefix: The root file name that is shared if a data store spans multiple files
        file_id: A unique ID for the given file
        name: Descriptive label (shared across all files)
        rec_type_spec: The format (python struct - format character) of all possible recs

    What follows is a binary blob that contains a list of records with the following format:
        <rec_ind (uint32)> <rec_type (uint8)> <rec_data>

    Each record type with unique ID must be specified in rec_type_spec using python struct's
    format character syntax (don't prepend with a byte order character).

    Each record type has a fixed size that is implicitly defined by its format string.

    Init parameters
    ---------------
    create_message : message of type messages.BinaryRecordCreate
        Relevant information needed for file writing
    mpi_rank : int
        MPI rank of this class
    """

    def __init__(self, create_message:messages.BinaryRecordCreate, mpi_rank:int):

        super().__init__()

        self._rank = mpi_rank

        self._manager_label = create_message.manager_label
        self._file_id = create_message.file_id
        self._save_dir = create_message.save_dir
        self._file_prefix = create_message.file_prefix
        self._file_postfix = create_message.file_postfix
        self._rec_label_dict = create_message.rec_label_dict
        self._rec_format_dict = create_message.rec_format_dict
        self._num_digits = create_message.num_digits

        self._file_handle = None
        self._rec_counter = 0

    def start_record_writing(self):
        
        header = json.dumps(
            OrderedDict(
                [('file_prefix', self._file_prefix),
                ('file_id', self._file_id),
                ('mpi_rank', self._rank),
                ('manager_label', self._manager_label),
                ('rec_formats', self._rec_format_dict),
                ('rec_labels', self._rec_label_dict)]
            )
        )

        self._file_handle = open(
            _get_absolute_path(
                self._save_dir, self._file_prefix, self._rank,
                self._num_digits, self._manager_label, self._file_postfix
            ),
            'wb'
        )
        self._file_handle.write(bytearray(header, encoding='utf-8'))

        self.class_log.info(
            "Wrote binary record file header to "
            f"{self._file_handle.name}"
        )

    def write_rec(self, rec_type_id:int, *args) -> None:
        """Writes binary record

        Parameters:
        -----------
        rec_type_id : int
            ID of the record type
        args : variable types
            Data for the specific rec id
        """
        try:
            rec_bytes = struct.pack(
                '=QBq' + self._rec_format_dict[rec_type_id],
                self._rec_counter, rec_type_id, time.time_ns(),
                *args
            )
            self._file_handle.write(rec_bytes)
            self._rec_counter += 1
        except struct.error as ex:
            raise BinaryRecordsError(
                f"Data does not match record {rec_type_id}'s data format",
                rec_type_id=rec_type_id, rec_type_fmt=self._rec_format_dict[rec_type_id],
                rec_data=args, orig_error=ex
            ) from ex

    def __del__(self):
        # shouldn't be necessary but automatic cleanup just in case
        if self._file_handle is not None:
            self._file_handle.close()

    @property
    def is_open(self) -> bool:
        """Whether or not this writer has an open file handle
        """
        if self._file_handle is None:
            return False

        return not self._file_handle.closed

    def close(self) -> None:
        """Closes any open file handles
        """
        if self._file_handle is not None:
            self._file_handle.close()

class BinaryRecordsFileReader(logging_base.LoggingClass):

    def __init__(
        self, save_dir, file_prefix, mpi_rank, num_digits,
        manager_label, file_postfix, *, metadata=False
    ):

        # self._save_dir = save_dir
        # self._file_prefix = file_prefix
        self._mpi_rank = mpi_rank
        # self._num_digits = num_digits
        # self._manager_label = manager_label
        # self._file_postfix = file_postfix
        self._metadata = metadata

        self._file_handle = None
        self._header = None
        self._data_start_byte = None


        self._filepath = _get_absolute_path(
            save_dir, file_prefix, mpi_rank, num_digits,
            manager_label, file_postfix
        )

        if not os.path.isfile(self._filepath):
            raise Exception(f"File {self._filepath} not found")

        self._extract_header()

    def _extract_header(self):

        with open(self._filepath, 'rb') as f:

            f.seek(0)
        
            header_bytes = bytearray()
            read_byte = f.read(1)

            if read_byte != b'{':
                raise BinaryRecordsError(
                    'Not a Binary Records file, JSON header not found at first byte.',
                    file_path=self._filepath, read_byte=read_byte
                )

            level = 0
            while read_byte:
                header_bytes.append(ord(read_byte))
                if read_byte == b'{':
                    level += 1
                elif read_byte == b'}':
                    level -= 1

                if level == 0:
                    break
                elif len(header_bytes) >= 10000:
                    raise BinaryRecordsError(
                        'Could not find end of JSON header before 10000 byte header limit.',
                        file_path=self._filepath
                    )

                read_byte = f.read(1)

            if level != 0:
                raise BinaryRecordsError(
                    'Could not find end of JSON header before end of file.',
                    file_path=self._filepath
                )

            self._data_start_byte = f.tell()

            self._header = json.loads(header_bytes.decode('utf-8'))

    def _read_record(self):
        # Assuming file_handle pointer is aligned to the beginning of a message
        # read header
        # cpdef unsigned long long rec_ind
        # cpdef unsigned char rec_type_id
        rec_head_bytes = self._file_handle.read(struct.calcsize('=QBq'))
        if not rec_head_bytes:
            return None

        try:
            rec_ind, rec_type_id, rec_time = struct.unpack('=QBq', rec_head_bytes)

            rec_fmt = self._header['rec_formats'][str(rec_type_id)]
            rec_data_bytes = self._file_handle.read(struct.calcsize('=' + rec_fmt))
            rec_data = struct.unpack('=' + rec_fmt, rec_data_bytes)
        except struct.error as ex:
            raise BinaryRecordsError(
                'File might be corrupted, record does not match format or unexpected EOF.',
                file_path=self._filepath
            )

        return rec_type_id, rec_ind, rec_time, rec_data

    def __iter__(self):
        return self

    def __next__(self):
        rv = self._read_record()
        if rv is None:
            raise StopIteration
        else:
            return rv

    def _get_rec_labels(self):
        return {int(key): value for key, value in self._header['rec_labels'].items()}

    def _bytes_to_string(self, c_bytes):
        return c_bytes.split(b'\0')[0].decode('utf-8')

    def convert_to_pandas(self):
        # find start of data
        self._file_handle = open(self._filepath, 'rb')
        self._file_handle.seek(self._data_start_byte)

        columns = self._get_rec_labels()
        data_dict = {key: [] for key in columns.keys()}

        rec_count = 0
        if self._metadata:
            for rv in self: # (rec_id, rec_ind, rec_time, rec_data)

                data_dict[rv[0]].append((rv[1], rv[2], self._mpi_rank, self._filepath) + rv[3])

            panda_frames = {
                key: pd.DataFrame(
                    data=data_dict[key],
                    columns=['rec_ind', 'rec_time', 'mpi_rank', 'file_path'] + columns[key]
                )
                for key in columns.keys()
            }
        else:
            for rv in self:
                data_dict[rv[0]].append((rv[1], rv[2]) + rv[3])

            panda_frames = {
                key: pd.DataFrame(
                    data=data_dict[key],
                    columns=['rec_ind', 'rec_time'] + columns[key]
                )
                for key in columns.keys()
            }

        # Converting bytes into strings
        for rec_id, table in panda_frames.items():
            if len(table) > 0:
                for col_name in table:
                    if table[col_name].dtype == np.object:
                        if isinstance(table[col_name].iloc[0], bytes):
                            table[col_name] = table[col_name].apply(self._bytes_to_string)

        panda_numeric_frames = {
            key: df.apply(pd.to_numeric, errors='ignore')
            for key, df in panda_frames.items()
        }

        self._file_handle.close()

        return panda_numeric_frames

    @property
    def filepath(self):
        return self._filepath