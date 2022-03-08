import os
import struct

from enum import IntEnum
from typing import List

import realtime_decoder.base as base
import realtime_decoder.messages as messages

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
        self, manager_label:str, save_dir:str='', file_prefix:str='',
        file_postfix:str=''
    ):

        self._manager_label = manager_label
        self._save_dir = save_dir
        self._file_prefix = file_prefix
        self._file_postfix = file_postfix
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

    # def get_new_writer_message(self):
    #     message = BinaryRecordCreateMessage(
    #         self._manager_label, self._next_file_index, self._save_dir, self._file_prefix,
    #         self._file_postfix, self._rec_label_dict, self._rec_format_dict
    #     )
    #     self._next_file_index += 1

    #     return message


class BinaryRecordsFileWriter(object):
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


        file_prefix = create_message.file_prefix
        file_id = create_message.file_id
        rank = mpi_rank
        manager_label = create_message.manager_label
        file_postfix = create_message.file_postfix
        rec_label_dict = create_message.rec_label_dict
        rec_format_dict = create_message.rec_fmt_dict

        header = json.dumps(
            OrderedDict(
                [('file_prefix', file_prefix),
                ('file_id', file_id),
                ('mpi_rank', rank),
                ('manager_label', manager_label),
                ('rec_formats', rec_format_dict),
                ('rec_labels', rec_label_dict)]
            )
        )

        self._file_handle = open(
            self._get_absolute_path(
                save_dir, file_prefix, rank, manager_label, file_postfix
            ),
            'wb'
        )
        self._file_handle.write(bytearray(header), encoding='utf-8')

        self._rec_counter = 0

    def _get_absolute_path(
        self, save_dir:str, file_prefix:str, rank:int, manager_label:str,
        file_postfix:str
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
            f"{file_prefix}.{rank:02d}.{manager_label}.{file_postfix}"
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
                '=QB' + self.rec_format_dict[rec_type_id],
                self._rec_counter, rec_type_id, *args
            )
            self._file_handle.write(rec_bytes)
            self._rec_counter += 1
        except struct.error as ex:
            raise BinaryRecordsError('Data does not match record {}\'s data format.'.format(rec_type_id),
                                     rec_type_id=rec_type_id, rec_type_fmt=self.rec_format_dict[rec_type_id],
                                     rec_data=args, orig_error=ex) from ex

    def __del__(self):
        # shouldn't be necessary but automatic cleanup just in case
        self._file_handle.close()

    @property
    def is_open(self) -> bool:
        """Whether or not this writer has an open file handle
        """
        return not self._file_handle.closed

    def close(self) -> None:
        """Closes any open file handles
        """
        self._file_handle.close()