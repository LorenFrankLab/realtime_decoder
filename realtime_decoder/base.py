import logging

from mpi4py import MPI
from enum import IntEnum
from typing import List, Dict
from abc import ABCMeta, abstractmethod

import realtime_decoder.binary_record as binary_record
import realtime_decoder.messages as messages
import realtime_decoder.datatypes as datatypes

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

    STIM_STATE = 10
    STIM_LOCKOUT = 11
    STIM_MESSAGE = 12
    STIM_HEAD_DIRECTION = 13

    TIMING = 100

class LoggingClass(object):
    """A class that logs information
    """
    def __init__(self):
        self.class_log = logging.getLogger(
            name=f'{self.__class__.__name__}'
        )

class MessageHandler(LoggingClass, metaclass=ABCMeta):
    """Base class for object that handles messages passed
    between processes
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def handle_message(self, msg, mpi_status:MPI.Status):
        pass

    @abstractmethod
    def handle_gui_message(self, msg, mpi_status:MPI.Status):
        pass

class MPIClass(LoggingClass):
    """Base class for object that can send and/or receive MPI
    messages
    """
    def __init__(self, comm:MPI.Comm, rank:int, config:Dict):
        self.comm = comm
        self.rank = rank
        self.config = config
        super().__init__()

class MPIRecvInterface(MPIClass, metaclass=ABCMeta):
    """Base class for object that can receive MPI messages
    """

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    @abstractmethod
    def receive(self):
        pass

class StandardMPIRecvInterface(MPIRecvInterface):
    """Used to receive MPI messages with a given tag
    """

    def __init__(
        self, comm:MPI.Comm, rank:int, config:Dict,
        msg_tag:int, 
        msg_handler:MessageHandler
    ):
        super().__init__(comm, rank, config)
        self._msg_handler = msg_handler
        self._msg_tag = msg_tag

        self._mpi_status = MPI.Status()
        self._req = self.comm.irecv(self._msg_tag)

    def receive(self) -> None:
        """Receives a message, and if available, immediately
        passes it to its message handler for processing
        """
        rdy, msg = self._req.test(status=self._mpi_status)
        if rdy:
            self._msg_handler.handle_message(
                msg, self._mpi_status)
            self._req = self.comm.irecv(self._msg_tag)

class MPISendInterface(MPIClass, metaclass=ABCMeta):
    """Base class for object that can receive MPI messages
    """

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

    @abstractmethod
    def send_record_register_messages(self):
        pass

class StandardMPISendInterface(MPISendInterface):
    """Used to send MPI messages
    """

    def __init__(self, comm:MPI.Comm, rank:int, config:Dict):
        super().__init__(comm, rank, config)

    def send_record_register_messages(
        self, msgs:List[messages.BinaryRecordType]
    ) -> None:

        """Sends binary record information to the global records
        manager

        Parameters
        ----------
        msgs : list of type messages.BinaryRecordType
            Binary record registration messages
        """

        for msg in msgs:
            self.class_log.info(f"Sending {msg}")
            self.comm.send(
                obj=msg, dest=self.config["rank"]["supervisor"][0],
                tag=messages.MPIMessageTag.COMMAND_MESSAGE
            )
        
        self.comm.send(
            obj=messages.BinaryRecordSendComplete(),
            dest=self.config['rank']['supervisor'][0],
            tag=messages.MPIMessageTag.COMMAND_MESSAGE
        )
        
class RealtimeProcess(MPIClass, metaclass=ABCMeta):
    """Base class for processes
    """

    def __init__(self, comm:MPI.Comm, rank:int, config:Dict):

        super().__init__(comm, rank, config)

    @abstractmethod
    def main_loop(self):
        pass


class BinaryRecordBase(LoggingClass, metaclass=ABCMeta):
    """The base class for objects that can write binary records to disk.
    
    Init parameters
    ---------------
    rec_ids : list of int, optional
        Record ids that can be written. Default is empty list
    rec_labels : list of a list of str, optional
        Labels for each rec_id. Default is empty list
    rec_formats : list of a list of str, optional
        Struct formats for each rec_id. Default is empty list
    send_interface : object of type MPISendInterface, optional
        An object responsible for sending messages via MPI
    manager_label : str, optional
        Denotes the global records manager to which this class
        can send its binary record information
    """

    def __init__(
        self, *, rec_ids:List[int]=[],
        rec_labels:List[List[str]]=[],
        rec_formats:List[List[str]]=[],
        send_interface:MPISendInterface=None,
        manager_label='state'
    ):

        super().__init__()
        self._rec_writer = None

        # validate inputs
        if not (
            len(rec_ids) == len(rec_labels) and
            len(rec_labels) == len(rec_formats)
        ):
            raise ValueError(
                "'rec_ids', 'rec_labels', and 'rec_labels' must all be the same length"
            )

        self._rec_ids = rec_ids
        self._rec_labels = rec_labels
        self._rec_formats = rec_formats
        self._send_interface = send_interface
        self._manager_label = manager_label

    def setup_mpi(self) -> None:
        """Sends binary records to global records manager
        """
        if self._send_interface is None:
            raise Exception(
                f"{self.__class__.__name__} does not send record registration messages"
            )

        self._send_interface.send_record_register_messages(self.get_records())
    
    def get_records(self) -> List[messages.BinaryRecordType]:
        """Gets binary record information

        Returns
        -------
        messages : list of type BinaryRecordType
        """

        messages = []
        for rec_id, labels, rec_fmt in zip(self._rec_ids, self._rec_labels, self._rec_formats):
            messages.append(
                self._get_register_rec_type_message(
                    rec_id, labels, rec_fmt
                )
            )

        return messages

    def _get_register_rec_type_message(
        self, rec_id:int, rec_labels:List[str],
        rec_struct_fmt:str) -> messages.BinaryRecordType:
        """Get binary record information stored in a message object

        Parameters
        ----------
        rec_id : int
            Record ID
        rec_labels : list of str
            Labels describing the record
        rec_struct_format : str
            Struct format describing the record

        Returns
        -------
        A message containing the binary record information, of type BinaryRecordType
        """
        if self._rec_writer is not None:
            raise binary_record.BinaryRecordsError(
                f"Attempting to register rec_id {rec_id}, rec_label {rec_labels}, rec_struct_fmt "
                f"{rec_struct_fmt} but cannot add more record types when file has already been "
                "created."
            )
        else:
            return messages.BinaryRecordType(
                self._manager_label, rec_id, rec_labels, rec_struct_fmt
            )

    def set_record_writer_from_message(self, message:messages.BinaryRecordCreate) -> None:
        """Sets the record writer for this class

        Parameters
        ----------
        message : object of type messages.BinaryRecordCreate
            message containing relevant information for creating the record writer
        """
        self.class_log.info("Setting record writer from message")
        if self._rec_writer:
            if self._rec_writer.is_open:
                raise binary_record.BinaryRecordsError(
                    "Attempting to set record writer when current record file is still open!"
                )
            else: # ok to open new writer
                self._rec_writer = binary_record.BinaryRecordsFileWriter(message, self._rank)
        else:
            self._rec_writer = binary_record.BinaryRecordsFileWriter(message, self._rank)

    def start_record_writing(self) -> None:
        """Starts record writing
        """
        self.class_log.info('Starting record writer.')
        if self._rec_writer:
            if not self._rec_writer.is_open:
                raise binary_record.BinaryRecordsError("Can't start recording, file not open!")
        else:
            raise binary_record.BinaryRecordsError("Can't start recording, record file never set!")

    def stop_record_writing(self):
        """Stops record writing
        """
        if self._rec_writer:
            self._rec_writer.close()

    def write_record(self, rec_id:int, *args) -> bool:
        """Writes record to disk

        Parameters
        ---------
        rec_id : int
            Record id
        args : variable types
            Data for the specific rec id
        Returns
        -------
        True if record written, False otherwise
        """

        if rec_id not in self._rec_ids:
            raise binary_record.BinaryRecordsError(
                f'{self.__class__.__name__} attempted to write unregistered id. '
                f"rec_id {rec_id}, record: {args}"
            )
        if self._rec_writer and self._rec_writer.is_open:
            self._rec_writer.write_rec(rec_id, *args)
            return True
        return False

    @property
    @abstractmethod
    def params(self):
        """Parameters that the class uses.
        """
        pass
    

class DataSourceReceiver(MPIClass, metaclass=ABCMeta):
    """An abstract class that ranks should use to communicate between neural data sources.

    This class should not be instantiated, only its subclasses.

    This provides an abstraction layer for sources of neural data (e.g., saved file simulator, acquisition system)
    to pipe data (e.g., spikes, lfp, position) to ranks that request data for processing.  This is only an abstraction
    for a streaming data (e.g. sockets, MPI) and makes a number of assumptions:

    1. The type of data and 'channel' (e.g., electrode channels 1, 2, 3) can be streamed to different client processes
    and registered by a client one channel at a time

    2. The streams can be started and stopped arbitrarily after the connection is established (no rule if data is lost
    during pause)

    3. The connection is destroyed when the iterator stops.
    """

    def __init__(self, comm, rank, config, datatype:datatypes.Datatypes):
        """

        Args:
            comm:
            rank:
            config:
            datatype: The type of data to request to be streamed, specified by spykshrk.realtime.datatypes.Datatypes
            *args:
            **kwds:
        """
        super().__init__(comm, rank, config)
        self.datatype = datatype

    @abstractmethod
    def register_datatype_channel(self, channel):
        """

        Args:
            channel: The channel of the data type to stream

        Returns:
            None

        """
        pass

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def deactivate(self):
        pass

    @abstractmethod
    def stop_iterator(self):
        pass

    @abstractmethod
    def __next__(self):
        pass