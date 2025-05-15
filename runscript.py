import os
import argparse
import time
import datetime
import logging
import logging.config

import oyaml as yaml

from multiprocessing import cpu_count
from mpi4py import MPI

from realtime_decoder import (
    datatypes, position, trodesnet, stimulation,
    main_process, ripple_process, encoder_process,
    decoder_process, gui_process, base, messages,
    merge_rec, instantiation
)

# from line_profiler import LineProfiler

class GuiProcessStub(base.RealtimeProcess, base.MessageHandler):

    def __init__(self, comm, rank, config):
        super().__init__(comm, rank, config)

        self._mpi_recv = base.StandardMPIRecvInterface(
            comm, rank, config, messages.MPIMessageTag.COMMAND_MESSAGE,
            self
        )

        self._mpi_send = base.StandardMPISendInterface(
            comm, rank, config
        )

    def main_loop(self):

        t  = time.time()
        send = True
        try:

            while True:
                self._mpi_recv.receive()
                # if time.time() - t > 10:
                #     raise Exception("Testing GUI exception")
                # if time.time() - t > 5 and send:
                #     self.comm.send(
                #         messages.StartupSignal(),
                #         dest=self.config['rank']['supervisor'][0],
                #         tag=messages.MPIMessageTag.COMMAND_MESSAGE)
                #     send = False

        except StopIteration:
            self.class_log.info("Terminating")
        except Exception as e:
            self.class_log.exception("Some exception happened!")

    def handle_message(self, msg, mpi_status):

        if isinstance(msg, messages.VerifyStillAlive):
            self._mpi_send.send_alive_message()
        elif isinstance(msg, messages.SetupComplete):
            self.class_log.info("Got a setupcomplete message")
            self.comm.send(
                messages.StartupSignal(),
                dest=self.config['rank']['supervisor'][0],
                tag=messages.MPIMessageTag.COMMAND_MESSAGE
            )
        elif isinstance(msg, messages.DecoderStarted):
            pass
        elif isinstance(msg, messages.TerminateSignal):
            raise StopIteration()

    def handle_gui_message(self, msg, mpi_status):
        pass

class PosInterfaceStub(base.DataSourceReceiver):

    def __init__(self, comm, rank, config, datatype):
        super().__init__(comm, rank, config, datatype)

    def register_datatype_channel(self, channel):
        pass

    def activate(self):
        pass

    def deactivate(self):
        pass

    def __next__(self):
        return None

    def stop_iterator(self):
        raise StopIteration()

def setup(config_path, numprocs):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    num_digits = len(str(comm.Get_size()))

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    os.makedirs(os.path.dirname(config['files']['output_dir']), exist_ok=True)
    prefix = config['files']['prefix']
    comm.Barrier()
    config['files']['prefix'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_') + prefix

    # setup logging
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'simple': {
                'format': (
                    '%(asctime)s.%(msecs)03d [%(levelname)s] '
                    f'[MPI-{rank:0{num_digits}d}] [PID: %(process)d] [%(filename)s:%(lineno)d] %(name)s: %(message)s'
                ),
                'datefmt': '%H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'simple',
            },
            'debug_file_handler': {
                'class': 'realtime_decoder.logging_base.MakeFileHandler',
                'level': 'DEBUG',
                'formatter': 'simple',
                'filename': (
                    '{odir}/{date_str}_debug.log/{date_str}_MPI-{rank:02d}_debug.log'.
                    format(
                        odir=config['files']['output_dir'],
                        date_str=config['files']['prefix'],
                        rank=rank
                    )
                ),
                'encoding': 'utf8',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'debug_file_handler'],
                'level': 'NOTSET',
                'propagate': True,
            }
        }
    })

    if rank == comm.Get_size() - 1:
        ofile = os.path.join(
            config['files']['output_dir'],
            config['files']['prefix'] + '.config.yaml'
        )
        with open(ofile, 'w') as f:
            yaml.dump(config, f)

    comm.Barrier()
    time.sleep(0.1*rank)
    print(f"Rank {rank} past barrier")

    #################################################
    # remove when done
    regloop = True
    #################################################
    
    if rank in config['rank']['supervisor']:
        process = instantiation.create_main_process(comm, rank, config)
    elif rank in config['rank']['ripples']:
        lfp_interface = trodesnet.TrodesDataReceiver(
            comm, rank, config, datatypes.Datatypes.LFP
        )
        pos_interface = trodesnet.TrodesDataReceiver(
            comm, rank, config, datatypes.Datatypes.LINEAR_POSITION
        )
        process = ripple_process.RippleProcess(
            comm, rank, config, lfp_interface, pos_interface
        )

        # prof = LineProfiler()
        # prof.add_module(ripple_process)
        # prof.runcall(process.main_loop)
        # prof.print_stats()
        # regloop = False
    elif rank in config['rank']['encoders']:
        spikes_interface = trodesnet.TrodesDataReceiver(
            comm, rank, config, datatypes.Datatypes.SPIKES
        )
        pos_interface = trodesnet.TrodesDataReceiver(
            comm, rank, config, datatypes.Datatypes.LINEAR_POSITION
        )
        pos_mapper = position.TrodesPositionMapper(
            config['encoder']['position']['arm_ids'],
            config['encoder']['position']['arm_coords']
        )
        process = encoder_process.EncoderProcess(
            comm, rank, config, spikes_interface, pos_interface,
            pos_mapper
        )
    elif rank in config['rank']['decoders']:
        pos_interface = trodesnet.TrodesDataReceiver(
            comm, rank, config, datatypes.Datatypes.LINEAR_POSITION
        )
        pos_mapper = position.TrodesPositionMapper(
            config['encoder']['position']['arm_ids'],
            config['encoder']['position']['arm_coords']
        )
        process = decoder_process.DecoderProcess(
            comm, rank, config, pos_interface, pos_mapper
        )

        # prof = LineProfiler()
        # prof.add_module(decoder_process)
        # prof.runcall(process.main_loop)
        # prof.dump_stats('decoder-prof-results')
        # prof.print_stats()
        # regloop = False
    elif rank in config['rank']['gui']:
        process = GuiProcessStub(comm, rank, config)
        process = gui_process.GuiProcess(comm, rank, config)
    else:
        regloop = False
        raise ValueError(f"Could not find rank {rank} listed in the config file!")

    if regloop:
        process.main_loop()

    comm.Barrier()
    if rank == 0:
        merge_rec.merge_with_temp(config, numprocs)
    

if __name__ == "__main__":

    comm = MPI.COMM_WORLD

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to config file")
    parser.add_argument(
        '--numprocs', '-n', type=int, default=comm.Get_size(),
        help="Max number of processes to spawn during record merging"
    )

    args = parser.parse_args()

    setup(args.config, args.numprocs)
