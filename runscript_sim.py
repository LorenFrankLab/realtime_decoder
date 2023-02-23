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
    merge_rec, trodes_sim
)

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

    regloop = True
    comm.Barrier()
    time.sleep(0.1*rank)
    print(f"Rank {rank} past barrier")

    if rank in config['rank']['supervisor']:
        trodes_client = trodes_sim.TrodesClientStub(config)
        stim_decider = stimulation.TwoArmTrodesStimDecider(
            comm, rank, config, trodes_client
        )
        process = main_process.MainProcess(
            comm, rank, config, stim_decider, trodes_client
        )
        # trodes_client.set_startup_callback(process.startup)
        # trodes_client.set_termination_callback(process.trigger_termination)
    elif rank in config['rank']['ripples']:
        lfp_interface = trodes_sim.TrodesSimReceiver(
            comm, rank, config, datatypes.Datatypes.LFP
        )
        pos_interface = trodes_sim.TrodesSimReceiver(
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
        spikes_interface = trodes_sim.TrodesSimReceiver(
            comm, rank, config, datatypes.Datatypes.SPIKES
        )
        pos_interface = trodes_sim.TrodesSimReceiver(
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
        pos_interface = trodes_sim.TrodesSimReceiver(
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
        # process = GuiProcessStub(comm, rank, config)
        process = gui_process.GuiProcess(comm, rank, config)
    elif rank in config['rank']['simulator']:
        process = trodes_sim.TrodesSimProcess(comm, rank, config)
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

