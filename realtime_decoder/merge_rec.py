import os
import glob
import time
import shutil
import argparse
import logging
import logging.config
import pandas as pd
import oyaml as yaml
import multiprocessing as mp

from typing import Dict 

from realtime_decoder import binary_record

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("merge_rec")

def init_shared(l, fname):
    global hdf5_lock
    global hdf5_filename
    hdf5_lock = l
    hdf5_filename = fname

def convert_pandas(reader):

    logger.debug(f"Reading from file {reader.filepath}")
    panda_dict = reader.convert_to_pandas()

    filepath_dict = {}
    hdf5_temp_filename = reader.filepath + '.tmp.h5'
    with pd.HDFStore(hdf5_temp_filename, 'w') as hdf5_store:
        for rec_id, df in panda_dict.items():
            if df.size > 0: # write pandas data to file
                filepath_dict[rec_id] = hdf5_temp_filename
                hdf5_store[f'rec_{rec_id}'] = df

    return filepath_dict

def merge_pandas(filename_items):

    rec_id = filename_items[0]
    filenames = filename_items[1]

    pandas = []

    for filename in filenames:
        store = pd.HDFStore(filename, 'r')
        pandas.append(store['rec_'+str(rec_id)])

    merged = pd.concat(pandas, ignore_index=True)
    merged = merged.apply(pd.to_numeric, errors='ignore')

    if 'timestamp' in merged.columns:
        merged.sort_values(['timestamp'], inplace=True)
        merged.reset_index(drop=True, inplace=True)

    hdf5_lock.acquire()

    logger.info("Saving merged rec ID {}.".format(rec_id))

    with pd.HDFStore(hdf5_filename, 'a') as hdf_store:
        hdf_store[f'rec_{rec_id}'] = merged

    hdf5_lock.release()

def copy_to_backup(config):

    try:

        backup_dir = config['files']['backup_dir']
        os.makedirs(backup_dir, exist_ok=True)
        
        copy_files = []
        copy_files.extend(
            glob.glob(os.path.join(config['files']['output_dir'], '*.yaml'))
        )
        copy_files.extend(
            glob.glob(os.path.join(config['files']['output_dir'], '*.h5'))
        )
        copy_files.extend(
            glob.glob(os.path.join(config['files']['output_dir'], '*.encoder.npz'))
        )
        copy_files.extend(
            glob.glob(os.path.join(config['files']['output_dir'], '*.occupancy.npz'))
        )
        
        for fname in copy_files:
            shutil.copy2(fname, backup_dir)
    
    except KeyError:
        pass

def merge_with_temp(config, numprocs):

    t0 = time.time()

    logger.info("Merging binary record files")

    if numprocs < 1:
        raise ValueError(
            "Number of processes must be at least 1. "
            f"Got {numprocs}"
        )

    postfix = config['files']['rec_postfix']
    save_dir = config['files']['output_dir']

    testfile = glob.glob(
        os.path.join(save_dir, f'*.{postfix}')
    )[0]
    
    fname = os.path.basename(testfile)
    prefix, rank, manager_label, _ = tuple(fname.split('.'))
    num_digits = len(rank)

    reader_list = []
    for rank in config['rank_settings']['enable_rec']:
        try:
            reader = binary_record.BinaryRecordsFileReader(
                save_dir, prefix, int(rank), num_digits,
                manager_label, postfix
            )
            reader_list.append(reader)
        except Exception as e:
            print("An exception occurred!")
            print(e)
            pass

    l = mp.Lock()
    fname = os.path.join(
        config['files']['output_dir'],
        config['files']['prefix'] + '.rec_merged.h5'
    )
    p = mp.Pool(
        numprocs, initializer=init_shared,
        initargs=(l, fname), maxtasksperchild=1
    )
    result = p.map(convert_pandas, reader_list)

    # since we're opening the file in append mode,
    # we want to make sure there isn't already
    # existing data
    if os.path.isfile(fname):
        raise Exception(
            f"Merged rec file {fname} already exists"
        )

    remapped_dict = {} # key: rec_id, value: list of rec files
    for filepath_dict in result:
        for rec_id, filepath in filepath_dict.items():
            recfile_list = remapped_dict.setdefault(rec_id, [])
            recfile_list.append(filepath)

    # for k, v in remapped_dict.items():
    #     print(f"Key: {k}, value: {v}")

    logging.info("Merging records...")
    p.map(merge_pandas, remapped_dict.items())

    t1 = time.time()

    logger.info("Removing temporary files")
    for k, v in remapped_dict.items():
        for fname in v:
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

    logger.info("Copying files to backup location...")

    copy_to_backup(config)

    t2 = time.time()

    logger.info(
        f"Took {(t1 - t0)/60:0.3f} minutes to merge record files"
    )
    logger.info(
        f"Took {(t2 - t1)/60:0.3f} minutes to remove temp files and "
        "copy files to backup location"
    )

if __name__ == "__main__":

    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help="Path to config file")
    parser.add_argument(
        '--numprocs', '-n', type=int, default=20,
        help="Max number of processes to spawn"
    )

    args = parser.parse_args()
    logger.debug(f"Using {args.numprocs} processes")

    with open(args.config, 'rb') as f:
        config = yaml.safe_load(f)
    
    t0 = time.time()
    merge_with_temp(config, args.numprocs)
    t1 = time.time()