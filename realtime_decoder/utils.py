import xml.etree.ElementTree as ET
import numpy as np
import os
import fcntl

from typing import List

def nop():
    pass

def get_ntrode_inds(config, ntrode_ids):
    # ntrode_ids should be a list of integers
    inds_to_extract = []
    xmltree = ET.parse(config["trodes"]["config_file"])
    root = xmltree.getroot()
    for ii, ntrode in enumerate(root.iter("SpikeNTrode")):
        ntid = int(ntrode.get("id"))
        if ntid in ntrode_ids:
            inds_to_extract.append(ii)

    return inds_to_extract

def get_network_address(config):
    xmltree = ET.parse(config["trodes"]["config_file"])
    root = xmltree.getroot()
    network_config = root.find("NetworkConfiguration")
    
    if network_config is None:
        raise ValueError("NetworkConfiguration section not defined")

    try:
        address = network_config.attrib["trodesHost"]
        port = network_config.attrib["trodesPort"]
    except KeyError:
        return None

    if "tcp://" in address:
        return address + ":" + port
    else:
        return "tcp://" + address + ":" + port

def estimate_new_stats(new_value, mean, M2, count):
    count += 1
    delta = (new_value - mean)
    mean += delta / count
    delta2 = (new_value - mean)
    M2 += (delta*delta2)
    return mean, M2, count

def normalize_to_probability(distribution):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.nansum(distribution)

def apply_no_anim_boundary(x_bins, arm_coor, image, fill=0):
    # note: mutates data!
    # from util.py script in offline decoder folder

    # calculate no-animal boundary
    arm_coor = np.array(arm_coor, dtype='float64')
    arm_coor[:,0] -= x_bins[1] - x_bins[0]
    bounds = np.vstack([[x_bins[-1], 0], arm_coor])
    bounds = np.roll(bounds, -1)

    boundary_ind = np.searchsorted(x_bins, bounds, side='right')
    #boundary_ind[:,1] -= 1

    for bounds in boundary_ind:
        if image.ndim == 1:
            image[bounds[0]:bounds[1]] = fill
        elif image.ndim == 2:
            image[bounds[0]:bounds[1], :] = fill
            image[:, bounds[0]:bounds[1]] = fill
    return image

def get_task_state(taskfile):

    with open(taskfile, 'rb') as f:
        fd = f.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        task_state = int(f.readline().decode().rstrip('\r\n'))

    return task_state

def read_rewards_file(rewards_file):

    with open(rewards_file, 'rb') as f:
        fd = f.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        
        line = f.readline().decode().rstrip('\r\n')
        stim_num, arm = tuple(line.split(' '))
        stim_num = int(stim_num)
        arm = int(arm)

    return stim_num, arm
    