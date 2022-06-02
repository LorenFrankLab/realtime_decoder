import numpy as np

from realtime_decoder import utils

##########################################################################
# Clusterless decoder transitions
##########################################################################

# used for 8-arm maze
# def sungod_transition_matrix_old(pos_bins, arm_coords, bias):

#     # this for tri-diagonal matrix 
#     from scipy.sparse import diags
#     n = len(pos_bins)
#     transition_mat = np.zeros([n, n])
#     k = np.array([(1/3) * np.ones(n - 1), (1/3) *
#                   np.ones(n), (1 / 3) * np.ones(n - 1)])
#     offset = [-1, 0, 1]
#     transition_mat = diags(k, offset).toarray()
#     box_end_bin = arm_coords[0, 1]

#     for x in arm_coords[:, 0]:
#         transition_mat[int(x), int(x)] = (5/9)
#         transition_mat[box_end_bin, int(x)] = (1/9)
#         transition_mat[int(x), box_end_bin] = (1/9)

#     for y in arm_coords[:, 1]:
#         transition_mat[int(y), int(y)] = (2 / 3)

#     transition_mat[box_end_bin, 0] = 0
#     transition_mat[0, box_end_bin] = 0
#     transition_mat[box_end_bin, box_end_bin] = 0
#     transition_mat[0, 0] = (2 / 3)

#     transition_mat[box_end_bin - 1, box_end_bin - 1] = (5/9)
#     transition_mat[box_end_bin - 1, box_end_bin] = (1/9)
#     transition_mat[box_end_bin, box_end_bin - 1] = (1/9)

#     transition_mat = transition_mat + bias
#     return transition_mag


# currently flat transition matrix
def sungod_transition_matrix(pos_bins, arm_coords, bias):

    n = len(pos_bins)
    transmat = np.zeros((n, n)) + bias

    # apply no animal boundary - make gaps between arms
    transmat = utils.apply_no_anim_boundary(
        pos_bins, arm_coords, transmat
    )

    # to smooth: take transition matrix to a power
    transmat = np.linalg.matrix_power(transmat, 1)

    # row normalize transition matrix
    transmat /= np.nansum(transmat, axis=1, keepdims=True)

    transmat[np.isnan(transmat)] = 0

    return transmat

##########################################################################
# Clusterless classifier transitions
##########################################################################

# Adapted from Eric's code at 
# https://github.com/Eden-Kramer-Lab/replay_trajectory_classification.
# Compatible with arm_coords specified in config file.
# Note that arm_coords[0] is assumed to be the center well.
# It is also assumed that arm_coords[0][0] is 0.

def _normalize_row_probability(x):
    '''Ensure the state transition matrix rows sum to 1
    '''
    x /= x.sum(axis=1, keepdims=True)
    x[~np.isfinite(x)] = 0
    return x

def _gaussian(x, mu, sigma):
    '''Normalized gaussian
    '''
    return np.exp(-0.5*((x - mu)/sigma)**2) / (sigma*np.sqrt(2*np.pi))


def random_walk(arm_coords, cm_per_bin, sigma):
    base_offset = arm_coords[0][1]*cm_per_bin
    arm_labels = np.zeros(arm_coords[-1][-1] + 1) * np.nan
    bin_centers = np.arange(arm_labels.shape[0], dtype=np.float) * np.nan
    for arm_ind, (a, b) in enumerate(arm_coords):
        arm_labels[a:b+1] = arm_ind

        dist_vec = np.arange(b - a + 1, dtype=np.float) * cm_per_bin
        if arm_ind != 0:
            dist_vec += base_offset + cm_per_bin
        bin_centers[a:b+1] = dist_vec

    n_states = bin_centers.shape[0]
    transmat = np.zeros((n_states, n_states))
    for ii, (arm_label, center) in enumerate(zip(arm_labels, bin_centers)):
        if np.isnan(arm_label):
            transmat_row = 0
        else:
            transmat_row = _gaussian(bin_centers, center, sigma)
            if arm_label == 0:
                mask = ~np.isnan(arm_labels)
            else:
                # transitions within a specific arm ok. transitions to arm 0
                # (center well) also ok.
                mask = np.logical_or(arm_labels==0, arm_labels==arm_label)
            transmat_row[~mask] = 0
        transmat[ii] = transmat_row

    return _normalize_row_probability(transmat)


def uniform(arm_coords, cm_per_bin, sigma):
    n_states = arm_coords[-1][-1]- arm_coords[0][0] + 1
    is_track_interior = np.zeros(n_states, dtype=bool)
    for ii, (a, b) in enumerate(arm_coords):
        is_track_interior[a:b+1] = True

    transmat = np.ones((n_states, n_states))
    transmat[~is_track_interior] = 0
    transmat[:, ~is_track_interior] = 0

    return _normalize_row_probability(transmat)


def identity(arm_coords, cm_per_bin, sigma):
    n_states = arm_coords[-1][-1]- arm_coords[0][0] + 1
    is_track_interior = np.zeros(n_states, dtype=bool)
    for ii, (a, b) in enumerate(arm_coords):
        is_track_interior[a:b+1] = True

    transmat = np.identity(n_states)
    transmat[~is_track_interior] = 0
    transmat[:, ~is_track_interior] = 0

    return _normalize_row_probability(transmat)


def strong_diagonal_discrete(n_states, diag):

    strong_diagonal = np.identity(n_states) * diag
    is_off_diag = ~np.identity(n_states, dtype=bool)
    strong_diagonal[is_off_diag] = (
        (1 - diag) / (n_states - 1))
    return strong_diagonal


CONTINUOUS_TRANSITIONS = {
    'random_walk': random_walk,
    'uniform': uniform,
    'identity': identity,
}


DISCRETE_TRANSITIONS = {
    'strong_diagonal': strong_diagonal_discrete,
}