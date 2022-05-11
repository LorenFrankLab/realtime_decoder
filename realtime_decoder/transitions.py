import numpy as np

from realtime_decoder import utils

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