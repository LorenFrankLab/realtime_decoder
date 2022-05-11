import numpy as np

from typing import List

from realtime_decoder import base, datatypes

class PositionBinStruct(object):
    def __init__(self, lower_bound, upper_bound, num_bins:int):

        self.pos_range = [lower_bound, upper_bound]
        self.num_bins = num_bins
        self.pos_bin_edges = np.linspace(
            lower_bound, upper_bound, num_bins + 1, endpoint=True, retstep=False
        )
        self.pos_bin_centers = (self.pos_bin_edges[:-1] + self.pos_bin_edges[1:]) / 2
        self.pos_bin_delta = self.pos_bin_centers[1] - self.pos_bin_centers[0]

    def which_bin(self, pos):
        return np.nonzero(np.diff(self._pos_bin_edges > pos))

    def get_bin(self, pos):

        return int((pos - self.pos_range[0])/self.pos_bin_delta)

class TrodesPositionMapper(base.PositionMapper):

    def __init__(self, arm_ids:List[int], arm_coords:List[List]):

        super().__init__()
        
        self._arm_ids = arm_ids
        self._arm_coords = arm_coords

        self._seg_to_arm_map = {}
        for segment, arm in enumerate(self._arm_ids):
            self._seg_to_arm_map[segment] = arm

        self._bin_info = {}
        for arm_ind, (a, b) in enumerate(arm_coords):
            # position bin bounds are [a, b] (inclusive)
            self._bin_info[arm_ind] = {}
            self._bin_info[arm_ind]['bins'] = np.arange(a, b+1)
            self._bin_info[arm_ind]['norm_edges'] = np.linspace(0, 1, (b-a+1)+1)


    def map_position(self, datapoint:datatypes.CameraModulePoint):

        segment = datapoint.segment
        segment_pos = datapoint.position
        
        arm = self._seg_to_arm_map[segment]
        bins = self._bin_info[arm]['bins']
        norm_edges = self._bin_info[arm]['norm_edges']

        # in general segment positions x are assigned to a position bin
        # such that bin_edge_lower <= x < bin_edge_upper
        bin_ind = np.searchsorted(norm_edges, segment_pos, side='right') - 1

        # the exception is the last bin, where we can have
        # bin_edge_lower <= x <= bin_edge_upper 
        if bin_ind > len(bins) - 1:
            bin_ind = len(bins) - 1

        return bins[bin_ind]


class KinematicsEstimator(object):

    def __init__(
        self, *, scale_factor=1, dt=1,
        xfilter=None, yfilter=None,
        speedfilter=None
    ):
        self._sf = scale_factor
        self._dt = dt

        self._b_x = np.array(xfilter)
        self._b_y = np.array(yfilter)
        self._b_speed = np.array(speedfilter)

        self._buf_x = np.zeros(self._b_x.shape[0])
        self._buf_y = np.zeros(self._b_y.shape[0])
        self._buf_speed = np.zeros(self._b_speed.shape[0])

        self._last_x = -1
        self._last_y = -1
        self._last_speed = -1

    def compute_kinematics(
        self, x, y, *, smooth_x=False,
        smooth_y=False, smooth_speed=False
    ):

        # very first datapoint
        if self._last_speed == -1:
            self._last_x = x
            self._last_y = y
            self._last_speed = 0
            return x, y, 0

        if smooth_x:
            xv = self._smooth(x * self._sf, self._b_x, self._buf_x)
        else:
            xv = x

        if smooth_y:
            yv = self._smooth(y * self._sf, self._b_y, self._buf_y)
        else:
            yv = y

        sv = np.sqrt((yv - self._last_y)**2 + (xv - self._last_x)**2) / self._dt
        if smooth_speed:
            sv = self._smooth(sv, self._b_speed, self._buf_speed)

        # now that the speed has been estimated, the current x and y values
        # become the most recent (last) x and y values
        self._last_x = xv
        self._last_y = yv
        self._last_speed = sv

        return xv, yv, sv

    def _smooth(self, newval, coefs, buf):

        # mutates data!
        buf[1:] = buf[:-1]
        buf[0] = newval
        rv = np.sum(coefs * buf, axis=0)

        return rv