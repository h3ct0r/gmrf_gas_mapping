#!/usr/bin/env python

from copy import copy
import rospy
from threading import Lock
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
import time
import collections
from olfaction_msgs.msg import gas_sensor
import tf
import collections
import matplotlib.pyplot as plt
import math


class TRandomFieldCell(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def set_mean(self, mean):
        self.mean = mean

    def set_std(self, std):
        self.std = std

    def get_mean(self):
        return self.mean

    def get_std(self):
        return self.std


class TObservationGMRF(object):
    def __init__(self, obs_value, obs_lambda, time_invariant):
        self.obs_value = obs_value
        self.obs_lambda = obs_lambda
        self.time_invariant = time_invariant

    def set_obs_value(self, obs_value):
        self.obs_value = obs_value

    def set_obs_lambda(self, obs_lambda):
        self.obs_lambda = obs_lambda

    def set_time_invariant(self, time_invariant):
        self.time_invariant = time_invariant

    def get_obs_value(self):
        return self.obs_value

    def get_obs_lambda(self):
        return self.obs_lambda

    def get_time_invariant(self):
        return self.time_invariant


class GMRFMap(object):
    def __init__(self, oc_map, cell_size=0.2):
        self._oc_map = oc_map

        self.GMRF_lambdaPrior = 0.5  # The information (Lambda) of prior factors
        self.GMRF_lambdaObs = 10.0  # The initial information (Lambda) of each observation (this will decrease with time)
        self.GMRF_lambdaObsLoss = 0.0  # The loss of information (Lambda) of the observations with each iteration

        rospy.loginfo("[CGMRF] m_resolution=%.2f GMRF_lambdaPrior=%.2f", cell_size, self.GMRF_lambdaPrior)

        self.map_min_x = oc_map.info.origin.position.x
        self.map_max_x = oc_map.info.origin.position.x + oc_map.info.width * oc_map.info.resolution
        self.map_min_y = oc_map.info.origin.position.y
        self.map_max_y = oc_map.info.origin.position.y + oc_map.info.height * oc_map.info.resolution

        rospy.loginfo("[CGMRF] Map cells=%.2f x=(%.2f,%.2f) y=(%.2f,%.2f)", len(self._oc_map.data), self.map_min_x, self.map_min_y,
                      self.map_max_x, self.map_max_y)

        # Adjust sizes to adapt them to full sized cells acording to the resolution:
        m_x_min = cell_size * round((self.map_min_x / float(cell_size)) + 0.1)
        m_y_min = cell_size * round(self.map_min_y / float(cell_size))
        m_x_max = cell_size * round(self.map_max_x / float(cell_size))
        m_y_max = cell_size * round(self.map_max_y / float(cell_size))
        self.res_coef = cell_size / round(oc_map.info.resolution, 2)

        rospy.loginfo("[CGMRF] map resolution:%.20f", oc_map.info.resolution)

        rospy.loginfo("[CGMRF] DEBUG ROUND x=(%.2f,%.2f) y=(%.2f,%.2f)", round(self.map_min_x / float(cell_size)),
                 round(self.map_min_y / float(cell_size)),
                 round(self.map_max_x / float(cell_size)),
                 round(self.map_max_y / float(cell_size)))

        rospy.loginfo("[CGMRF] Map x=(%.2f,%.2f) y=(%.2f,%.2f)", m_x_min, m_y_min,
                 m_x_max, m_y_max)

        rospy.loginfo("[CGMRF] res_coef=%.2f", self.res_coef)

        # Now the number of cells should be integers:
        self.m_size_x = round((m_x_max - m_x_min) / float(cell_size))
        self.m_size_y = round((m_y_max - m_y_min) / float(cell_size))
        self.N = int(self.m_size_x * self.m_size_y)
        rospy.loginfo("[CGMRF] m_size_x=%.2f m_size_y=%.2f", self.m_size_x, self.m_size_y)

        self.m_map = [TRandomFieldCell(0.0, 0.0) for i in xrange(self.N)]
        rospy.loginfo("[CGMRF] Map created: %u cells (N=%u), x=(%.2f,%.2f) y=(%.2f,%.2f)",
                      len(self.m_map), self.N, m_x_min, m_x_max, m_y_min, m_y_max)

        # init random field
        rospy.loginfo("[CGMRF] Generating Prior based on 'Squared Differences'")
        # Set initial factors: L "prior factors" + 0 "Observation factors"
        # L = (Nr - 1) * Nc + Nr * (Nc - 1); full connected
        self.nPriorFactors = (self.m_size_x - 1) * self.m_size_y + self.m_size_x * (self.m_size_y - 1)  # L
        self.nObsFactors = 0  # M
        self.nFactors = self.nPriorFactors + self.nObsFactors  # L + M
        rospy.loginfo("[CGMRF] %lu factors for a map size of N=%lu", self.nFactors, self.N)

        # Initialize H_prior, gradient = 0, and the vector of active observations = empty
        self.H_prior = []  # the prior part of H
        self.g = np.zeros(self.N)  # Gradient vector, initially the gradient is all 0's
        self.activeObs = [TObservationGMRF(0.0, 0.0, 0.0) for i in xrange(self.N)]  # No initial Observations

        self.cell_interconnections = set()

        # load default values for H_prior
        # Use region growing algorithm to determine the gascell interconnections (Factors)
        rospy.loginfo("[CGMRF] initalizing H prior...")
        self.init_default_hprior()
        rospy.loginfo("[CGMRF] default H prior initialized")

    def init_default_hprior(self):
        cx = 0
        cy = 0

        # for each cell in the gas_map
        for j in xrange(self.N):
            # Get cell_j indx-limits in Occuppancy gridmap
            cxoj_min = int(math.floor(cx * self.res_coef))
            cxoj_max = cxoj_min + int(math.ceil(self.res_coef - 1))
            cyoj_min = int(math.floor(cy * self.res_coef))
            cyoj_max = cyoj_min + int(math.ceil(self.res_coef - 1))

            seed_cxo = cxoj_min + int(math.ceil((self.res_coef/2)-1))
            seed_cyo = cyoj_min + int(math.ceil((self.res_coef/2)-1))

            if j % 100 == 0 and j != 0:
                rospy.loginfo("[CGMRF] j:%u", j)

            # rospy.loginfo("[CGMRF] cxoj_min:%u cxoj_max:%u cyoj_min:%u cyoj_max:%u", cxoj_min, cxoj_max, cyoj_min,
            #          cyoj_max)
            rospy.loginfo("[CGMRF] seed_cxo:%u seed_cyo:%u", seed_cxo, seed_cyo)

            # If a cell is free then add observation with very low information
            # to force non-visited cells to have a 0.0 mean
            # The map data, in row-major order, starting with (0,0).  Occupancy
            # probabilities are in the range [0,100].  Unknown is -1.
            cell_idx = int(seed_cxo + seed_cyo * self.m_size_x)
            if self._oc_map.data[cell_idx] < 50.0:
                act_ob = self.activeObs[j]
                act_ob.set_obs_value(0.0)
                act_ob.set_obs_lambda(10e-10)
                act_ob.set_time_invariant(True)  # Obs that will not dissapear with time.

            # Factor with the right node: H_ji = - Lamda_prior
            if cx < self.m_size_x - 1:
                i = j+1
                cxi = cx+1
                cyi = cy

                # Get cell_i indx-limits in Occuppancy gridmap
                cxoi_min = int(math.floor(cxi*self.res_coef))
                cxoi_max = int(cxoi_min + math.ceil(self.res_coef-1))
                cyoi_min = int(math.floor(cyi*self.res_coef))
                cyoi_max = int(cyoi_min + math.ceil(self.res_coef-1))

                objective_cxo = cxoi_min + int(math.ceil((self.res_coef/2)-1))
                objective_cyo = cyoi_min + int(math.ceil((self.res_coef/2)-1))
                rospy.loginfo("[CGMRF] objective_cyo:%u objective_cyo:%u", objective_cyo, objective_cyo)

                # Get overall indx of both cells together
                cxo_min = int(min(cxoj_min, cxoi_min))
                cxo_max = int(max(cxoj_max, cxoi_max))
                cyo_min = int(min(cyoj_min, cyoi_min))
                cyo_max = int(max(cyoj_max, cyoi_max))

                # Check using Region growing if cell j is connected to cell i (Occupancy gridmap)
                if self.exist_relation_between2cells(cxo_min, cxo_max, cyo_min, cyo_max, seed_cxo, seed_cyo,
                                                     objective_cxo, objective_cyo):
                    self.H_prior.append([i,j, -GMRF_lambdaPrior])

                    # Save relation between cells
                    self.cell_interconnections.add((j, i))
                    self.cell_interconnections.add((i, j))

            # # Factor with the upper node: H_ji = - Lamda_prior
            # if cy < (self.m_size_y - 1):
            #     i = j + self.m_size_x
            #     cxi = cx
            #     cyi = cy + 1
            #
            #     # Get cell_i indx-limits in Occuppancy gridmap
            #     cxoi_min = int(math.floor(cxi * self.res_coef))
            #     cxoi_max = int(cxoi_min + math.ceil(self.res_coef - 1))
            #     cyoi_min = int(math.floor(cyi * self.res_coef))
            #     cyoi_max = int(cyoi_min + math.ceil(self.res_coef - 1))
            #
            #     objective_cxo = cxoi_min + math.ceil(self.res_coef/2-1)
            #     objective_cyo = cyoi_min + math.ceil(self.res_coef/2-1)
            #
            #     # Get overall indx-limits of both cells together
            #     cxo_min = int(min(cxoj_min, cxoi_min))
            #     cxo_max = int(max(cxoj_max, cxoi_max))
            #     cyo_min = int(min(cyoj_min, cyoi_min))
            #     cyo_max = int(max(cyoj_max, cyoi_max))
            #
            #     # Check using Region growing if cell j is connected to cell i (Occupancy gridmap)
            #     if self.exist_relation_between2cells(cxo_min, cxo_max, cyo_min, cyo_max, seed_cxo, seed_cyo,
            #                                          objective_cxo, objective_cyo):
            #         self.H_prior.append([i, j, -GMRF_lambdaPrior])
            #
            #         # Save relation between cells
            #         self.cell_interconnections.add((j, i))
            #         self.cell_interconnections.add((i, j))

            # Factors of cell_j: H_jj = N factors * Lambda_prior
            nFactors_j = len(self.cell_interconnections)
            self.H_prior.append([i, j, nFactors_j * self.GMRF_lambdaPrior])

            # Increment j coordinates (row(x), col(y))
            cx += 1
            if cx >= self.m_size_x:
                cx = 0
                cy += 1

            #print(self.cell_interconnections)

        with open("/tmp/GMRF.txt", 'w') as f:
            for e in self.cell_interconnections:
                the_file.write('{} {}\n'.format(e[0], e[1]))

    def exist_relation_between2cells(self, cxo_min, cxo_max, cyo_min, cyo_max, seed_cxo, seed_cyo,
                                     objective_cxo, objective_cyo):

        cxo_min = max(cxo_min, 0)
        cxo_max = min(cxo_max, int(self._oc_map.info.width - 1))
        cyo_min = max(cyo_min, 0)
        cyo_max = min(cyo_max, int(self._oc_map.info.height - 1))

        if (seed_cxo < cxo_min) or (seed_cxo >= cxo_max) or (seed_cyo < cyo_min) or (seed_cyo >= cyo_max):
            # rospy.logwarn("Seed out of bounds (false)")
            return False

        if (objective_cxo < cxo_min) or (objective_cxo >= cxo_max) or (objective_cyo < cyo_min) or \
                (objective_cyo >= cyo_max):
            # rospy.logwarn("Objective out of bounds (false)")
            return False

        if (self._oc_map.data[int(seed_cxo + seed_cyo * self._oc_map.info.width)] < 50.0) != \
                (self._oc_map.data[int(objective_cxo + objective_cyo * self._oc_map.info.width)] < 50.0):
            # rospy.logwarn("Seed and objective have diff occupation (false)")
            return False

        # Create Matrix for region growing (row,col)
        mat_exp = np.zeros((cyo_max - cyo_min + 1, cxo_max - cxo_min + 1))
        mat_exp[seed_cyo - cyo_min][seed_cxo - cxo_min] = 1

        seeds_old = 0
        seeds_new = 1

        while seeds_old < seeds_new:
            seeds_old = seeds_new
            for col in xrange(mat_exp.shape[1]):
                for row in xrange(mat_exp.shape[0]):
                    # test if cell needs to be expanded
                    if mat_exp[row][col] == 1:
                        mat_exp[row][col] = 2  # mark as expanded

                        # check if neighbours have similar occupancy (expand)
                        for i in [-1, 0, 1]:
                            for j in [-1, 0, 1]:

                                # check that neighbour is inside the map
                                if 0 <= col + i < mat_exp.shape[1] and \
                                        0 <= row + j < mat_exp.shape[0]:

                                    if not ((i == 0 and j == 0) or not (mat_exp[row+j][col+i] == 0)):
                                        # check if expand
                                        idx_a = (row + cxo_min + col + cyo_min) * self._oc_map.info.width
                                        idx_b = (row + j + cxo_min + col + i + cyo_min) * self._oc_map.info.width

                                        if (idx_a < 0 or idx_a >= len(self._oc_map.data)) or \
                                            idx_b < 0 or idx_b >= len(self._oc_map.data):

                                            rospy.loginfo("[CGMRF] int(row) + j:%u int(col) + i:%u ", int(row) + i,
                                                          int(col) + j)
                                            rospy.loginfo("[CGMRF] mat_exp.shape: %s", mat_exp.shape)
                                            rospy.loginfo("[CGMRF] idx_a:%u idx_b:%u", idx_a, idx_b)
                                            rospy.loginfo("[CGMRF] cxo_min:%u cxo_max:%u cyo_min:%u cyo_max:%u",
                                                          cxo_min, cxo_max, cyo_min, cyo_max)
                                            rospy.loginfo("[CGMRF] col:%u row:%u",
                                                          col, row)
                                            rospy.loginfo("[CGMRF] i:%u j:%u",
                                                          i, j)

                                        if (self._oc_map.data[idx_a] < 50.0) == (self._oc_map.data[idx_b] < 50.0):
                                            if (row+j+cxo_min == objective_cxo) and (col+i+cyo_min == objective_cyo):
                                                # rospy.loginfo("Connection Success (true)")
                                                # Objective connected
                                                return True

                                            mat_exp[row+j][col+i] = 1
                                            seeds_new += 1

        return False

    # def plot_gas_points(self):
    #     plt.scatter([self.map_min_x, self.map_min_x, self.map_max_x, self.map_max_x],
    #                 [self.map_max_y, self.map_min_y, self.map_max_y, self.map_min_y])
    #
    #     gas_sizes = []
    #     gas_x = []
    #     gas_y = []
    #     for k in self.gas_measurements.keys():
    #         gas_x.append(k[0])
    #         gas_y.append(k[1])
    #         gas_sizes.append(np.mean(self.gas_measurements[k]) * 100)
    #
    #     plt.scatter(gas_x, gas_y, s=gas_sizes, color="darkorange")
    #     plt.show()


class GasEstimation(object):
    def __init__(self):
        self._costmap_sub_topic_name = '/map'
        self._gas_sensor_sub_topic_name = '/PID/Sensor_reading'
        self._min_sensor_val = 0.0
        self._max_sensor_val = 3.0
        self._frame_id = "/map"

        self.lock = Lock()
        self.listener = tf.TransformListener()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.latest_costmap = None
        self.gas_measurements = dict()
        self.GMRFMap = None

        # Subscribers
        self.costmap_sub = rospy.Subscriber(
            self._costmap_sub_topic_name,
            OccupancyGrid,
            self.handle_costmap_cb
        )
        self.gas_sensor_sub = rospy.Subscriber(
            self._gas_sensor_sub_topic_name,
            gas_sensor,
            self.handle_gas_sensor_cb
        )

    def run(self):
        """ main entry point """

        rate = rospy.Rate(5)

        while not self.is_initialized() and not rospy.is_shutdown():
            rospy.logwarn("Waiting for initialization...")
            rate.sleep()

        while not rospy.is_shutdown():
            if self.latest_costmap is None:
                rate.sleep()
                continue

            self.gas_estimation()
            #self.updateMapEstimation_GMRF(self.GMRF_lambdaObsLoss)
            #self.plot_gas_points()

            rate.sleep()

    def is_initialized(self):
        """ check for initial data needed for this node """

        try:
            rospy.wait_for_message(self._costmap_sub_topic_name, OccupancyGrid, timeout=5)
        except rospy.ROSException as rex:
            rospy.logwarn(rex)
            return False

        return True

    def handle_costmap_cb(self, msg):
        """ receive the occupancy grid map and register it """
        self.latest_costmap = msg
        self.GMRFMap = GMRFMap(self.latest_costmap)
        # my_map = newCGMRF_map(occupancyMap, cell_size, GMRF_lambdaPrior, colormap, max_pclpoints_cell);

    def handle_gas_sensor_cb(self, msg):
        """
        UNITS_UNKNOWN = 0
        UNITS_VOLT = 1
        UNITS_AMP = 2
        UNITS_PPM = 3
        UNITS_PPB = 4
        UNITS_OHM = 5
        UNITS_CENTIGRADE = 100
        UNITS_RELATIVEHUMIDITY = 101
        UNITS_NOT_VALID = 255
        :param msg:
        :return:
        """
        if msg.raw_units == 3:
            curr_reading = (msg.raw - self._min_sensor_val)/(self._max_sensor_val - self._min_sensor_val)
            if curr_reading < 0:
                curr_reading = 0.0
            if curr_reading > 1:
                curr_reading = 1.0
        else:
            raise ValueError("[GMRF] msg.raw_units is not valid for gas sensor")

        try:
            (trans, rot) = self.listener.lookupTransform(self._frame_id, msg.header.frame_id, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            #raise ValueError("[GMRF] cannot find transformation between gas sensor and map frame_id")
            rospy.logwarn("[GMRF] cannot find transformation between gas sensor and map frame_id")
            return

        # if there is no map to process
        if self.latest_costmap is None:
            rospy.logwarn("[GMRF] waiting for map before registering observation")
            return

        x_pos = trans[0]
        y_pos = trans[1]

        # add new observation to the map
        if curr_reading < 0 or curr_reading > 1:
            rospy.logwarn("[GMRF] Obs is out of bouns! %.2f [0,1]. Normalizing!", curr_reading)
            curr_reading = 1.0

        #rospy.loginfo("[GMRF] New obs: %.2f at (%.2f,%.2f)", curr_reading, x_pos, y_pos)

        p_key = (x_pos, y_pos)
        if p_key not in self.gas_measurements:
            self.gas_measurements[p_key] = collections.deque(maxlen=10)

        self.gas_measurements[p_key].append(curr_reading)

        # my_map->insertObservation_GMRF(curr_reading, x_pos, y_pos, GMRF_lambdaObs);

    @staticmethod
    def map_to_img(occ_grid):
        """ convert nav_msgs/OccupancyGrid to OpenCV mat
            small noise in the occ grid is removed by
            thresholding on the occupancy probability (> 50%)
        """

        data = occ_grid.data
        w = occ_grid.info.width
        h = occ_grid.info.height

        img = np.zeros((h, w, 1), np.uint8)
        img += 255  # start with a white canvas instead of a black one

        # occupied cells (0 - 100 prob range)
        # free cells (0)
        # unknown -1
        for i in range(0, h):
            for j in range(0, w):
                if data[i * w + j] >= 50:
                    img[i, j] = 0
                elif 0 < data[i * w + j] < 50:
                    img[i, j] = 255
                elif data[i * w + j] == -1:
                    img[i, j] = 205

        # crop borders if performing map stitching
        # img = img[20:380, 20:380]
        return img

    def gas_estimation(self):
        """
        """
        costmap_mat = self.map_to_img(self.latest_costmap)

        _, occ_area = cv2.threshold(costmap_mat, 100, 255, cv2.THRESH_BINARY_INV)
        _, free_area = cv2.threshold(costmap_mat, 250, 255, cv2.THRESH_BINARY)

        #cv2.imshow("occ_area", occ_area)
        #cv2.imshow("free_area", free_area)

        #cv2.waitKey(0)

    def updateMapEstimation_GMRF(self):

        pass


def main():
    rospy.init_node('gas_estimation')
    manager = GasEstimation()
    manager.run()


if __name__ == '__main__':
    main()
