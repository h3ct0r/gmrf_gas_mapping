#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
import numpy as np
from copy import deepcopy
from itertools import product
import cv2
import matplotlib.pyplot as plt
from t_observation_gmrf import TObservationGMRF
from t_random_field_cell import TRandomFieldCell
from td_kernel_dmvw import TDKernelDMVW
import collections
import time


class OccupancyGridManagerGKernel(object):
    def __init__(self, oc_map, target_resolution=0.2):
        self._reference_frame = None
        self._lambdaPrior = 0.5  # The information (Lambda) of prior factors
        self._lambdaObs = 10.0  # The initial information (Lambda) of each observation (will decrease over time)
        self._lambdaObsLoss = 0.0  # The loss of information (Lambda) of the observations with each iteration

        # OccupancyGrid starts on lower left corner
        self._original_map = None
        self._original_grid_data = None
        self._original_occ_grid_metadata = None

        self._target_map = None
        self._target_grid_data = None
        self._target_occ_grid_metadata = None

        self._target_resolution = float(target_resolution)
        self.res_coefficient = 0.0
        self.cell_interconnections = dict()

        self._init_occ_grid(oc_map)
        self._prepare_connectivity()

        self.gas_measurements = dict()

        # sample_observations = [
        #     [0.153, 3.4123405647476, -1.1078826220720974],
        #     [0.153, 3.4123405647476, -1.1078826220720974],
        #     [0.434355298, 3.4123405647476, -1.1078826220720974],
        #     [0.080, 3.4123405647476, -1.1078826220720974],
        #     [0.153, 3.4123405647476, -1.1078826220720974],
        #     [0.1090, 3.4123405647476, -1.1078826220720974],
        #     [0.153, 3.4123405647476, -1.1078826220720974],
        #     [0.153, 3.4123405647476, -1.1078826220720974],
        #     [0.153, 3.4123405647476, -1.1078826220720974],
        #     [0.77, 4.016034043884688, -1.0139397066082456],
        #     [0.20505, 4.001193773296208, -1.0400150643866275],
        #     [0.012320106228192648, 4.275501017866776, -0.4434828558444557]
        # ]
        #
        # for e in sample_observations:
        #     self.insert_observation(e[0], e[1], e[2])

        # Set parameters
        min_x = 0
        min_y = 0
        max_x = self.width
        max_y = self.height

        cell_size = 1
        kernel_size = 3 * cell_size
        wind_scale = 0.05
        time_scale = 0.001
        evaluation_radius = 2 * kernel_size
        self.local_mean_map = None

        # call Kernel
        self.kernel = TDKernelDMVW(min_x, min_y, max_x, max_y, cell_size, kernel_size, wind_scale, time_scale,
                              low_confidence_calculation_zero=True, evaluation_radius=evaluation_radius,
                              cell_interconnections=self.cell_interconnections)

        rospy.loginfo("Height (y / rows): " + str(self.height) +
                      ", Width (x / columns): " + str(self.width) +
                      ", starting from bottom left corner of the grid. " +
                      " Reference_frame: " + str(self.reference_frame) +
                      " target resolution: " + str(self._target_resolution) +
                      " origin: \n" + str(self.origin))

    def update_map(self, oc_map):
        self._init_occ_grid(oc_map)

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
        for i in xrange(h):
            for j in xrange(w):
                if data[i * w + j] >= 50:
                    img[i, j] = 0
                elif 0 < data[i * w + j] < 50:
                    img[i, j] = 255
                elif data[i * w + j] == -1:
                    img[i, j] = 205

        return img

    @property
    def resolution(self):
        return self._target_occ_grid_metadata.resolution

    @property
    def width(self):
        return self._target_occ_grid_metadata.width

    @property
    def height(self):
        return self._target_occ_grid_metadata.height

    @property
    def origin(self):
        return self._target_occ_grid_metadata.origin

    @property
    def reference_frame(self):
        return self._reference_frame

    def _init_occ_grid(self, msg):
        rospy.loginfo("Got a full OccupancyGrid update")

        # Contains resolution, width & height
        # np.set_printoptions(threshold=99999999999, linewidth=200)
        # data comes in row-major order http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html
        # first index is the row, second index the column
        self._original_grid_data = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self._original_occ_grid_metadata = msg.info
        self._reference_frame = msg.header.frame_id
        self._original_map = msg

        self._prepare_target_map(msg)

        # img1 = self.map_to_img(self._original_map)
        # img2 = self.map_to_img(self._target_map)
        # cv2.imshow("original", img1)
        # cv2.imshow("target", img2)
        # cv2.waitKey(0)

        # plt.imshow(self._target_grid_data)
        # plt.show()

    def _prepare_target_map(self, msg):

        def scale(im, nR, nC):
            """
            Scale image
            :param im: matrix image
            :param nR: n rows
            :param nC: n columns
            :return:
            """
            nR0 = len(im)  # source number of rows
            nC0 = len(im[0])  # source number of columns
            return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                     for c in range(nC)] for r in range(nR)]

        orig_min_x = msg.info.origin.position.x
        orig_max_x = msg.info.origin.position.x + msg.info.width * msg.info.resolution
        orig_min_y = msg.info.origin.position.y
        orig_max_y = msg.info.origin.position.y + msg.info.height * msg.info.resolution

        # Adjust sizes to adapt them to full sized cells according to the resolution:
        t_min_x = self._target_resolution * round(orig_min_x / self._target_resolution, 2)
        t_max_x = self._target_resolution * round(orig_max_x / self._target_resolution, 2)
        t_min_y = self._target_resolution * round(orig_min_y / self._target_resolution, 2)
        t_max_y = self._target_resolution * round(orig_max_y / self._target_resolution, 2)
        self.res_coefficient = self._target_resolution / round(msg.info.resolution, 2)  # prevent resolution round error

        t_size_x = int(round((t_max_x - t_min_x) / self._target_resolution))
        t_size_y = int(round((t_max_y - t_min_y) / self._target_resolution))

        target_map = deepcopy(msg)
        target_map.info.resolution = self._target_resolution
        target_map.info.width = t_size_x
        target_map.info.height = t_size_y

        # scale the map to the desired resolution
        scaled_2d_map = scale(self._original_grid_data, t_size_y, t_size_x)
        target_map.data = np.array(scaled_2d_map).flatten()

        self._target_grid_data = np.array(target_map.data, dtype=np.int8).reshape(target_map.info.height,
                                                                                  target_map.info.width)
        self._target_occ_grid_metadata = target_map.info
        self._target_map = target_map

    def _prepare_connectivity(self):

        def check_relation_between_cells(u, v):
            if not self.is_in_gridmap(u[0], u[1]) or not self.is_in_gridmap(v[0], v[1]):
                return False

            if abs(self.get_cost_from_costmap_x_y(u[0], u[1]) - self.get_cost_from_costmap_x_y(v[0], v[1])) > 50.0:
                return False

            return True

        for i in xrange(self.width):
            for j in xrange(self.height):

                for win_i in [-1, 0, 1]:
                    for win_j in [-1, 0, 1]:

                        # remove the center field
                        if win_i == 0 and win_j == 0:
                            continue

                        # remove diagonals
                        if (win_i == -1 and win_j == -1) or \
                                (win_i == -1 and win_j == 1) or \
                                (win_i == 1 and win_j == -1) or \
                                (win_i == 1 and win_j == 1):
                            continue

                        u = (i, j)
                        v = (i + win_i, j + win_j)

                        if check_relation_between_cells(u, v):
                            if u not in self.cell_interconnections:
                                self.cell_interconnections[u] = set()

                            self.cell_interconnections[u].add(v)

                            if v not in self.cell_interconnections:
                                self.cell_interconnections[v] = set()

                            self.cell_interconnections[v].add(u)


        # debug cell interconnections
        with open("/tmp/GMRF_v2.txt", 'w') as f:
            for k, value in self.cell_interconnections.items():
                for cv in value:
                    f.write('{} {} {} {}\n'.format(k[0], k[1], cv[0], cv[1]))

    def get_world_x_y(self, costmap_x, costmap_y):
        world_x = costmap_x * self.resolution + self.origin.position.x
        world_y = costmap_y * self.resolution + self.origin.position.y
        return world_x, world_y

    def get_costmap_x_y(self, world_x, world_y):
        costmap_x = int(
            round((world_x - self.origin.position.x) / self.resolution))
        costmap_y = int(
            round((world_y - self.origin.position.y) / self.resolution))
        return costmap_x, costmap_y

    def get_cost_from_world_x_y(self, x, y):
        cx, cy = self.get_costmap_x_y(x, y)
        try:
            return self.get_cost_from_costmap_x_y(cx, cy)
        except IndexError as e:
            raise IndexError("Coordinates out of grid (in frame: {}) x: {}, y: {} must be in between: [{}, {}], [{}, {}]. Internal error: {}".format(
                self.reference_frame, x, y,
                self.origin.position.x,
                self.origin.position.x + self.height * self.resolution,
                self.origin.position.y,
                self.origin.position.y + self.width * self.resolution,
                e))

    def get_cost_from_costmap_x_y(self, x, y):
        if self.is_in_gridmap(x, y):
            # data comes in row-major order http://docs.ros.org/en/melodic/api/nav_msgs/html/msg/OccupancyGrid.html
            # first index is the row, second index the column
            return self._target_grid_data[y][x]
        else:
            raise IndexError(
                "Coordinates out of gridmap, x: {}, y: {} must be in between: [0, {}], [0, {}]".format(
                    x, y, self.height, self.width))

    def is_in_gridmap(self, x, y):
        if -1 < x < self.width and -1 < y < self.height:
            return True
        else:
            return False

    def get_closest_cell_under_cost(self, x, y, cost_threshold, max_radius):
        """
        Looks from closest to furthest in a circular way for the first cell
        with a cost under cost_threshold up until a distance of max_radius,
        useful to find closest free cell.
        returns -1, -1 , -1 if it was not found.
        :param x int: x coordinate to look from
        :param y int: y coordinate to look from
        :param cost_threshold int: maximum threshold to look for
        :param max_radius int: maximum number of cells around to check
        """
        return self._get_closest_cell_arbitrary_cost(
            x, y, cost_threshold, max_radius, bigger_than=False)

    def get_closest_cell_over_cost(self, x, y, cost_threshold, max_radius):
        """
        Looks from closest to furthest in a circular way for the first cell
        with a cost over cost_threshold up until a distance of max_radius,
        useful to find closest obstacle.
        returns -1, -1, -1 if it was not found.
        :param x int: x coordinate to look from
        :param y int: y coordinate to look from
        :param cost_threshold int: minimum threshold to look for
        :param max_radius int: maximum number of cells around to check
        """
        return self._get_closest_cell_arbitrary_cost(
            x, y, cost_threshold, max_radius, bigger_than=True)

    def get_cell_interconnections(self):
        return self.cell_interconnections

    def _get_closest_cell_arbitrary_cost(self, x, y,
                                         cost_threshold, max_radius,
                                         bigger_than=False):

        # Check the actual goal cell
        try:
            cost = self.get_cost_from_costmap_x_y(x, y)
        except IndexError:
            return None

        if bigger_than:
            if cost > cost_threshold:
                return x, y, cost
        else:
            if cost < cost_threshold:
                return x, y, cost

        def create_radial_offsets_coords(radius):
            """
            Creates an ordered by radius (without repetition)
            generator of coordinates to explore around an initial point 0, 0
            For example, radius 2 looks like:
            [(-1, -1), (-1, 0), (-1, 1), (0, -1),  # from radius 1
            (0, 1), (1, -1), (1, 0), (1, 1),  # from radius 1
            (-2, -2), (-2, -1), (-2, 0), (-2, 1),
            (-2, 2), (-1, -2), (-1, 2), (0, -2),
            (0, 2), (1, -2), (1, 2), (2, -2),
            (2, -1), (2, 0), (2, 1), (2, 2)]
            """
            # We store the previously given coordinates to not repeat them
            # we use a Dict as to take advantage of its hash table to make it more efficient
            coords = {}
            # iterate increasing over every radius value...
            for r in range(1, radius + 1):
                # for this radius value... (both product and range are generators too)
                tmp_coords = product(range(-r, r + 1), repeat=2)
                # only yield new coordinates
                for i, j in tmp_coords:
                    if (i, j) != (0, 0) and not coords.get((i, j), False):
                        coords[(i, j)] = True
                        yield (i, j)

        coords_to_explore = create_radial_offsets_coords(max_radius)

        for idx, radius_coords in enumerate(coords_to_explore):
            # for coords in radius_coords:
            tmp_x, tmp_y = radius_coords
            # print("Checking coords: " +
            #       str((x + tmp_x, y + tmp_y)) +
            #       " (" + str(idx) + " / " + str(len(coords_to_explore)) + ")")
            try:
                cost = self.get_cost_from_costmap_x_y(x + tmp_x, y + tmp_y)
            # If accessing out of grid, just ignore
            except IndexError:
                pass
            if bigger_than:
                if cost > cost_threshold:
                    return x + tmp_x, y + tmp_y, cost

            else:
                if cost < cost_threshold:
                    return x + tmp_x, y + tmp_y, cost

        return -1, -1, -1

    def insert_observation(self, norm_reading, wx, wy):
        try:
            mx, my = self.get_costmap_x_y(wx, wy)
            p_key = (mx, my)

            #p_key = (wx, wy)
            if p_key not in self.gas_measurements:
                self.gas_measurements[p_key] = collections.deque(maxlen=30)

            self.gas_measurements[p_key].append({"measurement": norm_reading, "timestamp": int(time.time())})
        except Exception as ex:
            rospy.logwarn(ex)

    def update_map_estimation(self):

        def scale(im, nR, nC):
            """
            Scale image
            :param im: matrix image
            :param nR: n rows
            :param nC: n columns
            :return:
            """
            nR0 = len(im)  # source number of rows
            nC0 = len(im[0])  # source number of columns
            return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                     for c in range(nC)] for r in range(nR)]

        if len(self.gas_measurements.keys()) < 3:
            rospy.logwarn("measurements less than 3")
            return

        positions_x = []
        positions_y = []
        concentrations = []
        wind_directions = []
        wind_speeds = []
        timestamps = []

        # Create dummy measurement vectors
        for k in self.gas_measurements.keys():
            v = self.gas_measurements[k]
            x, y = k
            for e in v:
                measurement = e["measurement"]
                timestamp = e["timestamp"]

                positions_x.append(x)
                positions_y.append(y)
                concentrations.append(measurement)
                wind_directions.append(0)
                wind_speeds.append(0)
                timestamps.append(timestamp)

        self.kernel.set_measurements(positions_x, positions_y, concentrations, timestamps, wind_speeds, wind_directions)
        self.kernel.calculate_maps()

        # Show result as map
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('Kernel DM+V')
        target_grid_data = self._target_grid_data.copy()

        ax1.set_aspect(1.0)
        ax1.title.set_text("mean map")
        ax1.imshow(target_grid_data, alpha=0.4)
        # print("self.kernel.cell_grid_x", self.kernel.cell_grid_x)
        # print("self.kernel.cell_grid_y", self.kernel.cell_grid_y)
        local_cell_grid_x, local_cell_grid_y = np.mgrid[self.kernel.min_x:self.kernel.max_x-1:1,
                                               self.kernel.min_y:self.kernel.max_y-1:1]
        local_mean_map = np.array(scale(self.kernel.mean_map, self.width, self.height))

        #print("local_cell_grid_x.shape", local_cell_grid_x.shape)
        #print("local_cell_grid_y.shape", local_cell_grid_y.shape)
        #print("local_mean_map.shape:", local_mean_map.shape)

        #ax1.imshow(local_mean_map, alpha=0.4)
        data = ax1.contourf(local_cell_grid_x, local_cell_grid_y, local_mean_map, alpha=0.6)
        self.local_mean_map = local_mean_map

        where_are_NaNs = np.isnan(self.local_mean_map)
        self.local_mean_map[where_are_NaNs] = 0

        #data = ax1.contourf(self.kernel.cell_grid_x, self.kernel.cell_grid_y, self.kernel.mean_map, alpha=0.6)
        #plt.colorbar(data, ax=ax1)
        #target_grid_data = cv2.cvtColor(target_grid_data, cv2.COLOR_GRAY2BGR)
        #ax1.colorbar()

        ax2.set_aspect(1.0)
        ax2.title.set_text("variance map")
        ax2.imshow(target_grid_data, alpha=0.4)
        ax2.contourf(self.kernel.cell_grid_x, self.kernel.cell_grid_y, self.kernel.variance_map, alpha=0.6)
        #ax2.colorbar()

        ax3.set_aspect(1.0)
        ax3.title.set_text("confidence map")
        ax3.imshow(target_grid_data, alpha=0.4)
        ax3.contourf(self.kernel.cell_grid_x, self.kernel.cell_grid_y, self.kernel.confidence_map, alpha=0.6)
        #ax3.colorbar()

        #plt.draw()
        #plt.pause(0.001)
        #plt.show()
        plt.savefig('/tmp/kernel_dmv.jpg', dpi=fig.dpi)
        plt.close()

        #print(image_from_plot.shape)

    # def update_map_estimation(self):
    #
    #     def map_to_img(occ_grid):
    #         """ convert nav_msgs/OccupancyGrid to OpenCV mat
    #             small noise in the occ grid is removed by
    #             thresholding on the occupancy probability (> 50%)
    #         """
    #
    #         data = occ_grid.data
    #         w = occ_grid.info.width
    #         h = occ_grid.info.height
    #
    #         img = np.zeros((h, w, 1), np.uint8)
    #         img += 255  # start with a white canvas instead of a black one
    #
    #         # occupied cells (0 - 100 prob range)
    #         # free cells (0)
    #         # unknown -1
    #         for i in range(0, h):
    #             for j in range(0, w):
    #                 if data[i * w + j] >= 50:
    #                     img[i, j] = 0
    #                 elif 0 < data[i * w + j] < 50:
    #                     img[i, j] = 255
    #                 elif data[i * w + j] == -1:
    #                     img[i, j] = 205
    #
    #         # crop borders if performing map stitching
    #         # img = img[20:380, 20:380]
    #         return img
    #
    #     def valid_imshow_data(data):
    #         data = np.asarray(data)
    #         if data.ndim == 2:
    #             return True
    #         elif data.ndim == 3:
    #             if 3 <= data.shape[2] <= 4:
    #                 return True
    #             else:
    #                 print('The "data" has 3 dimensions but the last dimension '
    #                       'must have a length of 3 (RGB) or 4 (RGBA), not "{}".'
    #                       ''.format(data.shape[2]))
    #                 return False
    #         else:
    #             print('To visualize an image the data must be 2 dimensional or '
    #                   '3 dimensional, not "{}".'
    #                   ''.format(data.ndim))
    #             return False
    #
    #     import cv2
    #
    #     local_obs = self.active_obs.copy()
    #     local_obs = (local_obs - np.min(local_obs)) / (3.0 - np.min(local_obs))
    #
    #     kernel = cv2.getGaussianKernel(11, 2.0)
    #     local_obs = cv2.filter2D(local_obs, -1, kernel)
    #
    #     img = map_to_img(self._target_map)
    #     #cv2.imshow("local_obs", local_obs)
    #     #cv2.waitKey(0)
    #
    #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #     #valid_imshow_data(img)
    #
    #     local_obs_img = cv2.cvtColor(local_obs, cv2.COLOR_GRAY2BGR)
    #
    #     # cv2.imshow("img", img)
    #     # cv2.waitKey(0)
    #
    #     #plt.imshow(img, cmap='hot', interpolation='nearest')
    #     plt.imshow(local_obs_img)
    #     plt.show()
    #     pass

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