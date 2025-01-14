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
import scipy
import scipy.linalg
import numdifftools as nd


class OccupancyGridManagerResolution(object):
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

        self.m_map = [[TRandomFieldCell(0.0, 0.0) for i in range(self.width)] for j in range(self.height)]
        self.active_obs = [[[TObservationGMRF(0.0, 0.0, False)] for i in range(self.width)] for j in range(self.height)]

        print("self.self.active_obs:",
              len(self.active_obs),
              len(self.active_obs[0]),
              len(self.active_obs[0][0]))

        self.h_prior = [[-self._lambdaPrior for i in range(self.width)] for j in range(self.height)]
        for j in range(self.height):
            for i in range(self.width):
                if i == j:
                    self.h_prior[j][i] = i * self._lambdaPrior

        #self.h_prior = []
        # for j in range(self.height):
        #     for i in range(self.width):
        #         if i == j:
        #             self.h_prior[j][i](j * self._lambdaPrior)
        #         # else:
        #         #     self.h_prior.append(-self._lambdaPrior)
        self.h_prior = np.array(self.h_prior)
        print(self.h_prior)

        # L = (Nr - 1) * Nc + Nr * (Nc - 1)
        self.n_prior_factors = (self.width - 1) * self.height + self.width * (self.height - 1)
        # M
        self.n_obs_factors = 0
        # L + M
        self.n_factors = self.n_prior_factors + self.n_obs_factors

        rospy.loginfo("Height (y / rows): " + str(self.height) +
                      ", Width (x / columns): " + str(self.width) +
                      ", starting from bottom left corner of the grid. " +
                      " Reference_frame: " + str(self.reference_frame) +
                      " target resolution: " + str(self._target_resolution) +
                      " origin: \n" + str(self.origin))

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

    def insert_observation_gmrf(self, norm_reading, wx, wy):
        try:
            mx, my = self.get_costmap_x_y(wx, wy)
            observation = TObservationGMRF(norm_reading, self._lambdaObs, False)  # The obs will lose weight with time.
            self.active_obs[mx][my].append(observation)
        except Exception as ex:
            rospy.logwarn(ex)

    def update_map_estimation_gmrf(self):
        # 1 - hessian
        #h_tri = np.zeros((self.height, self.width))
        N = self.width * self.height
        h_tri = np.zeros((N, N))
        h_tri[0:0 + self.h_prior.shape[0], 0:0 + self.h_prior.shape[1]] += self.h_prior
        #h_tri = self.h_prior.copy()
        for j in xrange(self.width):
            for i in xrange(self.height):
                lambda_obj_ij = 0.0
                for e in self.active_obs[i][j]:
                    lambda_obj_ij += e.get_obs_lambda()

                if lambda_obj_ij != 0.0:
                    h_tri[i][j] = lambda_obj_ij
                    # h_tri.append((i, j, lambda_obj_ij))

        # 2 - gradient
        # reset and build the gradient vector
        g = np.zeros((self.height, self.width))
        for j in xrange(self.width):
            for i in xrange(self.height):
                # a - gradient due to observations
                g[i][j] += sum([(self.m_map[i][j].mean - ob.get_obs_value()) * ob.get_obs_lambda()
                               for ob in self.active_obs[i][j]])

                # b - gradient due to prior
                # consider only cells correlated with cell ij
                u = (j, i)
                connections = self.get_cell_interconnections()[u]
                for v in connections:
                    g[i][j] += (self.m_map[i][j].mean - self.m_map[v[1]][v[0]].mean) * self._lambdaPrior

        # plt.imshow(g, cmap='hot', interpolation='nearest')
        # plt.show()
        # https://www.quantstart.com/articles/Cholesky-Decomposition-in-Python-and-NumPy/
        #L = scipy.linalg.cholesky(A, lower=True)
        #U = scipy.linalg.cholesky(A, lower=False)

        #H = hessian(h_tri)
        #H = nd.Hessian(h_tri)([1, 2])
        print("h_tri.shape:", h_tri.shape)
        U = scipy.linalg.cholesky(h_tri, lower=False)
        m_inc = np.zeros((self.height, self.width))
        sigma = np.zeros((self.height, self.width))

        pass

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