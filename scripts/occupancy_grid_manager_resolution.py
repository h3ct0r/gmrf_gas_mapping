#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
from map_msgs.msg import OccupancyGridUpdate
import numpy as np
from copy import deepcopy
from itertools import product
import cv2
import matplotlib.pyplot as plt


class OccupancyGridManagerResolution(object):
    def __init__(self, oc_map, target_resolution):
        self._reference_frame = None

        # OccupancyGrid starts on lower left corner
        self._original_map = None
        self._original_grid_data = None
        self._original_occ_grid_metadata = None

        self._target_map = None
        self._target_grid_data = None
        self._target_occ_grid_metadata = None

        self._target_resolution = float(target_resolution)
        self.res_coefficient = 0.0
        self.cell_interconnections = set()

        self._init_occ_grid(oc_map)
        self._prepare_connectivity()
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
                        u = (i, j)
                        v = (i + win_i, j + win_j)

                        if check_relation_between_cells(u, v):
                            self.cell_interconnections.add((u, v))
                            self.cell_interconnections.add((v, u))

        print(self.cell_interconnections)
        with open("/tmp/GMRF_v2.txt", 'w') as f:
            for e in self.cell_interconnections:
                f.write('{} {} {} {}\n'.format(e[0][0], e[0][1], e[1][0], e[1][1]))

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
