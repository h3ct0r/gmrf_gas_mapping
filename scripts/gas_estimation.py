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


class GasEstimation(object):
    def __init__(self):
        self._costmap_sub_topic_name = '/map'
        self._gas_sensor_sub_topic_name = '/PID/Sensor_reading'
        self._min_sensor_val = 0.0
        self._max_sensor_val = 3.0
        self._frame_id = "/map"

        self.GMRF_lambdaPrior = 0.0  # The information (Lambda) of prior factors
        self.GMRF_lambdaObs = 0.0  # The initial information (Lambda) of each observation (this will decrease with time)
        self.GMRF_lambdaObsLoss = 0.0  # The loss of information (Lambda) of the observations with each iteration

        self.lock = Lock()
        self.listener = tf.TransformListener()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.map_min_x = 0.0
        self.map_max_x = 0.0
        self.map_min_y = 0.0
        self.map_max_y = 0.0

        self.latest_costmap = None
        self.gas_measurements = dict()

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

        while not self.is_initialized():
            rospy.logwarn("Waiting for initialization...")
            rate.sleep()

        while not rospy.is_shutdown():
            if self.latest_costmap is None:
                rate.sleep()
                continue

            self.gas_estimation()
            self.plot_gas_points()

            rate.sleep()

    def plot_gas_points(self):
        plt.scatter([self.map_min_x, self.map_min_x, self.map_max_x, self.map_max_x],
                    [self.map_max_y, self.map_min_y, self.map_max_y, self.map_min_y])

        gas_sizes = []
        gas_x = []
        gas_y = []
        for k in self.gas_measurements.keys():
            gas_x.append(k[0])
            gas_y.append(k[1])
            gas_sizes.append(np.mean(self.gas_measurements[k]) * 100)

        plt.scatter(gas_x, gas_y, s=gas_sizes, color="darkorange")
        plt.show()

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

        self.map_min_x = msg.info.origin.position.x
        self.map_max_x = msg.info.origin.position.x + msg.info.width * msg.info.resolution
        self.map_min_y = msg.info.origin.position.y
        self.map_max_y = msg.info.origin.position.y + msg.info.height * msg.info.resolution

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
            raise ValueError("[GMRF] cannot find transformation between gas sensor and map frame_id")

        # if there is no map to process
        if self.latest_costmap is None:
            return

        x_pos = trans[0]
        y_pos = trans[1]

        # add new observation to the map
        if curr_reading < 0 or curr_reading > 1:
            rospy.logwarn("[GMRF] Obs is out of bouns! %.2f [0,1]. Normalizing!", curr_reading)
            curr_reading = 1.0

        rospy.loginfo("[GMRF] New obs: %.2f at (%.2f,%.2f)", curr_reading, x_pos, y_pos)

        p_key = (x_pos, y_pos)
        if p_key not in self.gas_measurements:
            self.gas_measurements[p_key] = collections.deque(maxlen=10)

        self.gas_measurements[p_key].append(curr_reading)

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
        """ use opencv to filter the occupancy grid and extract
            a workable skeleton of the traversable area
            intersections are detected by a library of possible intersections
        """
        costmap_mat = self.map_to_img(self.latest_costmap)

        _, occ_area = cv2.threshold(costmap_mat, 100, 255, cv2.THRESH_BINARY_INV)
        _, free_area = cv2.threshold(costmap_mat, 250, 255, cv2.THRESH_BINARY)

        #cv2.imshow("occ_area", occ_area)
        #cv2.imshow("free_area", free_area)

        #cv2.waitKey(0)


def main():
    rospy.init_node('gas_estimation')
    manager = GasEstimation()
    manager.run()


if __name__ == '__main__':
    main()
