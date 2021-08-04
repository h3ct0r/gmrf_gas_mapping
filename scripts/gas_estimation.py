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
import matplotlib.pyplot as plt
import math
# from occupancy_grid_manager_resolution import OccupancyGridManagerResolution
from occupancy_grid_manager_gkernel import  OccupancyGridManagerGKernel
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class GasEstimationNode(object):
    def __init__(self):
        self._costmap_sub_topic_name = '/map'
        self._gas_sensor_sub_topic_name = '/PID/Sensor_reading'
        self._gas_var_pub_topic_name = '/gas_var_markers'

        self._min_sensor_val = 0.0
        self._max_sensor_val = 3.0
        self._frame_id = "/map"

        self.lock = Lock()
        self.listener = tf.TransformListener()
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.latest_costmap = None
        #self.gas_measurements = dict()
        self._oc_manager = None

        # subscribers
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

        # publishers
        self.var_map_pub = rospy.Publisher(self._gas_var_pub_topic_name, MarkerArray)

    def run(self):
        """ main entry point """

        rate = rospy.Rate(0.5)

        while not self.is_initialized() and not rospy.is_shutdown():
            rospy.logwarn("Waiting for initialization...")
            rate.sleep()

        # plt.ion()
        # plt.show()
        while not rospy.is_shutdown():
            if self.latest_costmap is None:
                rate.sleep()
                continue

            #self.gas_estimation()
            #self.updateMapEstimation_GMRF(self.GMRF_lambdaObsLoss)
            #self.plot_gas_points()

            self.pub_var_markers()
            self._oc_manager.update_map_estimation()
            #self._oc_manager.update_map_estimation_gmrf()

            rate.sleep()

    def is_initialized(self):
        """ check for initial data needed for this node """

        try:
            rospy.wait_for_message(self._costmap_sub_topic_name, OccupancyGrid, timeout=5)
        except rospy.ROSException as rex:
            rospy.logwarn(rex)
            return False

        if self._oc_manager is None:
            return False

        return True

    def handle_costmap_cb(self, msg):
        """ receive the occupancy grid map and register it """
        self.latest_costmap = msg
        self._oc_manager = OccupancyGridManagerGKernel(self.latest_costmap)
        # self._oc_manager = OccupancyGridManagerResolution(self.latest_costmap)

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
            rospy.logwarn("[GMRF] cannot find transformation between gas sensor and map frame_id")
            return

        # if there is no map to process
        if self.latest_costmap is None:
            rospy.logwarn("[GMRF] waiting for map before registering observation")
            return

        if self._oc_manager is None:
            rospy.logwarn("[GMRF] waiting for oc_manager before registering observation")
            return

        x_pos = trans[0]
        y_pos = trans[1]

        # add new observation to the map
        if curr_reading < 0 or curr_reading > 1:
            rospy.logwarn("[GMRF] Obs is out of bouns! %.2f [0,1]. Normalizing!", curr_reading)
            curr_reading = 1.0

        #rospy.loginfo("[GMRF] New obs: %.2f at (%.2f,%.2f)", curr_reading, x_pos, y_pos)

        #print("curr_reading:", curr_reading, "x_pos:", x_pos, "y_pos:", y_pos)

        self._oc_manager.insert_observation(curr_reading, x_pos, y_pos)
        # self._oc_manager.insert_observation_gmrf(curr_reading, x_pos, y_pos)

    # def gas_estimation(self):
    #     """
    #     """
    #     costmap_mat = self.map_to_img(self.latest_costmap)
    #
    #     _, occ_area = cv2.threshold(costmap_mat, 100, 255, cv2.THRESH_BINARY_INV)
    #     _, free_area = cv2.threshold(costmap_mat, 250, 255, cv2.THRESH_BINARY)
    #
    #     #cv2.imshow("occ_area", occ_area)
    #     #cv2.imshow("free_area", free_area)
    #
    #     #cv2.waitKey(0)

    def pub_var_markers(self):
        markerArray = MarkerArray()

        c_id = 0
        for i in xrange(self._oc_manager.width):
            for j in xrange(self._oc_manager.height):

                marker = Marker()
                marker.header.frame_id = self._frame_id
                marker.id = c_id # (i * self._oc_manager.width) + j
                marker.type = marker.SPHERE
                marker.action = marker.ADD
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.pose.orientation.w = 1.0

                wx, wy = self._oc_manager.get_world_x_y(i, j)
                marker.pose.position.x = wx
                marker.pose.position.y = wy
                marker.pose.position.z = 0.2
                markerArray.markers.append(marker)
                c_id += 1

        self.var_map_pub.publish(markerArray)


def main():
    rospy.init_node('gas_estimation_node')
    gas_node = GasEstimationNode()
    gas_node.run()


if __name__ == '__main__':
    main()
