#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tf
import rospy
import math
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry

from kalman_filter import KalmanFilter
from sphero_localization.srv import *

def pose_dist(pose1, pose2):
    """Return Euclidean distance between two ROS poses."""
    x1 = pose1.position.x
    y1 = pose1.position.y
    x2 = pose2.position.x
    y2 = pose2.position.y

    return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


class KalmanFilterNode(object):
    """
    ROS node implementation of Kalman filter.

    This node subscribes to a list of all existing Sphero's positions
    broadcast from OptiTrack system, associates one of them to the Sphero in
    the same namespace and uses Kalman filter to output steady position and
    velocity data for other nodes.
    """

    def __init__(self):
        """Initialize agent instance, create subscribers and publishers."""
        # Initialize class variables
        self.missing_counter = 0   # Counts iterations with missing marker information
        self.pub_frequency = rospy.get_param('/ctrl_loop_freq')
        self.sub_frequency = rospy.get_param('/data_stream_freq')
        self.debug_enabled = rospy.get_param('/debug_kalman')
        self.data_associated = rospy.get_param('/data_associated')
        if self.data_associated:
            self.search_radius = 0
        else:
            self.search_radius = rospy.get_param('/associate_radius')

        self.X_est = None
        self.filter = None
        self.initial_position = None

        # Create a publisher for commands
        pub = rospy.Publisher('odom_est', Odometry, queue_size=self.pub_frequency)
        if self.debug_enabled:
            # Debug publisher runs at the same frequency as incoming data.
            self.debug_pub = rospy.Publisher('debug_est', Odometry, queue_size=self.sub_frequency)

        # Create subscribers
        if self.data_associated:
            rospy.Subscriber('odom', Odometry, self.sensor_callback, queue_size=self.sub_frequency)
        else:
            rospy.Subscriber('odom', PoseArray, self.sensor_callback, queue_size=self.sub_frequency)

        # Get the initial positions of the robots.
        self.get_initial_position()  # Get initial position
        rospy.loginfo(rospy.get_namespace() + ' Initial position:\n%s\n', self.initial_position.position)

        # Initialize Kalman filter and estimation
        self.filter = KalmanFilter(1.0 / self.sub_frequency, self.initial_position)
        self.X_est = Odometry()
        self.X_est.pose.pose = self.initial_position

        # Create tf broadcaster
        br = tf.TransformBroadcaster()

        # Main while loop.
        rate = rospy.Rate(self.pub_frequency)
        while not rospy.is_shutdown():
            pub.publish(self.X_est)
            pos = self.X_est.pose.pose.position
            br.sendTransform((pos.x, pos.y, pos.z),
                             (0, 0, 0, 1),
                             rospy.Time.now(),
                             rospy.get_namespace() + 'base_link',
                             'map')
            rospy.logdebug(' x = % 7.5f', self.X_est.pose.pose.position.x)
            rospy.logdebug(' y = % 7.5f', self.X_est.pose.pose.position.y)
            rospy.logdebug('vx = % 7.5f', self.X_est.twist.twist.linear.x)
            rospy.logdebug('vy = % 7.5f\n', self.X_est.twist.twist.linear.y)
            rate.sleep()

    def get_initial_position(self):
        """Calls service which returns Sphero's initial position."""
        # If IDs
        if self.data_associated:
            while self.initial_position is None and not rospy.is_shutdown():
                rospy.sleep(0.1)
        else:
            rospy.wait_for_service('/return_initials')
            try:
                get_initials = rospy.ServiceProxy('/return_initials', ReturnInitials)
                response = get_initials(rospy.get_namespace())
                self.initial_position = response.initial
            except rospy.ServiceException as e:
                rospy.logerr(rospy.get_name() + ": Service call failed: %s", e)

    def associate(self, data):
        """
        Associate Sphero with its position and return it.

        Positions of all Spheros are determined using OptiTrack system and sent
        via mocap_node. It is currently impossible to label OptiTrack markers
        with some ID. Positions of all markers arrive in an unsorted list. We
        must associate each of the positions in list with a Sphero. To do this,
        we are looking which of the positions in the list is less than a Sphero
        radius away from the last available Sphero's position estimation. We
        assume that Sphero couldn't have traveled more than this distance
        between two consecutive position updates. OptiTrack is set to broadcast
        positions 100 times per second.
        """
        X_measured = None
        for pose in data.poses:
            if pose_dist(self.X_est.pose.pose, pose) < self.search_radius:
                X_measured = pose
                self.missing_counter = 0

        if X_measured is None:
            self.missing_counter += 1
            if self.missing_counter % 10 == 0 and self.missing_counter <= 90:
                rospy.logwarn(rospy.get_name() +
                              ": Marker missing for %d consecutive iterations.",
                              self.missing_counter)
            elif self.missing_counter == 100:
                rospy.logerr(rospy.get_name() + ": Lost tracking!!")

        return X_measured

    def sensor_callback(self, data):
        """Process received positions data and return Kalman estimation."""
        if self.initial_position is None:
            self.initial_position = data.pose.pose

        if self.filter is None:
            return

        # Get measurement
        if self.data_associated:
            X_measured = data.pose.pose
            time = data.header.stamp
        else:
            X_measured = self.associate(data)

        # If measurement data is not available, use only prediction step
        # Else, use prediction and update step
        if X_measured is None:
            self.X_est = self.filter.predict()
        else:
            self.X_est = self.filter.predict_update(X_measured)

        if self.data_associated:
            self.X_est.header.stamp = time  # TESTME: Can we remove this?

        if self.debug_enabled:
            self.debug_pub.publish(self.X_est)


if __name__ == '__main__':
    # Initialize the node and name it.
    rospy.init_node('Kalman')

    # Go to class functions that do all the heavy lifting
    # Do error checking
    try:
        kf = KalmanFilterNode()
    except rospy.ROSInterruptException:
        pass
