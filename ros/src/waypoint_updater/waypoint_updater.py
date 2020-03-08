#!/usr/bin/env python

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

MAX_DECEL = 1

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

#TODO: Decide the number of WPs
LOOKAHEAD_WPS = 30 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')


        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        self.loop()
        #rospy.spin()

    def loop(self):
        #TODO: Decide the Rate
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if not None in [self.pose, self.waypoint_tree]:
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y 

        #finde the closest waypoint's idx
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]
        
        # check if closest is ahead or behind car
        closest_waypoint = self.waypoints_2d[closest_idx]
        pre_closest_waypoint = self.waypoints_2d[closest_idx-1]
        
        cl_vect = np.array(closest_waypoint)
        pre_vect = np.array(pre_closest_waypoint)
        pos_vect = np.array([x,y])

        if np.dot(cl_vect - pre_vect, pos_vect - cl_vect) > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        lane.waypoints = self.base_waypoints.waypoints[closest_idx : farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            self.final_waypoints_pub.publish(lane)
            if self.stopline_wp_idx == -1:
                rospy.loginfo('No Stopline found , publish usual lane')
            else:
                rospy.loginfo('Stopline is beyond farthest traj pt')
        else:
            lane.waypoints = self.decelerate_waypoints(lane.waypoints, closest_idx)
            self.final_waypoints_pub.publish(lane)	
            rospy.loginfo('Decleration initiated')

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    
    def decelerate_waypoints(self, waypoints, closest_idx):
        tem = []
        for i , wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            stop_idx = max(self.stopline_wp_idx - closest_idx - 3, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel < 1.0:
                vel = 0.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            tem.append(p)
        return tem

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
