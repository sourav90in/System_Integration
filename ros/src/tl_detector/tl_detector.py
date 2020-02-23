#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import numpy as np

STATE_COUNT_THRESHOLD = 2
STOP_LINE_CURRENT_POSE_DIST_CUTOFF = 160

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.process_ctr = True

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.config["is_site"])
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg
 
    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        #additional logic to skip every alternate images
        if self.process_ctr < 2:
            self.process_ctr += 1
            return
        else:
            self.process_ctr = 0
       
        
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            if state == TrafficLight.RED or state == TrafficLight.UNKNOWN:
                #When a red light is known or a unclassified light is obtained
                light_wp = light_wp
            else:
                #Green light case
                light_wp = -1                
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose_state_x,pose_state_y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose_state_x (Pose): position to match a waypoint to in X
            pose_state_y: position to match a waypoint to in Y

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        x = pose_state_x
        y = pose_state_y 

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

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_wp = None
        min_dist = None
        car_position_wp = None
        stop_line_pos_for_light = None
        nearest_wp_to_stopline = None
        wp_delta_stopline_car_position = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position_wp = self.get_closest_waypoint(self.pose.pose.position.x,self.pose.pose.position.y)
            #find the closest visible traffic light (if one exists)
            min_dist = len(self.waypoints.waypoints)
            for i in range(len(self.lights)):
                stop_line_pos_for_light = stop_line_positions[i]
                nearest_wp_to_stopline = self.get_closest_waypoint(stop_line_pos_for_light[0],stop_line_pos_for_light[1])
                wp_delta_stopline_car_position = nearest_wp_to_stopline - car_position_wp
                if wp_delta_stopline_car_position >= 0 and wp_delta_stopline_car_position < min_dist:
                    min_dist = wp_delta_stopline_car_position
                    light = self.lights[i]
                    light_wp = nearest_wp_to_stopline              
        #Skip inference if traffic light is Cutoff distance away from current pose
        if light and (min_dist <= STOP_LINE_CURRENT_POSE_DIST_CUTOFF):
            state = self.get_light_state(light)
            return light_wp, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
