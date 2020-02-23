from styx_msgs.msg import TrafficLight
import cv2
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self,is_site):
        #TODO load classifier
        if is_site == False:
            SSD_GRAPH_FILE = r'light_classification/frozen_inference_graph_sim.pb'
        else:
            SSD_GRAPH_FILE = r'light_classification/frozen_inference_graph_site.pb'
        self.det_graph = self.load_graph(SSD_GRAPH_FILE)
        self.image_tensor = self.det_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.det_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.det_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.det_graph.get_tensor_by_name('detection_classes:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        with tf.Session(graph=self.det_graph) as sess: 
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                        feed_dict={self.image_tensor: image_np})
            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)
            max_score_idx = np.argmax(scores)
            if scores[max_score_idx] == 0.0:
                rospy.loginfo('No Light Detected')
                return TrafficLight.UNKNOWN
            elif scores[max_score_idx] < 0.5:
                rospy.loginfo('Low Conf Light Detected')
                return TrafficLight.LOWCONF
            elif classes[max_score_idx] == 1.0:
                rospy.loginfo('Green Light Detected')
                return TrafficLight.GREEN
            elif classes[max_score_idx] == 2.0:
                rospy.loginfo('Red Light Detected')
                return TrafficLight.RED
            elif classes[max_score_idx] == 3.0:
                rospy.loginfo('Yellow Light Detected')
                return TrafficLight.YELLOW
            else:
                rospy.loginfo('Off Light Detected')
                return TrafficLight.UNKNOWN
    
    def load_graph(self,graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph
