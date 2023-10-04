#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

IMAGE_TOPIC = '/airsim_node/drone/front_center_custom/Scene'

lower_white = np.array([0,0,0])
upper_white = np.array([97,136,255])


point1 = (0, 400)
point2 = (0,300)
point3 = (155,250)
point4 = (300,250)
point5 = (400,300)
point6 = (400, 400)

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)
        
    def filter_hsv(self,image):
       
        image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(image_hsv, lower_white, upper_white)

        vertices = np.array([[point1, point2,point3,point4,point5,point6]], dtype=np.int32)
       
        new_mask = np.zeros_like(image_hsv)

        cv2.fillPoly(new_mask, vertices,(255,255,255))

        cut_image = cv2.bitwise_and(new_mask, new_mask,mask=mask_white)
        return cut_image
    
   
    
    def callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

        image_hsv= self.filter_hsv(cv_image)

        cv2.imshow("HSV",image_hsv)
        #--cv2.imshow("Canny",edges)
        cv2.waitKey(1)

class ImageViewer:
    def __init__(self):
        self.subscriber = ImageSubscriber()

    def run(self):
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("Aplicación detenida")
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node("filter_hsv_node_py")
    image_viewer = ImageViewer()
    image_viewer.run()
