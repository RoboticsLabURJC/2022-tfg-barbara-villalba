#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

IMAGE_TOPIC = '/airsim_node/drone/front_center_custom/Scene'

point1 = (0, 400)
#--point2 = (100,200)
#--point3 = (300,200)
point2 = (0,300)
point3 = (200,200)
point4 = (400,320)
point5 = (400,400)

lower_white = np.array([0,22,0])
upper_white = np.array([103,39,255])

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)


    def interest_region(self,cv_image):
        mask = np.zeros_like(cv_image)
        points = np.array([[point1,point2,point3,point4,point5]], dtype=np.int32)

        cv2.fillPoly(mask,points,(255,255,255))
        masked_image = cv2.bitwise_and(cv_image, mask)


        return masked_image
    
    def detection_lines(self,img,rgb_image):
        linesP = cv2.HoughLinesP(img, 1, np.pi / 180, 70, None, 20, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                if(((l[0] >= 0 and l[2] <= 180) and (l[3] >=200 and l[1] <= 320)) 
                   or ((l[2] >= 180 and l[0] <= 400 ) and (l[3] >=200 and l[1] <= 320)) ):
                    print("Start: " + str(l[0]) + "," + str(l[1]) + "; End: " + str(l[2]) + "," + str(l[3]))
                    cv2.line(rgb_image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

        return rgb_image


        
    def callback(self, data):

        """
    position:
      x: 37.3194618225
      y: -11.8251657486
      z: -1.38555669785
        """

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 

       
        canny_image = cv2.Canny(cv_image, 100, 150,3)

        img = self.interest_region(canny_image)

        cv2.imshow("canny",img)
        cv2.imshow("lines",self.detection_lines(img,cv_image))
        cv2.waitKey(1)


    """
        grayscale = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        kernel_size = 3
        blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

        low_t = 50
        high_t = 150
        edges = cv2.Canny(blur, low_t, high_t)

        vertices = np.array([[point1, point2,point3,point4]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, vertices, (255,255,255))
        masked_edges = cv2.bitwise_and(edges, mask)

        linesP = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 70, None, 20, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                cv2.line(cv_image, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

        #--image = cv2.cvtColor(cv_image,cv2.COLOR_BGR2HSV_FULL)

        
        cv2.imshow("Canny",edges)
        cv2.imshow("Trapecio",masked_edges)
        
    """
        

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
    rospy.init_node("detection_lines_py")
    image_viewer = ImageViewer()
    image_viewer.run()
