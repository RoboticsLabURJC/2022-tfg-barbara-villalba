#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.point_cloud2 import PointCloud2
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
IMAGE_TOPIC = '/airsim_node/drone/front_center_custom/Scene'

MAX_SLOPE = 0.5
APERTURE_SIZE = 3
T_LOWER = 50  # Lower Threshold
T_UPPER =  150 # Upper threshold

bottom_left = []
top_left = []
bottom_right = []
top_right = []


class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)
        

    def region_selection(self,image):
        global bottom_left,top_left,bottom_right,top_right
        mask = np.zeros_like(image)   
       
        rows, cols = image.shape[:2]
        
        """
        bottom_left  = [cols * 0.1, rows * 0.95]
        top_left     = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right    = [cols * 0.6, rows * 0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)        
        """
        bottom_left = [0, rows]
        medium = [cols/2 , rows/2]
        bottom_right = [cols,rows]
        
        vertices = np.array([[bottom_left,medium,bottom_right]], dtype=np.int32)
        
        
        cv2.fillPoly(mask, vertices,(255,255,255))
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image
    
    def hough_lines(self,img):

        rho = 1              #Distance resolution of the accumulator in pixels.
        theta = np.pi/180    #Angle resolution of the accumulator in radians.
        threshold = 70       #Only lines that are greater than threshold will be returned.
        minLineLength = 20   #Line segments shorter than that are rejected.
        maxLineGap = 10     #Maximum allowed gap between points on the same line to link them
        return cv2.HoughLinesP(img, rho, theta, threshold, None, minLineLength, maxLineGap)

    def calculate_lane(self,lines,image):
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) / (x2 - x1) #-- Calculing the slope of line
                    if math.fabs(slope) < MAX_SLOPE: #-- Extreme slope with an umbral in 0.5 value (arbitrary), are discards
                        continue
                    if slope <= 0: #-- If the slope is negative, left group.
                        left_line_x.extend([x1, x2])
                        left_line_y.extend([y1, y2])
                    else: #-- If the slope is positive, left group..
                        right_line_x.extend([x1, x2])
                        right_line_y.extend([y1, y2])

            min_y = int(image.shape[0] * 0.625)  # <-- Just below the horizon, depend on the shape image. 250 value
            max_y = image.shape[0] # <-- Adjust in image

            #-- Only calculate polynomian equation if detected left_line and right_line

            if len(left_line_y) > 0 and len(left_line_x) > 0:
                poly_left = np.poly1d(np.polyfit(left_line_y,left_line_x,deg=1))
            
                left_x_start = int(poly_left(max_y))
                left_x_end = int(poly_left(min_y))
            else:
                pass
                
            if len(right_line_y) > 0 and len(right_line_x) > 0:
                poly_right = np.poly1d(np.polyfit(right_line_y,right_line_x,deg=1))

                right_x_start = int(poly_right(max_y))
                right_x_end = int(poly_right(min_y))

            else:
                pass

            if (len(left_line_y) > 0 and len(left_line_x) > 0) and (len(right_line_y) > 0 and len(right_line_x) > 0):
                return [[left_x_start, max_y, left_x_end, min_y],[right_x_start, max_y, right_x_end, min_y]]

        else:
            return None
        
        
    def draw_lane_lines(self,image,lines, color, thickness):

        line_img = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
       
        image = np.copy(image)
        if lines is None:
            return image
        for line in lines:
            x1, y1, x2, y2 =  line
            M = (int(((x1+x2))/2),int (((y1+y2))/2))
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
            #--print(M)
            #--self.draw_control_lines(line_img)
        img = cv2.addWeighted(image, 0.8, line_img, 1.0, 0.0)
        return img
    
    def draw_control_lines(self,img_lanes):
        
        cv2.circle(img_lanes, (200, 290), 2, (0, 255, 255), -1)
        cv2.circle(img_lanes, (200, 310), 2, (0, 255, 255), -1)
        cv2.circle(img_lanes, (200, 330), 2, (0, 255, 255), -1)

    def callback(self, data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        resized_image = cv2.resize(cv_image, (600, 600))
        
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
        
        gausBlur = cv2.GaussianBlur(gray_image, (3,3),0)
        
        canny_image = cv2.Canny(gausBlur, T_LOWER, T_UPPER,APERTURE_SIZE)
        img = self.region_selection(canny_image)

        

        lines = self.hough_lines(img)
        
        lanes_lines = self.calculate_lane(lines,img)
       
        img_lanes = self.draw_lane_lines(resized_image,lanes_lines,(255,0,0),3)


       
        cv2.imshow("img_lanes",img_lanes)
        cv2.imshow("Canny",canny_image)
        
        cv2.imshow("region_selection",img)

        
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
    rospy.init_node("detection_lines_py")
    image_viewer = ImageViewer()
    image_viewer.run()

