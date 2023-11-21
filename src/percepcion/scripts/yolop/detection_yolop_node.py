#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from torchvision import transforms
import torch
import time

IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = 'hustvl/yolop'
MODEL = 'yolop'


class ImageSubscriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bridge = CvBridge()
        self.transform = transforms.Compose([ 
                    transforms.ToTensor() 
                    ]) 
        
        self.model= torch.hub.load(ROUTE_MODEL, MODEL, pretrained=True)
        self.model = self.model.to(self.device)
        self.image_pub = rospy.Publisher('/yolop/image',Image,queue_size=1)
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)

    def callback(self, data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        resized = cv2.resize(cv_image, (320,320), interpolation = cv2.INTER_AREA)
        
        imagen_tensor = self.transform(resized).to(self.device).unsqueeze(0)

        _, da_seg_out, ll_seg_out = self.model(imagen_tensor)

        images = []

        for image in (da_seg_out,ll_seg_out):

            #--Convert tensor to a numpy and then it realise transpose : (H, W, C) 
            image_np = image.detach().cpu().numpy()
            image_array = np.transpose(image_np, (2, 3, 1, 0))

            #--Normalise numpy a values 0-255
            image_norm = cv2.normalize(image_array[:,:,1,:], None, 0,255, cv2.NORM_MINMAX, cv2.CV_8U)

            images.append(image_norm)

        colored_dag_seg_image =  np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.uint8)
        colored_dag_seg_image[images[0] == 255] = [0, 255, 0]

        colored__lane_image = np.zeros((images[1].shape[0], images[1].shape[1], 3), dtype=np.uint8)
        colored__lane_image[images[1] == 255] = [0, 0, 255]

        superimposed_image = cv2.addWeighted(resized, 0.6, colored__lane_image, 0.4, 0)
        image_result = cv2.addWeighted(superimposed_image, 0.6, colored_dag_seg_image, 0.4, 0)
        
        image_message = self.bridge.cv2_to_imgmsg(image_result, encoding="bgr8")
        
        self.image_pub.publish(image_message)

class ImageViewer:
    def __init__(self):
        self.subscriber = ImageSubscriber()

    def start(self):
        try:
          rospy.spin()
        except rospy.ROSInterruptException:
              pass
            

if __name__ == '__main__':
    rospy.init_node("det_node_py")
    image_viewer = ImageViewer()
    image_viewer.start()