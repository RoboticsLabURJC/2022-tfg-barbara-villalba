#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from torchvision import transforms
import torch
import time
import csv

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
        self.counter_time = 0.0
        self.list = []

        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)


    def average(self,val):
        self.list.append(val)
        return sum(self.list)/len(self.list)



    def callback(self, data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        
        imagen_tensor = self.transform(cv_image).to(self.device).unsqueeze(0)
      
        t1 = time.time()
        _, da_seg_out, ll_seg_out = self.model(imagen_tensor)
        t2 = time.time()

        inference_time = t2 - t1
        #--print(self.average(1/(t2-t1)))

        with open('/home/bb6/pepe_ws/src/yolop/medidas/pytorch/inference_time.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([inference_time])
    
        
        images = []

        
        for image in (da_seg_out,ll_seg_out):

            #--Convert tensor to a numpy and then it realise transpose : (H, W, C) 
            image_np = image.detach().cpu().numpy()
            image_array = np.transpose(image_np, (2, 3, 1, 0))

            #--Normalise numpy a values 0-255
            image_norm = cv2.normalize(image_array[:,:,1,:], None, 0,1, cv2.NORM_MINMAX, cv2.CV_8U)

            images.append(image_norm)

        color_image =  np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.uint8)
        color_image[images[0] == 1] = [0, 255, 0]
        color_image[images[1] == 1] = [0, 0, 255]

        superimposed_image = cv2.addWeighted(cv_image, 0.6, color_image, 0.4, 0)
      

        cv2.imshow('Image', superimposed_image)
        cv2.waitKey(1)
        
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