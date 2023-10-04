import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import supervision as sv
import torchvision
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time
IMAGE_TOPIC = '/airsim_node/drone/front_center_custom/Scene'

class Perception:
    def __init__(self):
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.model_type = "default"
        self.device = "cuda"

        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)

        self.mask_generator = SamAutomaticMaskGenerator(sam)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)
        
        self.list = []
        
        
        
        #--Python time
        #--
        
        
    def SAM(self,image):
        
        #--Marca de tiempo arriba y abajo, y restando
        #--Marca de tiempo arriba
        
        currentTime = time.time()
        sam_result = self.mask_generator.generate(image)
        lastTime = time.time()
        
        time_mask_generator = int(lastTime - currentTime)
        
        #--Resta de los tiempos
        
        print("Mascaras: " + str(len(sam_result)) + ", " + "Tiempo: " + str(time_mask_generator))

        """
        mask_annotator = sv.MaskAnnotator()
        detections = sv.Detections.from_sam(sam_result)
        annotated_image = mask_annotator.annotate(image, detections)
        
        return annotated_image
        """
    
    def callback(self, data):

     cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
     image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) 
     
     #--img = self.region_selection(cv_image)
     
     self.SAM(image)
        
     #--cv2.imshow("Image",img_sam)

     cv2.waitKey(1)


class ImageViewer:
    def __init__(self):
        self.subscriber = Perception()

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





