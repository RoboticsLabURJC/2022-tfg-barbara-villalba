#! /usr/bin/env python3
import torch
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import onnxruntime as ort
from sklearn.cluster import DBSCAN
from numba import jit
import matplotlib.pyplot as plt

IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = "/home/bb6/YOLOP/weights/yolop-320-320.onnx"

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/yolop/image',Image,queue_size=1)
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)
        ort.set_default_logger_severity(4)
        self.ort_session = ort.InferenceSession(ROUTE_MODEL,providers=['CUDAExecutionProvider'])
        self.list = []
        self.bottom_left_base = [0,320]
        self.bottom_right_base = [320,320]
        self.bottom_left  = [0, 280]
        self.bottom_right = [320,280]
        self.top_left     = [155,140]
        self.top_right    = [165, 140]
        self.point_cluster = np.ndarray
        
        
        self.vertices = np.array([[self.bottom_left_base,self.bottom_left, self.top_left, self.top_right, self.bottom_right,self.bottom_right_base]], dtype=np.int32)

    def resize_unscale(self,img, new_shape=(640, 640), color=114):
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        canvas = np.zeros((new_shape[0], new_shape[1], 3))
        canvas.fill(color)
        # Scale ratio (new / old) new_shape(h,w)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w,h
        new_unpad_w = new_unpad[0]
        new_unpad_h = new_unpad[1]
        pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  # wh padding

        dw = pad_w // 2  # divide padding into 2 sides
        dh = pad_h // 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

        canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img

        return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)


    def clustering(self,img):
        points_lane = np.column_stack(np.where(img > 0))
        dbscan = DBSCAN(eps=25, min_samples=1,metric="euclidean")
        dbscan.fit(points_lane)
        n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        print("Clusters: " + str(n_clusters_))
        #print("Noise points: " + str(list(dbscan.labels_).count(-1)))
        self.point_cluster = []

        result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        for cluster in range(len(set(dbscan.labels_))):
            
            self.point_cluster.append(points_lane[dbscan.labels_==cluster,:]) 

        # Concatenar todos los clusters en un solo array
        all_points = np.concatenate(self.point_cluster, axis=0)

        result[all_points[:,0], all_points[:,1]] = 255

        return result
    
    def draw_region(self,img):

        mask = np.zeros_like(img) 

        cv2.fillPoly(mask, self.vertices,(255,255,255))
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image

    def infer_yolop(self,cvimage):
        global diff_right,diff_left
        canvas, r, dw, dh, new_unpad_w, new_unpad_h = self.resize_unscale(cvimage, (320,320))

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,640,640)

        t1 = time.time()
        # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
        _, da_seg_out, ll_seg_out = self.ort_session.run(
            ['det_out', 'drive_area_seg', 'lane_line_seg'],
            input_feed={"images": img}
        )

        images = []

        
        for image in (da_seg_out,ll_seg_out):

            #--Convert tensor to a numpy and then it realise transpose : (H, W, C) 
            #--image_np = image.numpy()
            image_array = np.transpose(image, (2, 3, 1, 0))

            #--Normalise numpy a values 0-255
            image_norm = cv2.normalize(image_array[:,:,1,:], None, 0,1, cv2.NORM_MINMAX, cv2.CV_8U)

            images.append(image_norm)


        #--kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(33,33))
        #--closing = cv2.morphologyEx(images[1], cv2.MORPH_CLOSE, kernel)

        cvimage = cv2.resize(cvimage,(320,320),cv2.INTER_LINEAR)
        cvimage[images[1] == 1] = [0, 0, 255]


        mask = self.draw_region(images[1])

        masked_image = self.draw_region(cvimage)



        result = self.clustering(mask)
    

        t2 = time.time()

        iterations = str (int(1/(t2-t1)))

        cv2.putText(
                cvimage, 
                text = "FPS: " + iterations,
                org=(0, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
        return cvimage,masked_image,result
        #return cv2.resize(superimposed_image,(640,640),cv2.INTER_LINEAR) 320-320

    def callback(self, data):

        #--t1 = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        image_result,mask,img_cluster = self.infer_yolop(cv_image)

        cv2.imshow('Image',image_result)
        cv2.imshow('Mask',mask)
        cv2.imshow('img_cluster',img_cluster)
        #cv2.imshow('Image', histogram)
        #--t3 = time.time()

        #fps2 = 1 / (t3- t1)
        #print(f"FINAL FPS: {fps2:.3f}")
        
        # Press `q` to exit.
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