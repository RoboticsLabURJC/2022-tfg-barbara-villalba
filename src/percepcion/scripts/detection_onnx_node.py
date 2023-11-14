#! /usr/bin/env python3
import torch
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import onnxruntime as ort
import csv
from sklearn.cluster import DBSCAN
from collections import Counter
from numba import jit

IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = "/home/bb6/YOLOP/weights/yolop-320-320.onnx"

diff_left = []
diff_right = []

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/yolop/image',Image,queue_size=1)
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)
        ort.set_default_logger_severity(4)
        self.ort_session = ort.InferenceSession(ROUTE_MODEL,providers=['CUDAExecutionProvider'])
        self.list = []



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
    

    def average(self,val):
        self.list.append(val)
        return sum(self.list)/len(self.list)
    

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
        
        
        
        
        """
                with open('/home/bb6/pepe_ws/src/yolop/medidas/onnx/640-640/inference_time.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([(inference_time)])
        """

        
        
        # Add fps to total fps.
        #--total_fps += fps
        # Increment frame count.
        #--frame_count += 1

        images = []

        
        for image in (da_seg_out,ll_seg_out):

            #--Convert tensor to a numpy and then it realise transpose : (H, W, C) 
            #--image_np = image.numpy()
            image_array = np.transpose(image, (2, 3, 1, 0))

            #--Normalise numpy a values 0-255
            image_norm = cv2.normalize(image_array[:,:,1,:], None, 0,1, cv2.NORM_MINMAX, cv2.CV_8U)

            images.append(image_norm)

        
        

        
        points_lane = np.column_stack(np.where(images[1] > 0))
        points_road = np.column_stack(np.where(images[0] > 0))

        dbscan = DBSCAN(eps=30, min_samples=40,metric="euclidean")
        dbscan.fit(points_lane)
        n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        print("Clusters: " + str(n_clusters_))
        print("Noise points: " + str(list(dbscan.labels_).count(-1)))

        result = np.zeros((images[1].shape[0], images[1].shape[1], 3), dtype=np.uint8)
        right_points = 0 
        left_points = []
        right_x = 0
        left_x = 0
        centroid_of_right_cluster = []
        centroid_of_left_cluster = []

        final_final = [[]]

        for cluster in range(len(set(dbscan.labels_))):

            point_cluster = points_lane[dbscan.labels_==cluster,:]
            if point_cluster.size > 0 and not np.isnan(point_cluster).all():
                if(cluster == 0):
                    
                    result[point_cluster[:,0],point_cluster[:,1]] = [0,0,255]

                elif(cluster == 1):
                    point_cluster = points_lane[dbscan.labels_==cluster,:]
                    result[point_cluster[:,0],point_cluster[:,1]] = [0,255,0]

                elif(cluster == 2):
                    result[point_cluster[:,0],point_cluster[:,1]] = [255,0,0]

                #--result[point_cluster[:,0],point_cluster[:,1]] = [0,0,255]
                
                """
                centroid_of_cluster = np.nanmean(point_cluster, axis=0)
                #--print("Centroid Cluster [" + str(cluster) + "]: " + str(int(centroid_of_cluster[1])) + ", " + str(int(centroid_of_cluster[0]))) 
                if(int(centroid_of_cluster[1]) >= 160):
                    #--print("Centroid Right Cluster [" + str(cluster) + "]: " + str(int(centroid_of_cluster[1])) + ", " + str(int(centroid_of_cluster[0]))) 
                    right_points = point_cluster
                    centroid_of_right_cluster = centroid_of_cluster
                    #--print(centroid_of_right_cluster)
                    result[right_points[:, 0], right_points[:, 1]] = [0,0,255]
                    right_x = right_points[:, 1]
                    pepe_der = points_road[np.logical_and(points_road[:,1] > right_x.min(),points_road[:,1] > right_x.max())]
                    diff_right = np.setdiff1d(points_road[:,1],pepe_der)
                    #print(diff_right)
                    
                    #result[points_road[mask],points_road[diff_right]] = [0,255,0]
                #print(diff_right)
                if (int(centroid_of_cluster[1]) >= 50 and int(centroid_of_cluster[1]) <= 160):
                    left_points = point_cluster
                    centroid_of_left_cluster = centroid_of_cluster
                    result[left_points[:, 0], left_points[:, 1]] = [0,0,255]
                    left_x = left_points[:, 1]
                    sum(points_road,left_x)
                    
                    
                    #pepe_izq = points_road[points_road[:,1] < left_x]
                    #result[pepe_izq[:,0],pepe_izq[:,1]] = [0,255,0]
                    #diff_left = np.setdiff1d(points_road[:,1],pepe_izq)

                
                if(len(diff_left) > 0 and len(diff_right)> 0):

                    filtro = np.concatenate((diff_left, diff_right))
                    #--print(filtro)
                    #-filtro = diff_left + diff_right

                    #print(diff_right)

                    final_final = points_road[np.in1d(points_road[:,1],filtro)]
                """
                

                 
                 
                    #--print(final_left.shape,final_right.shape)
                    #final_final = np.concatenate((final_left,final_right), axis=0)
                    #result[final_final[:,0],final_final[:,1]] = [0,255,0]

                
        #--pepe = points_road[np.logical_and(left_x.min() <= points_road[:,1], np.logical_and(left_x.max() <= points_road[:,1], np.logical_and(points_road[:,1] <= right_x.min(), points_road[:,1] <= right_x.max())))]
        #pepe = points_road[np.logical_and(points_road[:,1] > 50,points_road[:,1] < 160)]

            #points_in_between = points_road[np.logical_and(points_road[:,1] > left_points.min(),points_road[:,1] < left_points.max())]
            #print(points_in_between[:,1])
        #--result[pepe[:,0],pepe[:,1]] = [0,255,0]
        
        
            #--print(left_x.min(),left_x.max())
            #--print(right_x.min(),right_x.max())
          
            #--print(left_x)
            #print("Road")
            #print(points_road[:,1].min(),points_road[:,1].max())

            # Crea máscaras booleanas para cada condición
            #mask1 = (points_road[:,1] >= left_x.min()) & (points_road[:,1] < left_x.max())
            #--mask2 = (points_road[:,1] >= right_x.min()) & (points_road[:,1] < right_x.max())

            # Combina las máscaras y selecciona los elementos
            
        """
         

            
            print(points_road[:,1])



            #points_in_between = points_road[(points_road[:,1] > left_x.min()) & (left_x.max() > points_road[:,1])]
            points_in_between = points_road[np.logical_and(left_x.min() <= points_road[:,1], np.logical_and(left_x.max() <= points_road[:,1], np.logical_and(points_road[:,1] <= right_x.min(), points_road[:,1] <= right_x.max())))]
        """
       
        color_image =  np.zeros((images[0].shape[0], images[0].shape[1], 3), dtype=np.uint8)
        color_image[images[0] == 1] = [0, 255, 0]
        color_image[images[1] == 1] = [0,0,255]


        superimposed_image = cv2.addWeighted(cv2.resize(cvimage,(320,320),cv2.INTER_LINEAR), 0.6, color_image, 0.4, 0)
 
        t2 = time.time()
        #print(1/(t2-t1))
        return superimposed_image,result
        #return cv2.resize(superimposed_image,(640,640),cv2.INTER_LINEAR) 320-320
    


   
    def callback(self, data):

        #--t1 = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 

    
        image_result,img = self.infer_yolop(cv_image)

        cv2.imshow('Image', image_result)
        cv2.imshow('Image-cluster',img)
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