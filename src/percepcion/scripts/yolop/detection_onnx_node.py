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
from scipy.ndimage import binary_dilation
from yolop.msg import MassCentre
from scipy.interpolate import interp1d


IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = "/home/bb6/YOLOP/weights/yolop-320-320.onnx"

MIN_VALUE_Y = 150
MAX_VALUE_Y = 320


class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)
        ort.set_default_logger_severity(4)
        self.ort_session = ort.InferenceSession(ROUTE_MODEL,providers=['CUDAExecutionProvider'])
        self.list = []
        self.bottom_left_base = [0,320]
        self.bottom_right_base = [320,320]
        self.bottom_left  = [0, 320]
        self.bottom_right = [320,320]
        self.top_left     = [0,180]
        self.top_right    = [320, 180]
        self.vertices = np.array([[self.bottom_left,self.top_left,self.top_right,self.bottom_right]], dtype=np.int32)
        self.point_cluster = np.ndarray
        self.kernel = np.array([[0,1,0], 
                                [1,1,1], 
                                [0, 1,0]]) 
        
       
        
        self.msg = MassCentre()
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.img_result = np.zeros((320, 320, 3), dtype=np.uint8)
    
        self.mass_centre_pub = rospy.Publisher('/yolop/detection_lane/mass_centre_lane',MassCentre,queue_size=10)
    #--Resize image for more later yolop
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
    
    def calculate_lineal_regression(self,points_cluster):

        valuesX = np.arange(170,320) 
        coefficients = np.polyfit(points_cluster[:,0],points_cluster[:,1],1)
        values_fy = np.polyval(coefficients,valuesX).astype(int)
        fitLine_filtered = [(x, y) for x, y in zip(valuesX, values_fy) if 0 <= y <= 319]
        line = np.array(fitLine_filtered)

        return line


        """
        #--PARA TODOS LOS CLUSTERS
        dict_lines = {label: [] for label in set(dict_clusters.keys())}
        print(dict_lines)
        valuesX = np.arange(150,320) 
        #print(valuesX)
        for id_cluster,points_cluster in dict_clusters.items():
            points = points_cluster["points_cluster"]
            
            coefficients = np.polyfit(points[:,0],points[:,1],1)
            values_fy = np.polyval(coefficients,valuesX).astype(int)

            fit = [y if y <= 319 else 319 for y in values_fy]
            fitLine_filtered = [(x, y) for x, y in zip(valuesX, fit) if 0 <= y <= 319]

            line = np.array(fitLine_filtered)
            dict_lines[id_cluster] = {"points_line" : line}
            return dict_lines
        """


       
        
    def clustering(self,img):
        #--Convert image in points
        points_lane = np.column_stack(np.where(img > 0))
        dbscan = DBSCAN(eps=25, min_samples=1,metric="euclidean")

        if points_lane.size > 0:
            dbscan.fit(points_lane)
            n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
            print("Clusters: " + str(n_clusters_))
            dict_clusters = {label: [] for label in set(dbscan.labels_)}
            
            for cluster in range(len(set(dbscan.labels_))):
                points_cluster = points_lane[dbscan.labels_==cluster,:]
                centroid_cluster = points_cluster.mean(axis=0).astype(int)
                dict_clusters[cluster] = {"points_cluster": points_cluster, "centroid": centroid_cluster}


            return dict_clusters
        
        else:
            return None
    
    def draw_region(self,img):

        mask = np.zeros_like(img) 

        cv2.fillPoly(mask, self.vertices,(255,255,255))
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image
    
    def calculate_mass_centre_lane(self,img):

        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contornos, -1, 255, thickness=cv2.FILLED)

        momentos = cv2.moments(img)

        # Calcular el centro de masa
        cx = int(momentos['m10'] / momentos['m00'])
        cy = int(momentos['m01'] / momentos['m00'])

        return cx,cy
    
    def dilate_lines(self,left_lane_points,right_lane_points):
        img_left_lane = np.zeros((320, 320), dtype=np.uint8)
        img_right_lanes = np.zeros((320, 320), dtype=np.uint8)

        img_left_lane[left_lane_points[:,0],left_lane_points[:,1]] = 255
        img_right_lanes[right_lane_points[:,0],right_lane_points[:,1]] = 255

        
        img_left_lane_dilatada = binary_dilation(img_left_lane, structure=self.kernel)
        img_left_lane_dilatada = (img_left_lane_dilatada).astype(np.uint8)

        img_right_lane_dilatada = binary_dilation(img_right_lanes, structure=self.kernel)
        img_right_lane_dilatada = (img_right_lane_dilatada).astype(np.uint8)


        return img_left_lane_dilatada,img_right_lane_dilatada
    
    def interpolate_lines(self,img_det,points_line_left,points_line_right):

        points_road = np.column_stack(np.where(img_det[0] > 0))

        if(points_road.size > 0):
            f1 = interp1d(points_line_left[:, 0], points_line_left[:, 1],fill_value="extrapolate")
            f2 = interp1d(points_line_right[:, 0], points_line_right[:, 1],fill_value="extrapolate") 
            y_values_f1 = f1(points_road[:, 0])
            y_values_f2 = f2(points_road[:, 0])
            indices = np.where((y_values_f1 <= points_road[:, 1]) & (points_road[:, 1] <= y_values_f2))
            
            
            points_beetween_lines = points_road[indices]

            return points_beetween_lines

    def infer_yolop(self,cvimage):
        
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

        
        mask = self.draw_region(images[1])

        cvimage = cv2.resize(cvimage,(320,320),cv2.INTER_LINEAR)
        cv2.line(cvimage,(160,0),(160,320),(0,0,0))
        masked_image = self.draw_region(cvimage)

        #print(images)

        dict_clusters = self.clustering(mask)

        margin_right_points = []
        margin_left_points = []

        for id_cluster,values in dict_clusters.items():
            points_cluster = values['points_cluster']
            centroid = values['centroid']
            color = self.colors[id_cluster % len(self.colors)]
            if(points_cluster.size > 0):
                print(id_cluster,centroid[1])
                if (centroid[1] >= 70 and centroid[1] <= 160):
                  
                    margin_left_points.append(points_cluster)
                    cvimage[points_cluster[:,0], points_cluster[:,1]] = color
                    
                  
                elif (centroid[1] > 160):
                    margin_right_points.append(points_cluster)
                
                    cvimage[points_cluster[:,0], points_cluster[:,1]] = color

       
        if margin_right_points and margin_left_points:
            points_line_right =  self.calculate_lineal_regression(np.concatenate(margin_right_points,axis=0))
            points_line_left = self.calculate_lineal_regression(np.concatenate(margin_left_points,axis=0))

        
            img_line_left,img_line_right = self.dilate_lines(points_line_left,points_line_right)


            cvimage[img_line_left == 1] = [255,255,255]
            cvimage[img_line_right == 1] = [255,255,255]

            points_road = np.column_stack(np.where(images[0] > 0))

            f1 = interp1d(points_line_left[:, 0], points_line_left[:, 1],fill_value="extrapolate")
            f2 = interp1d(points_line_right[:, 0], points_line_right[:, 1],fill_value="extrapolate") 

            y_values_f1 = f1(points_road[:, 0])
            y_values_f2 = f2(points_road[:, 0])

            indices = np.where((y_values_f1 <= points_road[:, 1]) & (points_road[:, 1] <= y_values_f2))
            
           
            points_beetween_lines = points_road[indices]
            #centro_de_masas = np.mean(points_beetween_lines, axis=0).astype(int)
            
            
            cvimage[points_beetween_lines[:,0],points_beetween_lines[:,1]] = [255,0,0]
            #cv2.circle(cvimage, (centro_de_masas[1],centro_de_masas[0]), radius=10, color=(0, 0, 0),thickness=-1)
        
       

        t2 = time.time()

        fps = str (int(1/(t2-t1)))

        cv2.putText(
                cvimage, 
                text = "FPS: " + fps,
                org=(0, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
        
        return cvimage,masked_image
        

    def callback(self, data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
     
        org_img,masked_image= self.infer_yolop(cv_image)

        cv2.imshow('Image-Original-Network',org_img)
        cv2.imshow('Masked-Image',masked_image)
        #cv2.imshow('img_cluster',img_cluster)
        
    
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