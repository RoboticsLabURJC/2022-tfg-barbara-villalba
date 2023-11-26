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

IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = "/home/bb6/YOLOP/weights/yolop-320-320.onnx"

MIN_VALUE_Y = 150
MAX_VALUE_Y = 320

@jit(nopython=True)
def calculate_values_xy(left_fit,right_fit):
        
    left_lane = []
    right_lane = []

    # Agregar valores en cada iteración de dos bucles for
    for y in range(MIN_VALUE_Y,MAX_VALUE_Y):
    
        left_fitx= left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]

        right_fitx = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]

 
        if(left_fitx >= 0 and left_fitx < 320):
           left_lane.append([y,int(left_fitx)])
           
        if(left_fitx > 319):
           left_lane.append([y,319])

        if(right_fitx >= 0 and right_fitx < 320):
            right_lane.append([y,int(right_fitx)])
            

        if(right_fitx > 319):
            right_lane.append([y,319])

    return left_lane,right_lane




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
        self.bottom_left  = [0, 270]
        self.bottom_right = [320,270]
        self.top_left     = [155,150]
        self.top_right    = [165, 150]
        self.vertices = np.array([[self.bottom_left_base,self.bottom_left, self.top_left, self.top_right, self.bottom_right,self.bottom_right_base]], dtype=np.int32)
        self.point_cluster = np.ndarray
        self.kernel = np.array([[0,1,0], 
                                [1,1,1], 
                                [0, 1,0]]) 

        

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
    
    def get_lane_line_indices(self,img):

        # Encuentra las coordenadas x e y de todos los distintos de cero
        # (es decir, blancos) píxeles en el marco.
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
            
        # Almacene los índices de píxeles para las líneas de los carriles izquierdo y derecho
        left_lane_inds = []
        right_lane_inds = []
            
       # Identificar los píxeles distintos de cero (es decir, blancos) en x e y dentro de la ventana
        good_left_inds = ((nonzeroy >= 0) & (nonzeroy <= 319) & (nonzerox >= 0) & (nonzerox < 160)).nonzero()[0]
        #print(((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero())
        #print(((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0])
        
        good_right_inds = ((nonzeroy >= 0) & (nonzeroy <= 319) & (nonzerox > 160) & (nonzerox <= 319)).nonzero()[0]

        #print(good_left_inds,good_right_inds)
                                                            
        # Agregar estos índices a las listas.
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
            
       
        # Concatenar los arrays de índices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    
        # Extrae las coordenadas de píxeles para las líneas de los carriles izquierdo y derecho
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds] 
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty,leftx, 2)
        right_fit = np.polyfit(righty,rightx,2) 

        """
        valuesLineX = np.arange(150,320)

        fitLineLeftY = np.polyval(left_fit, valuesLineX).astype(int)
        fitLineRightY = np.polyval(right_fit, valuesLineX).astype(int)
        
        # Convertir los valores de fitLineY que sean mayores de 320 a 320
        fitLineLeftY_capped = [y if y <= 319 else 319 for y in fitLineLeftY]

        # Convertir los valores de fitLineY que sean mayores de 320 a 320
        fitLineRightY_capped = [y if y <= 319 else 319 for y in fitLineRightY]

        # Filtrar los valores de fitLineY_capped que sean mayores de 150 y menores de 320
        fitLineLeftY_filtered = [(x, y) for x, y in zip(valuesLineX, fitLineLeftY_capped) if 0 <= y <= 319]

        # Filtrar los valores de fitLineY_capped que sean mayores de 150 y menores de 320
        fitLineRightY_filtered = [(x, y) for x, y in zip(valuesLineX, fitLineRightY_capped) if 0 <= y <= 319]

        #print(len(fitLineLeftY_filtered))
        #print( np.array(fitLineRightY_filtered))

        """
        

        left_lane,right_lane = calculate_values_xy(left_fit,right_fit)

        left_lane_points = np.array(left_lane)
        right_lane_points = np.array(right_lane)
    
        return left_lane_points,right_lane_points
    

    def clustering(self,img):
        #--Convert image in points
        points_lane = np.column_stack(np.where(img > 0))

        dbscan = DBSCAN(eps=25, min_samples=1,metric="euclidean")
        dbscan.fit(points_lane)
        n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        print("Clusters: " + str(n_clusters_))
        self.point_cluster = []

        result = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        #--Save all clusters in list
        for cluster in range(len(set(dbscan.labels_))):
            
            self.point_cluster.append(points_lane[dbscan.labels_==cluster,:]) 

        #--Concatenate all clusters in only numpy.ndarray
        all_points = np.concatenate(self.point_cluster, axis=0)

        result[all_points[:,0], all_points[:,1]] = 255

        return result,all_points
    
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

        
        
        cvimage = cv2.resize(cvimage,(320,320),cv2.INTER_LINEAR)
        org_img = cvimage.copy()
        #cvimage[images[1] == 1] = [0, 0, 255]


        mask = self.draw_region(images[1])

        masked_image = self.draw_region(cvimage)

        img_clustering,points_lanes = self.clustering(mask)
        cvimage[img_clustering == 255] = [0, 0, 255]


        left_lane_points,right_lane_points= self.get_lane_line_indices(img_clustering)

        img_left_lane = np.zeros((320, 320), dtype=np.uint8)
        img_right_lanes = np.zeros((320, 320), dtype=np.uint8)

        mask_ = np.zeros((320, 320), dtype=np.uint8)
        
        img_left_lane[left_lane_points[:,0],left_lane_points[:,1]] = 255
        img_right_lanes[right_lane_points[:,0],right_lane_points[:,1]] = 255

        
        img_left_lane_dilatada = binary_dilation(img_left_lane, structure=self.kernel)
        img_left_lane_dilatada = (img_left_lane_dilatada).astype(np.uint8)

        img_right_lane_dilatada = binary_dilation(img_right_lanes, structure=self.kernel)
        img_right_lane_dilatada = (img_right_lane_dilatada).astype(np.uint8)

        cvimage[img_left_lane_dilatada == 1] = [0,255,0]
        cvimage[img_right_lane_dilatada == 1] = [0,0,255]
        

        mask_ = img_left_lane + img_right_lanes
        contornos, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_, contornos, -1, 255, thickness=cv2.FILLED)

        momentos = cv2.moments(mask_)

        # Calcular el centro de masa
        cx = int(momentos['m10'] / momentos['m00'])
        cy = int(momentos['m01'] / momentos['m00'])

        cv2.circle(cvimage,(cx,cy),10,(255,255,255),-1)

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