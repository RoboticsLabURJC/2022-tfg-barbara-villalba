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

@jit(nopython=True)
def get_first_points_lanes(img_clustering):

    first_left_lane_point = []
    first_right_lane_point = []

       
    #--LEFT
    for y in range(319,-1,-1):
        for x in range(0,120):
            if img_clustering[y,x] == 255:
                first_left_lane_point.append([x,y])
                break
        if first_left_lane_point:
            break

    #--RIGHT
    for y in range(319,-1,-1):
        for x in range(319,120,-1):
            if img_clustering[y,x] == 255:
                first_right_lane_point.append([x,y])
                break
        if first_right_lane_point:
            break

    return first_left_lane_point,first_right_lane_point



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
        self.vertices = np.array([[self.bottom_left_base,self.bottom_left, self.top_left, self.top_right, self.bottom_right,self.bottom_right_base]], dtype=np.int32)
        self.point_cluster = np.ndarray
        

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
    
    def get_lane_line_indices_sliding_windows(self,img,right_x,left_x):
        """
        Get the indices of the lane line pixels using the 
        sliding windows technique.
            
        :param: plot Show plot or not
        :return: Best fit lines for the left and right lines of the current lane 
        """
        no_of_windows = 50
        height = img.shape[0]
        width = img.shape[1]

        # El ancho de la ventana deslizante es +/- margen
        margin = 50
        minpix = 0
    
        #Copiamos la imagen
        frame_sliding_window = img.copy()
    
        # Establecer la altura de las ventanas correderas.
        window_height = np.int_(height/no_of_windows)       
    
        # Encuentra las coordenadas x e y de todos los distintos de cero
        # (es decir, blancos) píxeles en el marco.
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
            
        # Almacene los índices de píxeles para las líneas de los carriles izquierdo y derecho
        left_lane_inds = []
        right_lane_inds = []
            
        # Posiciones actuales para índices de píxeles para cada ventana,
        # que continuaremos actualizando
        leftx_current = left_x #tu MAN
        rightx_current = right_x
    
        # Ir a través de una ventana a la vez

            
        for window in range(no_of_windows):
        # Identificar los límites de la ventana en x e y (y derecha e izquierda)
            #esquina abajo
            win_y_low = width - (window + 1) * window_height
            #esquina arriba
            win_y_high = width - window * window_height

            win_xleft_low = leftx_current - margin
            if(win_xleft_low <= 0):
                win_xleft_low = 0
            win_xleft_high = leftx_current + margin

            cv2.rectangle(frame_sliding_window,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (255,255,255), 1)
            

            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(frame_sliding_window,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (255,255,255), 1)
        
            # Identificar los píxeles distintos de cero (es decir, blancos) en x e y dentro de la ventana
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                                                                
            # Agregar estos índices a las listas.
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
                
            # Si encontró > píxeles minpix, vuelva a centrar la siguiente ventana en la posición media
            if len(good_left_inds) > minpix:
                leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
                
            if len(good_right_inds) > minpix:        
                rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))
            

        """         
        # Concatenar los arrays de índices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # Extrae las coordenadas de píxeles para las líneas de los carriles izquierdo y derecho
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds] 
        righty = nonzeroy[right_lane_inds]
    
        # Ajustar una curva polinómica de segundo orden a las coordenadas de píxeles para
        # las líneas de los carriles izquierdo y derecho
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2) 
        """

        return frame_sliding_window




    def clustering(self,img):
        #--Convert image in points
        points_lane = np.column_stack(np.where(img > 0))

        dbscan = DBSCAN(eps=25, min_samples=1,metric="euclidean")
        dbscan.fit(points_lane)
        n_clusters_ = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
        #print("Clusters: " + str(n_clusters_))
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
        cvimage[images[1] == 1] = [0, 0, 255]


        mask = self.draw_region(images[1])

        masked_image = self.draw_region(cvimage)


        img_clustering,points_lanes = self.clustering(mask)

        first_left_lane_point,first_right_lane_point = get_first_points_lanes(img_clustering)

        img_frame_window = self.get_lane_line_indices_sliding_windows(img_clustering,first_right_lane_point[0][0],first_left_lane_point[0][0])
        

        #--print(first_left_lane_point,first_right_lane_point)
        
        cv2.circle(masked_image,(first_left_lane_point[0][0],first_left_lane_point[0][1]),5,(0,255,0),-1)
        cv2.circle(masked_image,(first_right_lane_point[0][0],first_right_lane_point[0][1]),5,(0,255,255),-1)

        #print(first_point)

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
        return cvimage,masked_image,img_clustering,img_frame_window

    def callback(self, data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        image_result,mask,img_cluster,img_frame_window = self.infer_yolop(cv_image)

        cv2.imshow('Image',image_result)
        cv2.imshow('Mask',mask)
        cv2.imshow('img_cluster',img_cluster)
        cv2.imshow('Image-windows',img_frame_window)
        #time.sleep(1)
        
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