#! /usr/bin/env python3
import torch
import rospy
from sensor_msgs.msg import Image,PointCloud2,NavSatFix,Imu
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
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest
import sensor_msgs.point_cloud2
import warnings
import signal
import sys
import math
import keyboard
from tf.transformations import euler_from_quaternion
from scipy.spatial import ConvexHull


IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = "/home/bb6/YOLOP/weights/yolop-320-320.onnx"

MIN_VALUE_X = 185
MAX_VALUE_X = 320

HEIGH = 320
WIDTH = 320

X_PER_PIXEL = 3.0 #--Width road 

#--Topics
STATE_SUB = "mavros/state"
MODE_SUB = "/commands/mode"
LOCAL_VEL_PUB = "/mavros/setpoint_velocity/cmd_vel_unstamped"
LIDAR = "/airsim_node/PX4/lidar/LidarCustom"
MASS_CENTRE = "/yolop/detection_lane/mass_centre_lane"

#--Services
ARMING_CLIENT = "/mavros/cmd/arming"
SET_MODE_CLIENT = "/mavros/set_mode"
TAKE_OFF_CLIENT = "/mavros/cmd/takeoff"

OFFBOARD = "OFFBOARD"

coefficients_left_global = np.array([])
coefficients_right_global = np.array([])

cx_global = 0.0
cy_global = 0.0

frames_ = 0  

vy_lineal = 0
vz_angular = 0
vz_lineal = 0

STATES = [
    (77,87),
    (88,98),
    (99,109),
    (110, 120),
    (121, 131),
    (132, 142),
    (143, 153),
    (154, 164),
    (165, 175),
    (176, 186),
    (187, 197),
    (198, 208),
    (209, 219),
    (220, 230),
]

print(len(STATES))

is_first = False

is_exit_lane = False

counter = 0
def calculate_fps(t1,list_fps):
        fps = 1/(time.time() - t1)
        list_fps.append(fps)
        return sum(list_fps)/len(list_fps)


    

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)
        self.imu_sub = rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback)
        ort.set_default_logger_severity(4)
        self.ort_session = ort.InferenceSession(ROUTE_MODEL,providers=['CUDAExecutionProvider'])
        self.list = []
        self.bottom_left_base = [0,320]
        self.bottom_right_base = [320,320]
        self.bottom_left  = [0, 320]
        self.bottom_right = [320,320]
        self.top_left     = [0,200]
        self.top_right    = [320, 200]
        self.vertices = np.array([[self.bottom_left,self.top_left,self.top_right,self.bottom_right]], dtype=np.int32)
        self.point_cluster = np.ndarray
        self.kernel = np.array([[0,1,0], 
                                [1,1,1], 
                                [0, 1,0]]) 
      
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.isfirst = False 
        self.fps = 0
        self.list = []
        self.counter_time = 0.0

        self.cx = 0
        self.cy = 0

        self.list_left_coeff_a = []
        self.list_left_coeff_b = []
        self.list_left_coeff_c = []

        self.mean_left_coeff = np.array([])

        self.list_right_coeff_a = []
        self.list_right_coeff_b = []
        self.list_right_coeff_c = []

        self.mean_right_coeff = np.array([])

        self.left_fit = None
        self.right_fit = None

        
        self.counter_it_left = 0
        self.counter_it_right = 0

        self.prev_distance = None
        self.prev_density = None

        self.orientation_angle = 0.0
        self.cx_prev = 0


    def imu_callback(self,data):
        quaternion = (
        data.orientation.x,
        data.orientation.y,
        data.orientation.z,
        data.orientation.w
       )

        # Convierte los cuaterniones a ángulos de Euler (roll, pitch, yaw)
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        # Ahora `roll` contiene el ángulo de balanceo en radianes
        # Puedes convertirlo a grados si lo prefieres
        roll = roll * 180.0 / 3.14159

        #print("Roll: ", roll)


 
    def resize_unscale(self,img, new_shape=(640, 640), color=114):
        """
        Resize image for model onnx

        Args:   
                new_shape: Shape image that it must match with shape model onnx
                color: Color image

        Return: 
            Image resize 
        """
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
    
    def calculate_left_regression(self,points_cluster):
        global coefficients_left_global
        """
        Calculate line thorugh lineal regression

        Args: 
                points_cluster: Numpy array, points cluster

        Return: 
            Line : numpy array,points line 
        """

        valuesX = np.arange(MIN_VALUE_X,MAX_VALUE_X) 
        punto = np.array([290, 0])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    points_cluster = np.vstack((points_cluster,punto*15))
                  
                    coefficients = np.polyfit(points_cluster[:,0],points_cluster[:,1],2)
                    

                    self.list_left_coeff_a.append(coefficients[0])
                    self.list_left_coeff_b.append(coefficients[1])
                    self.list_left_coeff_c.append(coefficients[2])
#
                    a = np.mean(self.list_left_coeff_a[-10:])
                    b = np.mean(self.list_left_coeff_b[-10:])
                    c = np.mean(self.list_left_coeff_c[-10:])

                    mean_coeff = np.array([a,b,c])
                
                    

                    self.counter_it_left += 1

                    if(self.counter_it_left  > 5):
                      self.list_left_coeff_a.clear()
                      self.list_left_coeff_b.clear()
                      self.list_left_coeff_c.clear()
                    
                      self.counter_it_left = 0

                    self.left_fit = coefficients
                except np.RankWarning:
                    print("Polyfit may be poorly conditioned")
                    
        except:
            print("He fallado")
            
        
        values_fy = np.polyval(mean_coeff,valuesX).astype(int)
       
        fitLine_filtered = [(x, y) for x, y in zip(valuesX, values_fy) if 0 <= y <= 319]
        line = np.array(fitLine_filtered)

        m = (2*a*160 + b)

        max_y_index = np.argmax(line[:,0])

            
        extrem_point_line = line[max_y_index]

        return line,extrem_point_line,m
    

    def calculate_right_regression(self,points_cluster):
        global coefficients_right_global
        """
        Calculate line thorugh lineal regression

        Args: 
                points_cluster: Numpy array, points cluster

        Return: 
            Line : numpy array,points line 
        """

        valuesX = np.arange(MIN_VALUE_X,MAX_VALUE_X) 
        punto = np.array([290,319])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    points_cluster = np.vstack((points_cluster,punto*2))
                    coefficients = np.polyfit(points_cluster[:,0],points_cluster[:,1],2)

                    self.list_right_coeff_a.append(coefficients[0])
                    self.list_right_coeff_b.append(coefficients[1])
                    self.list_right_coeff_c.append(coefficients[2])
#
                    a = np.mean(self.list_right_coeff_a[-10:])
                    b = np.mean(self.list_right_coeff_b[-10:])
                    c = np.mean(self.list_right_coeff_c[-10:])
#
                    mean_coeff = np.array([a,b,c])


                   

                    self.counter_it_right += 1

                    if(self.counter_it_right  > 5):
                      self.list_right_coeff_a.clear()
                      self.list_right_coeff_b.clear()
                      self.list_right_coeff_c.clear()  
                      self.counter_it_right = 0
                except np.RankWarning:
                    print("Polyfit may be poorly conditioned")
                  
        except:
            print("He fallado")
          
        
        
        values_fy = np.polyval(mean_coeff,valuesX).astype(int)
        fitLine_filtered = [(x, y) for x, y in zip(valuesX, values_fy) if 0 <= y <= 319]
        line = np.array(fitLine_filtered)

        m = (2*a*160 + b)
       

        max_y_index = np.argmax(line[:,0])

            
        extrem_point_line = line[max_y_index]


        return line,extrem_point_line,m
    
    def score_cluster(self,cluster, center):
        points_cluster, centroid = cluster
      
        proximity = np.linalg.norm(centroid - center)
        density = len(points_cluster)
        return density / proximity
    

    def clustering(self,img,cv_image):

       
        #--Convert image in points
        points_lane = np.column_stack(np.where(img > 0))
        
        dbscan = DBSCAN(eps=1, min_samples=1,metric="euclidean")
        left_clusters = []
        right_clusters = []
        center = np.array([200,200])
        #interest_point = np.array((180,160))
       

        final_left_clusters = []
        final_right_clusters = []

        img = cv_image.copy()
        counter_left_points = 0
        
        if points_lane.size > 0:
            dbscan.fit(points_lane)
            labels = dbscan.labels_

            #print("Labels: " + str(labels))
            clusters = set(labels)

            """
            
            # Ignore noise if present
            
            if -1 in clusters:
                clusters.remove(-1)
            """
          
          
                
            for cluster in clusters:
                points_cluster = points_lane[labels==cluster,:]
                centroid = points_cluster.mean(axis=0).astype(int)
                color = self.colors[cluster % len(self.colors)]
                img[points_cluster[:,0], points_cluster[:,1]] = color
                cv2.circle(img, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)

                #print("Centroid: " + str(centroid[1]))


                # Check if the centroid is within the desired lane
                if centroid[1] < 160:  # left lane
                    left_clusters.append((points_cluster,centroid))
                    #img[points_cluster[:,0], points_cluster[:,1]] = [0,255,0]
                    #print("Izquierda: " + str(len(points_cluster)))
                    #cv2.circle(img, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)
                elif centroid[1] > 160:  # right lane
                    right_clusters.append((points_cluster, centroid))
                    #print(centroid)
                    #cv2.circle(img, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)
                    #img[points_cluster[:,0], points_cluster[:,1]] = [0,0,255]

               
            # Now, among the closest clusters, select the one with the highest density
           
           
        
            if left_clusters:
                left_clusters = [max(left_clusters, key=lambda x: self.score_cluster(x, center))]
            
            
            
            
           
            if right_clusters:
                right_clusters = [max(right_clusters, key=lambda x: self.score_cluster(x, center))]

            
            for points_cluster,centroid in left_clusters:
               
                final_left_clusters.append(points_cluster)
                counter_left_points +=len(points_cluster)
                #
                
                #color = self.colors[cluster % len(self.colors)]
                cv_image[points_cluster[:,0], points_cluster[:,1]] = [0,255,0]
                
            
            for points_cluster, centroid in right_clusters:
               
                #print("Derecha: " + str(len(points_cluster)))
                #print("----------------")
                final_right_clusters.append(points_cluster)
               
                
                #color = self.colors[cluster % len(self.colors)]
                cv_image[points_cluster[:,0], points_cluster[:,1]] = [0,0,255]
                #
            
            """
           
            if (counter_left_points == 0):
                print("No tengo puntos")
            
             """
            return final_left_clusters,final_right_clusters,cv_image,img
        
        else:
            return None,None
        
        
        
        
    
    def draw_region(self,img):


        mask = np.zeros_like(img) 

        cv2.fillPoly(mask, self.vertices,(255,255,255))
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image
    
    
    
    def calculate_mass_centre_lane(self,points_lane,img_det,cv_image):

        global cx_global
        #print(len(points_lane))
        if(points_lane.size > 50):
            
            # Supongamos que todos los puntos tienen la misma masa
            m_i = 1

            # Calcula la masa total
            m_total = m_i * len(points_lane)

            # Calcula la suma de las posiciones de los puntos ponderadas por su masa
            r_i_sum = np.sum(points_lane * m_i, axis=0)

            # Calcula la posición del centro de masas
            r_CM = r_i_sum / m_total

            cx_global = int(r_CM[1])
            cy_global = int(r_CM[0])
            

            return int(r_CM[1]),int(r_CM[0])
            
             
        
        else:
            print("No tengo puntos para calcular")
            #print(cx_global)
            return cx_global
        

    def calculate_angle(self,A, B,img):
        
        Ax, Ay = A
        Bx, By = B

        #--Point P: imagen vertical
        Px, Py = [160, 0]

        # Vectores PA y PB
        PAx, PAy = Ax - Px, Ay - Py
        PBx, PBy = Bx - Px, By - Py

       


        # Producto escalar y magnitudes
        dot_product = PAx * PBx + PAy * PBy
        magnitude_PA = math.sqrt(PAx**2 + PAy**2)
        magnitude_PB = math.sqrt(PBx**2 + PBy**2)

        # Ángulo en radianes

    

        angle_in_radians = math.acos(dot_product / (magnitude_PA * magnitude_PB))

        # Convertir a grados
        angle_in_degrees = math.degrees(angle_in_radians)

        cross_product_z = PAx * PBy - PAy * PBx

        """
        
        if (cross_product_z < 0 and angle_in_degrees > 1):
            print("Gira izquierda")
        elif(cross_product_z > 0 and angle_in_degrees > 1):
            print("Gira derecha")
        """
       
        return angle_in_degrees
        
    
    def dilate_lines(self,left_line_points,right_line_points):

        """
        Dilate the points of each line 

        Args: 
              left_lane_points: Points line left
              right_lane_points: Points line right 

        Returns: 
              img_left_line_dilatada: Image left line dilate
              img_right_line_dilatada: Image right line dilate
              
        """

        img_left_line = np.zeros((WIDTH, HEIGH), dtype=np.uint8)
        img_right_line = np.zeros((WIDTH, HEIGH), dtype=np.uint8)

        img_left_line[left_line_points[:,0],left_line_points[:,1]] = 255
        img_right_line[right_line_points[:,0],right_line_points[:,1]] = 255

        
        img_left_line_dilatada = binary_dilation(img_left_line, structure=self.kernel)
        img_left_line_dilatada = (img_left_line_dilatada).astype(np.uint8)

        img_right_line_dilatada = binary_dilation(img_right_line, structure=self.kernel)
        img_right_line_dilatada = (img_right_line_dilatada).astype(np.uint8)


        return img_left_line_dilatada,img_right_line_dilatada
    
    def interpolate_lines(self,cvimage,points_line_left,points_line_right):

        """
       We calculate the interpolation of the lines in order to limit the lane area we want to show. 
       We use the scipy.interpolate library  

        Args: 
              points_line_left: Points line left dilate
              points_line_right: Points line right dilate

        Returns: 
              points_beetween_lines: Points lane between 2 lines
              
              
        """

        gray_image = cv2.cvtColor(cvimage, cv2.COLOR_BGR2GRAY) 

        np_gray = np.array(gray_image)

        x, y = np.nonzero(np_gray)


        img_points = np.column_stack((x, y))

        f1 = interp1d(points_line_left[:, 0], points_line_left[:, 1],fill_value="extrapolate")
        f2 = interp1d(points_line_right[:, 0], points_line_right[:, 1],fill_value="extrapolate") 
        y_values_f1 = f1(img_points[:, 0])
        y_values_f2 = f2(img_points[:, 0])
        
        indices = np.where((y_values_f1 < img_points[:, 1]) & (img_points[:, 1] < y_values_f2))
        
        points_between_lines = img_points[indices]
        filtered_points_between_lines = points_between_lines[points_between_lines[:,0] > 180]

        return filtered_points_between_lines

    def infer_yolop(self,cvimage):

        canvas, r, dw, dh, new_unpad_w, new_unpad_h = self.resize_unscale(cvimage, (320,320))

        img = canvas.copy().astype(np.float32)  # (3,320,320) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,320,320)

        self.t1 = time.time()
        # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
        _, da_seg_out, ll_seg_out = self.ort_session.run(
            ['det_out', 'drive_area_seg', 'lane_line_seg'],
            input_feed={"images": img}
        )
        #print(1/(time.time() - self.t1))

        images = []

        for image in (da_seg_out,ll_seg_out):

            #--Convert tensor to a numpy and then it realise transpose : (H, W, C) 
            #--image_np = image.numpy()
            image_array = np.transpose(image, (2, 3, 1, 0))

            #--Normalise numpy a values 0-255
            image_norm = cv2.normalize(image_array[:,:,1,:], None, 0,1, cv2.NORM_MINMAX, cv2.CV_8U)

            images.append(image_norm)


        return images
    
    
    #-- En los numpys, [y,x]
    def calculate_margins_points(self,left_clusters,right_clusters,cvimage,img_da_seg):
        
            global is_first,is_exit_lane
            
            left = np.concatenate(left_clusters,axis=0)
            right = np.concatenate(right_clusters,axis=0)

            

            cvimage[left[:,0],left[:,1]] = [0,0,255]
            cvimage[right[:,0],right[:,1]] = [0,255,0]

            points_line_right,extrem_point_line_right,m_right =  self.calculate_right_regression(right)
            points_line_left,extrem_point_line_left,m_left = self.calculate_left_regression(left)

            

            #angle = np.arctan(abs((m_right - m_left) / (1 + m_left*m_right)))
            
            

            cv2.circle(cvimage, (extrem_point_line_left[1],extrem_point_line_left[0]), radius=7, color=(0, 0, 0),thickness=-1)
            cv2.circle(cvimage, (extrem_point_line_right[1],extrem_point_line_right[0]), radius=7, color=(0, 0, 0),thickness=-1)





            

            img_line_left,img_line_right = self.dilate_lines(points_line_left,points_line_right)

            cvimage[img_line_left == 1] = [255,255,255]
            cvimage[img_line_right == 1] = [255,255,255]

            points_beetween_lines = self.interpolate_lines(cvimage,points_line_left,points_line_right)


            #--Detect right turn
            if (extrem_point_line_left[1] > 140 and extrem_point_line_right[1] > WIDTH/2):
                is_exit_lane = True
                cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

            #--Detect left turn
            elif(extrem_point_line_left[1] < WIDTH/2 and extrem_point_line_right[1] < WIDTH/2):
                is_exit_lane = True
                cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

            else:

               
                max_y_index = np.argmin(points_line_right[:,0])
                extrem_point_line2 = points_line_right[max_y_index]
                
                max_y_left_index = np.argmin(points_line_left[:,0])
                extrem_point_left_line2 = points_line_left[max_y_left_index]

                #print("Pendiente derecha: " + str((extrem_point_line_right[0] - extrem_point_line2[0])/(extrem_point_line2[1] - extrem_point_line2[1]) ))
                #print("Pendiente izquierda: " + str((extrem_point_line_left[0] - extrem_point_left_line2[0])/(extrem_point_line_left[1] - extrem_point_left_line2[1]) ))

                print(extrem_point_line2[1] - extrem_point_left_line2[1])

                if(extrem_point_line2[1] - extrem_point_left_line2[1] >= 100):
                    is_exit_lane = True
                    cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

                else:
                    is_exit_lane = False





            cvimage[points_beetween_lines[:,0],points_beetween_lines[:,1]] = [255,0,0]

            
            
            self.cx,self.cy = self.calculate_mass_centre_lane(points_beetween_lines,img_da_seg,cvimage)

            if (is_first is False):
                self.cx_prev = self.cx
                is_first = True
            


            self.orientation_angle = self.calculate_angle([self.cx,self.cy],[160,self.cy],cvimage)

            

            cv2.circle(cvimage, (self.cx,self.cy), radius=4, color=(0, 0, 0),thickness=-1)

          

            # Crea una imagen en blanco del tamaño que necesites
            imagen = np.zeros((320, 320), dtype=np.uint8)

            
            #cv2.line(cvimage,(160,0),(160,320),(0,0,0),)
               
            return cvimage
    


    def drawStates(self,out_image):

        thickness = 1
        imageH = out_image.shape[0]
        
        for i in range(len(STATES)):
           
            for x in STATES[i]:
                sp = (x, 0)
                ep = (x, imageH)
                cv2.line(out_image, sp, ep, (255, 255, 255), thickness)
                cv2.putText(
                    out_image,
                    "S" + str(int(i)),
                    (STATES[i][0] + 3, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.2,
                    (255,255,255),
                    1,
                    cv2.LINE_AA,
                )


    


    
    def callback(self, data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        #cv_image = cv2.GaussianBlur(cv_image, (5, 5), 0)

        t1 = time.time()
        images_yolop = self.infer_yolop(cv_image)
        mask_cvimage = self.draw_region(cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))
        mask = self.draw_region(images_yolop[1])
        left_clusters,right_clusters,img_cluster,img = self.clustering(mask,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))

        
        out_img = self.calculate_margins_points(left_clusters,right_clusters,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR),images_yolop[0])
        #print(1/(time.time() - t1))
        #self.drawStates(out_img)
        
        
    
        # Dibuja el contorno en la imagen
        #cv2.drawContours(imagen, [contorno], -1, (255, 255, 255), 2)

        # Muestra la imagen
        
      
        #cv2.line(img_cluster,(160,320),(160,0),(255,0,255),3)
        #cv2.line(img_cluster,(160,180),(160,320),(255,0,255),1)
#
      #
        #
        #cv2.line(img_cluster,(210,180),(320,250),(0,255,255),2)
#
        #cv2.line(img_cluster,(55,180),(0,230),(0,0,255),2)
        #cv2.line(img_cluster,(230,180),(320,230),(0,0,255),2)


    
        image_resize = cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR) 
       
        #mask_cvimage[images_yolop[0] == 1] = [255,0,0]
        #mask_cvimage[images_yolop[1] == 1] = [0,255,0]

        #points_lane = np.column_stack(np.where(images_yolop[1] > 0))
        #print(points_lane)
        #mask_cvimage[points_lane[0,:],points_lane[1,:]] = [255,0,0]


        #out_img = self.calculate_margins_points(dict_clusters,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR),images_yolop[0])
        
        self.counter_time =  self.counter_time + (time.time() - self.t1)

        if(not self.isfirst):
            self.fps = calculate_fps(self.t1,self.list)
            self.isfirst = True

        #--Update each 0.8 seconds
        if(self.counter_time > 0.8):
            self.fps = calculate_fps(self.t1,self.list)
            self.counter_time = 0.0

        cv2.line(img,(0,200),(320,200),(0,255,255),2)
        #cv2.imshow('image',mask_cvimage)
        cv2.imshow('image_cluster',img_cluster)
        cv2.imshow('image',out_img)
        

        # Press `q` to exit.
        cv2.waitKey(3)
        

class LaneFollow():
    def __init__(self):
        self.current_state = State()
        self.distance_z = 0.0
        self.lidar_sub = rospy.Subscriber(LIDAR,PointCloud2,callback=self.lidar_cb)
        self.state_sub = rospy.Subscriber(STATE_SUB, State, callback=self.state_cb) 
        self.local_raw_pub = rospy.Publisher(LOCAL_VEL_PUB, Twist, queue_size=10)
        #rospy.wait_for_service(SET_MODE_CLIENT)
        self.set_mode_client = rospy.ServiceProxy(SET_MODE_CLIENT, SetMode)
        self.localization_gps = NavSatFix()
        self.localization_gps_sub = rospy.Subscriber("/mavros/global_position/global",NavSatFix,callback=self.localization_gps_cb)
        self.prev_error_height = 0
        self.prev_error = 0
        self.error = 0
        self.derr = 0
        self.integral_error = 0.0 
        self.error_height = 0
        self.KP_v = 0.009
        self.KD_v = 0.7
        self.velocity = Twist()
      

        self.max_linear_velocity = 1.8  # The maximum speed you want to reach
        self.current_linear_velocity = 0.0  # The current speed, which is initially 0
        self.acceleration = 0.01  # The rate at which you want to increase the speed

        self.t1 = 0.0

        self.point_A_vertex = [47.642184,-122.140238]
        self.point_B_vertex = [47.642187,-122.140186]
        self.point_C_vertex = [47.642198,-122.140241]
        self.point_D_vertex = [47.642201,-122.140188]



    def localization_gps_cb(self,msg):

        self.localization_gps = msg


    def lidar_cb(self,cloud_msg):
        z_points = [point[0] for point in sensor_msgs.point_cloud2.read_points(cloud_msg, field_names=("z"), skip_nans=True)]
        self.distance_z = sum(z_points) / len(z_points) 
        print(self.distance_z)  

    def state_cb(self, msg):
        self.current_state = msg

    
    def height_velocity_controller(self):
        global vz_lineal
        error = round((2.80 - self.distance_z),2)
        derr = error - self.prev_error_height

        self.velocity.linear.z = (self.KP_v * error) + (self.KD_v * derr)
        vz_lineal = self.velocity.linear.z
        #print("Error altura: " + str(error))
    def velocity_controller(self,cx):

        global vy_lineal,vz_angular

        self.error = (WIDTH/2 - cx)*(X_PER_PIXEL/WIDTH)
        self.integral_error += self.error 
        print(self.error)

        if(abs(self.error) <= 0.01):
           self.integral_error = 0 

       
       
        self.velocity.angular.z = (1.1 * self.error)  + (0.005 *(self.error - self.prev_error)) + (0.0001 *(self.integral_error))
        self.velocity.linear.y =  (0.9 * self.error) + (0.001 *(self.error - self.prev_error)) + (0.00001 *(self.integral_error))

        

        vy_lineal = self.velocity.linear.y
        vz_angular = self.velocity.angular.z

    def handler(self,signum, frame):
        self.velocity.linear.x = 0.0
        self.velocity.linear.y = 0.0
        self.velocity.angular.z = 0.0
        self.local_raw_pub.publish(self.velocity)
        print('You pressed Ctrl+C!')

        sys.exit(0)




if __name__ == '__main__':
   
    rospy.init_node("det_node_py")
    image_viewer = ImageSubscriber()

   
    lane_follow = LaneFollow()

    set_mode = SetModeRequest()
    set_mode.custom_mode = 'OFFBOARD'

    rate = rospy.Rate(10)


    last_req = rospy.Time.now()

    start_time = time.time()  
    frames = 0

    lane_follow.velocity.linear.x = 1.5
    lane_follow.velocity.linear.y = 0.0
    lane_follow.velocity.angular.z = 0.0

    signal.signal(signal.SIGINT, lane_follow.handler)

    """
    
    if (A[0] <= P[0] <= C[0]) and (B[1] <= P[1] <= A[1]) and (B[0] <= P[0] <= D[0]) and (D[1] <= P[1] <= C[1]):
    print("El punto P está dentro del rectángulo.")
    
    else:
        print("El punto P está fuera del rectángulo.")
    """

    

    while (not rospy.is_shutdown()):
        print("Centre lane error: " + str((WIDTH/2 - image_viewer.cx)))
        print("Error angle: " + str(image_viewer.orientation_angle))

        

        #print("Diferencia de centroides en valor absoluto: " + str(abs(image_viewer.cx - image_viewer.cx_prev)))
        #image_viewer.cx_prev = image_viewer.cx

        """
       
        if (lane_follow.current_state.mode != OFFBOARD and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if (lane_follow.set_mode_client.call(set_mode).mode_sent is True):
                rospy.loginfo("OFFBOARD enabled")
        
       
        if (lane_follow.current_state.mode == OFFBOARD):
            lane_follow.velocity_controller(image_viewer.cx)
            lane_follow.height_velocity_controller()
        

        lane_follow.local_raw_pub.publish(lane_follow.velocity)
     
        lane_follow.prev_error = lane_follow.error
        lane_follow.prev_error_height = lane_follow.error_height
      
        
        if time.time() - start_time >= 1:
          
            frames_ = frames
            start_time = time.time()
            frames = 0
        """
        rate.sleep()
