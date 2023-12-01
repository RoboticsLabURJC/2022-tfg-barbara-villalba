#! /usr/bin/env python3
import torch
import rospy
from sensor_msgs.msg import Image,PointCloud2
import sensor_msgs.point_cloud2
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
from nav_msgs.msg import Odometry
import warnings
from numpy import RankWarning




IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = "/home/bb6/YOLOP/weights/yolop-320-320.onnx"

MIN_VALUE_X = 170
MAX_VALUE_X = 320

HEIGH = 320
WIDTH = 320

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

left_fit_global = np.ndarray([])
right_fit_global = np.ndarray([])

cx_global = 0.0
cy_global = 0.0




class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)
        ort.set_default_logger_severity(4)
        self.ort_session = ort.InferenceSession(ROUTE_MODEL,providers=['CUDAExecutionProvider'])
        self.list = []
        self.bottom_left_base = [0,320]
        self.bottom_right_base = [320,320]
        self.bottom_left  = [0, 210]
        self.bottom_right = [320,210]
        self.top_left     = [80,175]
        self.top_right    = [200, 175]
        self.vertices = np.array([[self.bottom_left_base,self.bottom_left, self.top_left, self.top_right, self.bottom_right,self.bottom_right_base]], dtype=np.int32)
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
    
    def calculate_lineal_regression(self,points_cluster):

        """
        Calculate line thorugh lineal regression

        Args: 
                points_cluster: Numpy array, points cluster

        Return: 
            Line : numpy array,points line 
        """

        valuesX = np.arange(MIN_VALUE_X,MAX_VALUE_X) 
        coefficients = np.polyfit(points_cluster[:,0],points_cluster[:,1],1)
        values_fy = np.polyval(coefficients,valuesX).astype(int)
        fitLine_filtered = [(x, y) for x, y in zip(valuesX, values_fy) if 0 <= y <= 319]
        line = np.array(fitLine_filtered)

        return line

    def clustering(self,img):

        """
        Calculate clusters with DBSCAN algorithm and we save each cluster with his points and his centroids

        Args: 
               img:  normalise image ll_seg_out (out yolop)

        Return: 
               dict_clusters: dictionary with all clusters {id_cluster,points,centroid}
        """
        #--Convert image in points
        points_lane = np.column_stack(np.where(img > 0))

        dbscan = DBSCAN(eps=15, min_samples=1,metric="euclidean")
        if points_lane.size > 0:
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

            return result
        
        else:
            return None
        
    
    def draw_region(self,img):


        mask = np.zeros_like(img) 

        cv2.fillPoly(mask, self.vertices,(255,255,255))
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image
    
    def calculate_mass_centre_lane(self,points_lane):

        global cx_global,cy_global

        img = np.zeros((WIDTH, HEIGH), dtype=np.uint8)
        img[points_lane[:,0],points_lane[:,1]] = 255

        contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contornos, -1, 255, thickness=cv2.FILLED)

        momentos = cv2.moments(img)

        # Calcular el centro de masa
        try:
            cx = int(momentos['m10'] / momentos['m00'])
            cy = int(momentos['m01'] / momentos['m00'])

            cx_global = cx
            cy_global = cy

        except BaseException as e:
            cx = cx_global
            cy = cy_global


        return cx,cy
    
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
    
    def interpolate_lines(self,img_det,points_line_left,points_line_right):

        """
       We calculate the interpolation of the lines in order to limit the lane area we want to show. 
       We use the scipy.interpolate library  

        Args: 
              points_line_left: Points line left dilate
              points_line_right: Points line right dilate

        Returns: 
              points_beetween_lines: Points lane between 2 lines
              
              
        """

        points_road = np.column_stack(np.where(img_det > 0))

        if(points_road.size > 0):

            f1 = interp1d(points_line_left[:, 0], points_line_left[:, 1],fill_value="extrapolate")
            f2 = interp1d(points_line_right[:, 0], points_line_right[:, 1],fill_value="extrapolate") 
            y_values_f1 = f1(points_road[:, 0])
            y_values_f2 = f2(points_road[:, 0])
            indices = np.where((y_values_f1 <= points_road[:, 1]) & (points_road[:, 1] <= y_values_f2))
            
            
            points_between_lines = points_road[indices]

            return points_between_lines

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

        images = []

        for image in (da_seg_out,ll_seg_out):

            #--Convert tensor to a numpy and then it realise transpose : (H, W, C) 
            #--image_np = image.numpy()
            image_array = np.transpose(image, (2, 3, 1, 0))

            #--Normalise numpy a values 0-255
            image_norm = cv2.normalize(image_array[:,:,1,:], None, 0,1, cv2.NORM_MINMAX, cv2.CV_8U)

            images.append(image_norm)


        return images

        
    def calculate_margins_points(self,img,cvimage,img_da_seg):
        global left_fit_global,right_fit_global
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

        valuesLineX = np.arange(170,320)
        
        #print("Left: " + str(leftx.size) + " , " + str(lefty.size))
        #print("Right: " + str(rightx.size) + " , " + str(righty.size))

        try:
            left_fit = np.polyfit(lefty,leftx, 1)
            right_fit = np.polyfit(righty,rightx,1) 
            left_fit_global = left_fit
            right_fit_global = right_fit

        except BaseException as e:
           print("Entre")
           left_fit =  left_fit_global
           right_fit = right_fit_global
    
        #print(left_fit,right_fit)

        fitLineLeftY = np.polyval(left_fit, valuesLineX).astype(int)
        fitLineRightY = np.polyval(right_fit, valuesLineX).astype(int)
        
       

        # Filtrar los valores de fitLineY_capped que sean mayores de 150 y menores de 320
        fitLineLeftY_filtered = [(x, y) for x, y in zip(valuesLineX, fitLineLeftY) if 0 <= y <= 319]

        # Filtrar los valores de fitLineY_capped que sean mayores de 150 y menores de 320
        fitLineRightY_filtered = [(x, y) for x, y in zip(valuesLineX, fitLineRightY) if 0 <= y <= 319]

        left_lane_points = np.array(fitLineLeftY_filtered)
        right_lane_points = np.array(fitLineRightY_filtered)

        img_left_lane_dilatada,img_right_lane_dilatada = self.dilate_lines(left_lane_points,right_lane_points)

        cvimage[img_left_lane_dilatada == 1] = [255,255,255]
        cvimage[img_right_lane_dilatada == 1] = [255,255,255]

        points_beetween_lines = self.interpolate_lines(img_da_seg,left_lane_points,right_lane_points)


        cvimage[points_beetween_lines[:,0],points_beetween_lines[:,1]] = [255,0,0]

        self.cx,self.cy = self.calculate_mass_centre_lane(points_beetween_lines)

        cv2.circle(cvimage, (self.cx,self.cy), radius=10, color=(0, 0, 0),thickness=-1)
       


        return cvimage
    
    def calculate_fps(self):
        fps = 1/(time.time() - self.t1)
        return fps
    
    def callback(self, data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv2.GaussianBlur(cv_image,(3,3),0) 

        """
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype = np.float64)
        hsv[:,:,1] = hsv[:,:,1]*1.25 # scale pixel values up for channel 1
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*1.35 # scale pixel values up for channel 2
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        """


     
        images_yolop = self.infer_yolop(cv_image)
        #images_yolop2 = self.infer_yolop(cv_image)
        mask_cvimage = self.draw_region(cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))
        mask = self.draw_region(images_yolop[1])

        
        img_clustering= self.clustering(mask)

        if img_clustering is None:
            return 


        cv_image = cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR)
        #img[images_yolop[1] == 1] = [0,0,255]
        cvimage = cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR)
        #cvimage[images_yolop2[1] == 1] = [0,0,255]
        out_img = self.calculate_margins_points(img_clustering,cv_image,images_yolop[0])
                   
        self.counter_time =  self.counter_time + (time.time() - self.t1)

        if(not self.isfirst):
            self.fps = self.calculate_fps()
            self.isfirst = True

        #--Update each 1 seconds
        if(self.counter_time > 0.25):
            self.fps = self.calculate_fps()
            self.counter_time = 0.0


        #cv2.line(out_img,(50,0),(50,320),(0,0,0),3)
        #cv2.line(out_img,(60,0),(60,320),(0,0,0),3)
        cv2.putText(
                out_img, 
                text = "FPS: " + str(int(self.fps)),
                org=(0, 15),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )



        cv2.imshow('Image',out_img)
        cv2.imshow('Sin Brillo',cvimage)
        cv2.imshow('Mask',mask_cvimage)
    
        # Press `q` to exit.
        cv2.waitKey(3)
        

class LaneFollow():
    def __init__(self):
        self.current_state = State()
        self.distance_z = 0.0
        self.lidar_sub = rospy.Subscriber(LIDAR,PointCloud2,callback=self.lidar_cb)
        self.state_sub = rospy.Subscriber(STATE_SUB, State, callback=self.state_cb) 
        self.local_raw_pub = rospy.Publisher(LOCAL_VEL_PUB, Twist, queue_size=10)
        rospy.wait_for_service(SET_MODE_CLIENT)
        self.set_mode_client = rospy.ServiceProxy(SET_MODE_CLIENT, SetMode)
        self.prev_error_height = 0
        self.prev_error = 0
        self.error = 0
        self.error_height = 0
        self.KP_w = 0.03
        self.KD_w = 0.005
        self.KP_v = 0.005
        self.KD_v = 0.07
    
        self.velocity = Twist()


    def lidar_cb(self,cloud_msg):
        #self.msg_lidar = msg
        #print(ros_numpy.point_cloud2.get_xyz_points(msg))
        for point in sensor_msgs.point_cloud2.read_points(cloud_msg, field_names=("z"), skip_nans=True):
            self.distance_z = point[0] 

    def state_cb(self, msg):
        self.current_state = msg

    
    def height_velocity_controller(self):
        error = round((2.5 - self.distance_z),2)
        derr = error - self.prev_error_height

        self.velocity.linear.z = (self.KP_v * error) + (self.KD_v * derr)
        print("Error: " + str(error))
    def velocity_controller(self,cx):
        self.error = WIDTH/2 - cx
        derr = self.error - self.prev_error
        #print(self.error)

        self.velocity.angular.z = (self.KP_w * self.error) + (self.KD_w * derr)

        
if __name__ == '__main__':
    rospy.init_node("det_node_py")
    image_viewer = ImageSubscriber()
    lane_follow = LaneFollow()

    set_mode = SetModeRequest()
    set_mode.custom_mode = 'OFFBOARD'

    rate = rospy.Rate(20)

    lane_follow.velocity.linear.x = 1.5
   

    last_req = rospy.Time.now()

    while (not rospy.is_shutdown()):
        
        if (lane_follow.current_state.mode != OFFBOARD and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if (lane_follow.set_mode_client.call(set_mode).mode_sent is True):
                rospy.loginfo("OFFBOARD enabled")

        print(lane_follow.distance_z)
        lane_follow.velocity_controller(image_viewer.cx)
        lane_follow.height_velocity_controller()
        lane_follow.prev_error_height = lane_follow.error_height
        lane_follow.prev_error = lane_follow.error_height
        lane_follow.local_raw_pub.publish(lane_follow.velocity)

        rate.sleep()

    #image_viewer.start()