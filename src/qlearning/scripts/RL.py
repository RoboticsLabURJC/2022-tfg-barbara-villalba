#! /usr/bin/env python3
import torch
import rospy
from sensor_msgs.msg import Image,PointCloud2
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
from mavros_msgs.msg import State,ExtendedState
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest,CommandHome,CommandHomeRequest
import sensor_msgs.point_cloud2
import warnings
import signal
import sys
import math
import random

IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = "/home/bb6/YOLOP/weights/yolop-320-320.onnx"

MIN_VALUE_X = 185
MAX_VALUE_X = 320

HEIGH = 320
WIDTH = 320

X_PER_PIXEL = 3.0 #--Width road 

#--Topics
STATE_SUB = "mavros/state"
EXTENDED_STATE = "/mavros/extended_state"
MODE_SUB = "/commands/mode"
LOCAL_VEL_PUB = "/mavros/setpoint_velocity/cmd_vel_unstamped"
LIDAR = "/airsim_node/PX4/lidar/LidarCustom"
MASS_CENTRE = "/yolop/detection_lane/mass_centre_lane"

#--Services
ARMING_CLIENT = "/mavros/cmd/arming"
SET_MODE_CLIENT = "/mavros/set_mode"
TAKE_OFF_CLIENT = "/mavros/cmd/takeoff"
SET_HOME_CLIENT = "/mavros/cmd/set_home"
SET_MODE_CLIENT = "/mavros/set_mode"

OFFBOARD = "OFFBOARD"

STATE_ON_GROUND = 1
STATE_IN_AIR = 2
STATE_IN_TAKE_OFF = 3
STATE_IN_LANDING = 4

coefficients_left_global = np.array([])
coefficients_right_global = np.array([])

is_not_detected_left = False
is_not_detected_right = False

cx_global = 0.0
cy_global = 0.0

frames_ = 0  

vy_lineal = 0
vz_angular = 0
vz_lineal = 0

n_episodes = 0
n_steps = 0


STATES = [(i, i+20) for i in range(60, 240, 20)]
STATES.insert(0,(0,60))
STATES.append((240,320))

STATE_TERMINAL_LEFT = 0
STATE_TERMINAL_RIGHT = 10


#--Speeds and rate speeds
ACTIONS = [
    [0.2, -0.1],
    [0.2, -0.09],
    [0.3, -0.08],
    [0.3, -0.07],
    [0.5, -0.06],
    [0.8, 0.05],
    [1.0, -0.04],
    [1.3, -0.03],
    [1.6, -0.02],
    [1.8, -0.01],
    [2.0, 0.0],
    [1.8, 0.01],
    [1.6, 0.02],
    [1.3, 0.03],
    [1.0, 0.04],
    [0.8, 0.05],
    [0.5, 0.06],
    [0.3, 0.07],
    [0.3, 0.08],
    [0.2, 0.09],
    [0.2, 0.1],
]

def calculate_fps(t1,list_fps):
        fps = 1/(time.time() - t1)
        list_fps.append(fps)
        return sum(list_fps)/len(list_fps)


class QLearning:
    def __init__(self):
    
        self.QTable = np.zeros((len(STATES),len(ACTIONS)))  

        self.MAX_EPISODES = 1
        self.epsilon = 0.5
        self.alpha = 0.4 #--Between 0-1. 
        self.gamma = 0.5 #--Between 0-1.
        self.episodes = [0,0]
        self.end_episode = False
        self.begin_episode = False
        self.is_first_episode = False
        self.n_episode = 0

        self.current_state = 0

        self.local_raw_pub = rospy.Publisher(LOCAL_VEL_PUB, Twist, queue_size=10)

        rospy.wait_for_service(SET_HOME_CLIENT)
        self.set_home_client = rospy.ServiceProxy(SET_HOME_CLIENT, CommandHome)

        rospy.wait_for_service(SET_MODE_CLIENT)
        self.set_mode_client = rospy.ServiceProxy(SET_MODE_CLIENT, SetMode)

        rospy.wait_for_service(ARMING_CLIENT)
        self.arming_client = rospy.ServiceProxy(ARMING_CLIENT, CommandBool)

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True

        self.set_mode = SetModeRequest()
        self.set_mode.custom_mode = 'AUTO.RTL'

        self.current_state = State()
        self.state_sub = rospy.Subscriber(STATE_SUB, State, callback=self.state_cb)
        self.extended_state = ExtendedState()
        self.extend_state_sub = rospy.Subscriber(EXTENDED_STATE,ExtendedState,callback=self.extended_state_cb)

        self.set_home_position = CommandHomeRequest()
        self.set_home_position.current_gps = False
        self.set_home_position.yaw = 42.0
        self.set_home_position.latitude = 47.6416705
        self.set_home_position.longitude =  -122.1405088
        self.set_home_position.altitude =  101.36056744315884

        self.lidar_sub = rospy.Subscriber(LIDAR,PointCloud2,callback=self.lidar_cb)
        self.distance_z = 0.0

        self.velocity = Twist()

        self.last_req = 0.0
        self.has_armed = False
        self.has_taken_off = False
        self.end_program = False

    def state_cb(self, msg):
        self.current_state = msg

    def extended_state_cb(self,msg):
        self.extended_state = msg
       

    def lidar_cb(self,cloud_msg):
        for point in sensor_msgs.point_cloud2.read_points(cloud_msg, field_names=("z"), skip_nans=True):
            self.distance_z = point[0] 
            
        
    def getState(self,cx):
       
        state = None
        for id_state in range(len(STATES)):
          
            if cx >= STATES[id_state][0]  and cx <= STATES[id_state][1]: 
                state = id_state
                
        return state

    def chooseAction(self,state):

        n = random.uniform(0, 1)

        #--Exploration
        if n < self.epsilon:
            id_action = random.randint(0, len(ACTIONS) - 1)
            return ACTIONS[id_action]
        
        #--Explotation
        else:
            return np.argmax(self.QTable[state, :])
        

    def getSpeedAction(self,id_action):
        return ACTIONS[id_action][0]
        
    def getAngularSpeedAction(self,id_action):
        return ACTIONS[id_action][1]


    def functionQLearning(self,state,next_state,action,reward):
        self.QTable[state, action] = self.QTable[state, action] + self.alpha * (reward + self.gamma * np.argmax(self.QTable[next_state, :]) - self.QTable[state, action])


    def check_to_fly(self):
        if (self.arming_client.call(self.arm_cmd).success is True and self.has_armed is False):
            rospy.loginfo("Armed")
            self.has_armed = True
            return True

    def take_off(self):

       

        if self.check_to_fly():
            self.set_mode.custom_mode = 'AUTO.TAKEOFF'
            if (self.set_mode_client.call(self.set_mode).mode_sent is True and self.has_taken_off is False):
                rospy.loginfo("TAKE OFF") 
                self.has_taken_off = True


    
    def return_home_position(self):
        
        if (self.set_home_client.call(self.set_home_position)):
            rospy.loginfo("He cambiado la posicion de casa") 

        if (self.set_mode_client.call(self.set_mode).mode_sent is True):
            rospy.loginfo("RETURN MODE") 
            self.has_return = True

    def execute_action(self,action):


        self.velocity.linear.x = action[0]
        self.velocity.angular.z = action[1]
        #self.local_raw_pub.publish(self.velocity)

    def stop(self):
        
        self.velocity.linear.x = 0
        self.velocity.angular.z = 0

    def reward_function(self,cx,angle,speed):
        error_lane_center = (WIDTH/2 - cx)*(X_PER_PIXEL/WIDTH)

        #--Weight for each component 
        w1 = 0.5
        w2 = 0.5

        if is_not_detected_left or is_not_detected_right:
            reward = -10

        else:

            reward = (1/abs(error_lane_center)) * w1 + angle * w2

        return reward

    def startEpisode(self):
        
        self.episodes[0] += 1

    def is_beggin_episode(self):
        global n_episodes
        if(self.begin_episode is False):
            n_episodes += 1
            self.begin_episode = True


    
class QLearningTraining:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback_img)
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
      
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        self.isfirst = False 
        self.fps = 0
        self.list = []
        self.counter_time = 0.0

        self.cx = 0
        self.cy = 0

        self.orientation_angle = 0.0

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

        self.qlearning = QLearning()





 
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
        punto = np.array([300, 0])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    points_cluster = np.vstack((points_cluster,punto*2))
                  
                    coefficients = np.polyfit(points_cluster[:,0],points_cluster[:,1],2)
                    

                    self.list_left_coeff_a.append(coefficients[0])
                    self.list_left_coeff_b.append(coefficients[1])
                    self.list_left_coeff_c.append(coefficients[2])
#
                    a = np.mean(self.list_left_coeff_a[-10:])
                    b = np.mean(self.list_left_coeff_b[-10:])
                    c = np.mean(self.list_left_coeff_c[-10:])

                    mean_coeff = np.array([a,b,c])
                
                    coefficients_left_global = mean_coeff

                    self.counter_it_left += 1

                    if(self.counter_it_left  > 5):
                      self.list_left_coeff_a.clear()
                      self.list_left_coeff_b.clear()
                      self.list_left_coeff_c.clear()
                    
                      self.counter_it_left = 0

                    self.left_fit = coefficients
                except np.RankWarning:
                    print("Polyfit may be poorly conditioned")
                    coefficients_left_global = mean_coeff
        except:
            print("He fallado")
            mean_coeff = coefficients_left_global
        
        values_fy = np.polyval(mean_coeff,valuesX).astype(int)
       
        fitLine_filtered = [(x, y) for x, y in zip(valuesX, values_fy) if 0 <= y <= 319]
        line = np.array(fitLine_filtered)

        return line
    

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
        punto = np.array([320,319])

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    points_cluster = np.vstack((points_cluster,punto))
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


                    coefficients_right_global = mean_coeff

                    self.counter_it_right += 1

                    if(self.counter_it_right  > 5):
                      self.list_right_coeff_a.clear()
                      self.list_right_coeff_b.clear()
                      self.list_right_coeff_c.clear()  
                      self.counter_it_right = 0
                except np.RankWarning:
                    print("Polyfit may be poorly conditioned")
                    mean_coeff = coefficients_right_global
        except:
            print("He fallado")
            mean_coeff = coefficients_right_global
        
        values_fy = np.polyval(mean_coeff,valuesX).astype(int)
        fitLine_filtered = [(x, y) for x, y in zip(valuesX, values_fy) if 0 <= y <= 319]
        line = np.array(fitLine_filtered)

        return line
    
    def score_cluster(self,cluster, center):
        points_cluster, centroid = cluster
      
        proximity = np.linalg.norm(centroid - center)
        density = len(points_cluster)
        return density / proximity
    

    def clustering(self,img,cv_image):

       
        #--Convert image in points
        points_lane = np.column_stack(np.where(img > 0))
        dbscan = DBSCAN(eps=10, min_samples=1,metric="euclidean")
        left_clusters = []
        right_clusters = []
        center = np.array([200,160])
        #interest_point = np.array((180,160))
       

        final_left_clusters = []
        final_right_clusters = []

        img = cv_image.copy()
        
        if points_lane.size > 0:
            dbscan.fit(points_lane)
            labels = dbscan.labels_

            # Ignore noise if present
            clusters = set(labels)
            if -1 in clusters:
                clusters.remove(-1)
          
          
                
            for cluster in clusters:
                points_cluster = points_lane[labels==cluster,:]
                centroid = points_cluster.mean(axis=0).astype(int)
                color = self.colors[cluster % len(self.colors)]
                #img[points_cluster[:,0], points_cluster[:,1]] = color

                # Check if the centroid is within the desired lane
                if centroid[1] < 160:  # left lane
                    left_clusters.append(points_cluster)
                elif centroid[1] > 160:  # right lane
                    right_clusters.append((points_cluster, centroid))
                    #print(centroid)
                    cv2.circle(cv_image, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)
                    img[points_cluster[:,0], points_cluster[:,1]] = [0,0,255]

               
            # Now, among the closest clusters, select the one with the highest density
           
           
            """
            
            if left_clusters:
                left_clusters = [max(left_clusters, key=lambda x: self.score_cluster(x, center))]
            """
            
           
            if right_clusters:
                right_clusters = [max(right_clusters, key=lambda x: self.score_cluster(x, center))]

            
            for points_cluster in left_clusters:
               

                
                
                #color = self.colors[cluster % len(self.colors)]
                cv_image[points_cluster[:,0], points_cluster[:,1]] = [0,255,0]
                #cv2.circle(cv_image, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)
            
            for points_cluster, centroid in right_clusters:
               
                final_right_clusters.append(points_cluster)
                
                #color = self.colors[cluster % len(self.colors)]
                cv_image[points_cluster[:,0], points_cluster[:,1]] = [0,0,255]
                #
            
           
            
            return left_clusters,final_right_clusters,cv_image,img
        
        else:
            return None,None

    def draw_region(self,img):


        mask = np.zeros_like(img) 

        cv2.fillPoly(mask, self.vertices,(255,255,255))
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image
    
    
    
    def calculate_mass_centre_lane(self,points_lane):

            
        # Supongamos que todos los puntos tienen la misma masa
        m_i = 1

        # Calcula la masa total
        m_total = m_i * len(points_lane)

        # Calcula la suma de las posiciones de los puntos ponderadas por su masa
        r_i_sum = np.sum(points_lane * m_i, axis=0)

        # Calcula la posición del centro de masas
        r_CM = r_i_sum / m_total


        return int(r_CM[1]),int(r_CM[0])
      

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
    
    
        
    def calculate_margins_points(self,left_clusters,right_clusters,cvimage,img_da_seg):
        
            
            if(left_clusters and right_clusters):
                left = np.concatenate(left_clusters,axis=0)
                right = np.concatenate(right_clusters,axis=0)

                

                cvimage[left[:,0],left[:,1]] = [0,0,255]
                cvimage[right[:,0],right[:,1]] = [0,255,0]

                points_line_right =  self.calculate_right_regression(right)
                points_line_left = self.calculate_left_regression(left)

            
                img_line_left,img_line_right = self.dilate_lines(points_line_left,points_line_right)

                cvimage[img_line_left == 1] = [255,255,255]
                cvimage[img_line_right == 1] = [255,255,255]

                points_beetween_lines = self.interpolate_lines(cvimage,points_line_left,points_line_right)

                cvimage[points_beetween_lines[:,0],points_beetween_lines[:,1]] = [255,0,0]

                
                cx,cy = self.calculate_mass_centre_lane(points_beetween_lines)
                orientation_angle = self.calculate_angle([cx,cy],[160,cy],cvimage)

                #cv2.circle(cvimage, (self.cx,self.cy), radius=10, color=(0, 0, 0),thickness=-1)
               
                #cv2.line(cvimage,(160,320),(self.cx,self.cy),(0,0,0),3)
               

            return cvimage,cx,cy,orientation_angle
    


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
                    (STATES[i][0] + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255,255,255),
                    1,
                    cv2.LINE_AA,
                )


    def perception(self,cv_image):
        images_yolop = self.infer_yolop(cv_image)
        mask_cvimage = self.draw_region(cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))
        mask = self.draw_region(images_yolop[1])
        left_clusters,right_clusters,img_cluster,img = self.clustering(mask,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))

        if left_clusters  != None and right_clusters != None:
            out_img,cx,cy,orientation_angle = self.calculate_margins_points(left_clusters,right_clusters,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR),images_yolop[0])
            return out_img,cx,cy,orientation_angle
        
        else:
            return None,-1,-1,-1



    def callback_img(self, data):
        global n_episodes,n_steps

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 
        print(n_episodes)


        if(self.qlearning.extended_state.landed_state == STATE_ON_GROUND and self.qlearning.end_program is False):
            self.qlearning.take_off()

        elif(self.qlearning.extended_state.landed_state == STATE_IN_AIR):
            print("STATE_IN_AIR")

            if(n_episodes < self.qlearning.MAX_EPISODES):
               self.qlearning.is_beggin_episode()

               if(self.qlearning.begin_episode):
                   print("begin_episode")
                   
                   if(self.qlearning.is_first_episode is False):
                    _,cx,cy,angle_orientation = self.perception(cv_image)
                    self.qlearning.current_state = self.qlearning.getState(cx)
                    print("first_episode")
                    self.qlearning.is_first_episode = True

                   
                   if((self.qlearning.current_state == STATE_TERMINAL_LEFT or self.qlearning.current_state == STATE_TERMINAL_RIGHT) or (n_steps > 1)):
                       self.qlearning.stop()
                       self.qlearning.return_home_position()
                       self.qlearning.has_armed = False
                       self.qlearning.has_taken_off = False
                       self.qlearning.begin_episode = False
                       self.qlearning.is_first_episode = False
                    
                   else:
                       n_steps += 1
                       action = self.qlearning.chooseAction(self.qlearning.current_state)
                      
                       self.qlearning.execute_action(action)
                       #self.qlearning.stop()
                       """
                       
                       _,cx,cy,angle_orientation = self.perception(cv_image)
                       next_state = self.qlearning.getState(cx)
                       reward = self.qlearning.reward_function(cx,angle_orientation,self.qlearning.getSpeedAction(action))
                       self.qlearning.functionQLearning(self.qlearning.current_state,next_state,action,reward)
                       self.qlearning.current_state = next_state
                       """
                     
 
            else:
                
                self.qlearning.end_program = True

if __name__ == '__main__':
    rospy.init_node("RL_node_py")
    
    control = QLearningTraining()
   
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Parado")


    
        
   
