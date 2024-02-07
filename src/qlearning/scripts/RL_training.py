#! /usr/bin/env python3
import torch
import rospy
from sensor_msgs.msg import Image,PointCloud2,NavSatFix
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import onnxruntime as ort
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from yolop.msg import MassCentre
from scipy.interpolate import interp1d
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State,ExtendedState
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest,CommandHome,CommandHomeRequest
import sensor_msgs.point_cloud2
import warnings
import signal
import sys
import math
import random
import csv
import pandas as pd
import airsim
import datetime
import os


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

exit = False


n_steps = 0

state = 0

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

STATE_TERMINAL = len(STATES)

ACTIONS = [
    [1.45 ,1.5*-0.1],
    [1.45 ,1.5*-0.09],
    [1.46 ,1.5*-0.08],
    [1.46 ,1.5*-0.07],
    [1.47 ,1.5*-0.06],
    [1.47 ,1.5*0.05],
    [1.48 ,1.5*-0.04],
    [1.48 ,1.5*-0.03],
    [1.49 ,1.5*-0.02],
    [1.49 ,1.5 * -0.01],
    [1.5,0.0],
    [1.49, 1.5 * 0.01],
    [1.49, 1.5 * 0.02],
    [1.48, 1.5 * 0.03],
    [1.48, 1.5 * 0.04],
    [1.47, 1.5 * 0.05],
    [1.47, 1.5 * 0.06],
    [1.46, 1.5 * 0.07],
    [1.46, 1.5 * 0.08],
    [1.45, 1.5 * 0.09],
    [1.45,1.5 * 0.1],
]

def calculate_fps(t1,list_fps):
        fps = 1/(time.time() - t1)
        list_fps.append(fps)
        return sum(list_fps)/len(list_fps)


class QLearning:
    def __init__(self):
    
        
        
        self.QTable = np.zeros((len(STATES)+1,len(ACTIONS)))
        #self.QTable = np.genfromtxt('/home/bb6/pepe_ws/src/qlearning/trainings/03-febrero/q_table.csv', delimiter=',',skip_header=1,usecols=range(1,22))
        self.accumulatedReward = 0


        self.MAX_EPISODES = 300
        self.epsilon_initial = 0.95
        self.epsilon = 0.95




        self.alpha = 0.4 #--Between 0-1. 
        self.gamma = 0.7 #--Between 0-1.

        self.current_state_q = 0
        self.states = 0

        self.local_raw_pub = rospy.Publisher(LOCAL_VEL_PUB, Twist, queue_size=10)

        rospy.wait_for_service(SET_HOME_CLIENT)
        self.set_home_client = rospy.ServiceProxy(SET_HOME_CLIENT, CommandHome)

        rospy.wait_for_service(SET_MODE_CLIENT)
        self.set_mode_client = rospy.ServiceProxy(SET_MODE_CLIENT, SetMode)

        rospy.wait_for_service(ARMING_CLIENT)
        self.arming_client = rospy.ServiceProxy(ARMING_CLIENT, CommandBool)

        self.arm_cmd = CommandBoolRequest()
        self.arm_cmd.value = True

        self.set_mode_return = SetModeRequest()
        self.set_mode_return.custom_mode = 'AUTO.RTL'

        self.set_mode_takeoff = SetModeRequest()
        self.set_mode_takeoff.custom_mode = 'AUTO.TAKEOFF'

        self.set_mode_offboard = SetModeRequest()
        self.set_mode_offboard.custom_mode = 'OFFBOARD'

        self.set_mode_hold = SetModeRequest()
        self.set_mode_hold.custom_mode = 'AUTO.LOITER'

        self.set_mode_land = SetModeRequest()
        self.set_mode_land.custom_mode = 'AUTO.LAND'

        self.current_state = State()
        self.localization_gps = NavSatFix()
        
        self.state_sub = rospy.Subscriber(STATE_SUB, State, callback=self.state_cb)
        self.localization_gps_sub = rospy.Subscriber("/mavros/global_position/global",NavSatFix,callback=self.localization_gps_cb)
        self.extended_state = ExtendedState()
        self.extend_state_sub = rospy.Subscriber(EXTENDED_STATE,ExtendedState,callback=self.extended_state_cb)

        
        self.set_home_position = CommandHomeRequest()
        self.set_home_position.current_gps = False

        self.FIRST_LOCALIZATION = 1
        self.SECOND_LOCALIZATION = 2

        self.client_airsim = airsim.MultirotorClient(ip="192.168.2.16")
        self.client_airsim.confirmConnection()
        self.client_airsim.enableApiControl(True)



        """
        LOCALIZATION GPS 2
        yaw: 7.5
        latitude: 47.641977
        longitude: -122.1402614
        altitude: 99.93487272033067

        """

        """
        yaw: -80.0
        LOCALIZATION GPS 3
        latitude: 47.6424258
        longitude: -122.1404547
        altitude: 99.94147849127997


        """



        self.lidar_sub = rospy.Subscriber(LIDAR,PointCloud2,callback=self.lidar_cb)
        self.distance_z = 0.0

        self.velocity = Twist()
        self.velocity.linear.x = 0.0
        self.velocity.angular.z = 0.0

        self.last_req = 0.0
        self.has_armed = False
        self.has_return = False
        self.has_taken_off = False
        self.pepe = False

        self.KP_v = 0.01
        self.KD_v = 0.7

        self.prev_error_height = 0
        self.error = 0

        self.lastTime = 0.0

        self.CENTER_WEIGHT = 0.5 
        self.ORIENTATION_WEIGHT = 0.5


    def localization_gps_cb(self,msg):

        self.localization_gps = msg
        

    def state_cb(self, msg):
        self.current_state = msg

        self.lastTime = time.time()

    def extended_state_cb(self,msg):
        self.extended_state = msg
       

    def lidar_cb(self,cloud_msg):
        for point in sensor_msgs.point_cloud2.read_points(cloud_msg, field_names=("z"), skip_nans=True):
            self.distance_z = point[0] 

    def getState(self,cx):
       
        state_ = None
        for id_state in range(len(STATES)):
                
                if cx >= STATES[id_state][0] and cx <= STATES[id_state][1]: 
                    state_ = id_state

        if(state_ == None):
                #print("Estado frontera")
                state_ = STATE_TERMINAL 


                 
        return state_
        

    def chooseAction(self,state):

        n = random.uniform(0, 1)
       

        #--Exploration
        if n < self.epsilon:
            id_action = np.random.choice(len(ACTIONS))
            return ACTIONS[id_action],id_action
        #--Explotation
        else:
            id_action = np.argmax(self.QTable[state,:])
            return ACTIONS[id_action],id_action
        
    def functionQLearning(self,state,next_state,action,reward):

        #print("Table, State: " + str(state) + " Action: " + str(action) + "Next_State: " + str(next_state))
        #print(self.QTable[state, action])
        #print(np.argmax(self.QTable[next_state, action]))

        self.QTable[state, action] = self.QTable[state, action] + self.alpha * (reward + self.gamma * np.max(self.QTable[next_state]) - self.QTable[state, action])
        print("QTable[" + str(state) + "," + str(action) + "] : " + str(self.QTable[state, action]) + "reward: " + str(reward))
        
        self.accumulatedReward += reward
        #print(self.accumulatedReward,reward)

    def check_to_fly(self):
        if (self.arming_client.call(self.arm_cmd).success is True and self.has_armed is False):
            rospy.loginfo("Armed")
            self.has_armed = True
            return True

    def take_off(self):

        if self.check_to_fly():
            if (self.set_mode_client.call(self.set_mode_takeoff).mode_sent is True and self.has_taken_off is False):
                rospy.loginfo("TAKE OFF") 
                self.has_taken_off = True


    def land(self):
        if (self.set_mode_client.call(self.set_mode_land).mode_sent is True):
                rospy.loginfo("Land") 
                


   
    def reset_position(self):

        

        number_position = random.randint(1,2)
        x = 0
        y = 0
        z = 0

        pitch = 0.0
        yaw = 0.0
        roll = 0.0
        

    
        if(number_position == self.FIRST_LOCALIZATION):
            x = 0.0042578126303851604
            y = -0.008479003794491291
            z = -0.2734218239784241

            yaw =  0.5250219330319704
        

        elif(number_position == self.SECOND_LOCALIZATION):
            x = 28.493175506591797
            y = 16.92218589782715
            z = -0.2761923372745514

            yaw =  0.2228621193016977
        
           
        """
                else: 
            x = 52.03059387207031
            y = 21.48541259765625
            z = -0.2765296697616577

            yaw =  0.10057752672224413
        """



        position = airsim.Vector3r(x,y,z)
        orientation = airsim.to_quaternion(pitch,roll,yaw)
        pose = airsim.Pose(position,orientation)

        self.client_airsim.simSetVehiclePose(pose, True,"PX4")
        self.client_airsim.enableApiControl(True)
        self.client_airsim.armDisarm(False)
        print("Reinicie")

        time.sleep(1)



    def height_velocity_controller(self):
        
        self.error = round((2.80 - self.distance_z),2)
        derr = self.error - self.prev_error_height

        self.velocity.linear.z = (self.KP_v * self.error) + (self.KD_v * self.prev_error_height)
        


    def execute_action(self,action):

        
        self.set_mode_client.call(self.set_mode_offboard)
            
        
        #print("Correct heigh")
        self.height_velocity_controller()

        self.velocity.linear.x = action[0]
        #self.velocity.linear.y = action[1]
        self.velocity.angular.z = action[1]
        self.local_raw_pub.publish(self.velocity)
        self.prev_error_height = self.error
     

    def stop(self):

        self.set_mode_client.call(self.set_mode_offboard)
            
        
        self.velocity.linear.x = 0
        self.velocity.linear.y = 0
        self.velocity.linear.z = 0
        self.velocity.angular.z = 0
        self.local_raw_pub.publish(self.velocity)

    def reward_function(self,cx,angle):
        global exit
        
        reward = 0
        error_lane_center = (WIDTH/2 - cx)
        error_angle_orientation = 0.0 - angle
        
        if (self.is_exit_lane(error_lane_center,cx,error_angle_orientation)):
            exit = True
            reward = -10

        else:

            
            reward = (1 / (1 + abs(error_lane_center)))
            
            


        return reward
    
    def is_exit_lane(self,error,cx,angle_error):

        print(abs(angle_error))
        if (cx != -1) and ((is_not_detected_left is False ) or (is_not_detected_right is False)) and (abs(angle_error) < 9.0):
           return False
       
        else:
           print("Me sali")
           return True
        

    def is_finish_route(self):

        #print(self.localization_gps.latitude,self.localization_gps.longitude,self.localization_gps.altitude)
        #print(self.localization_gps.latitude >= 47.6426689,self.localization_gps.longitude >= -122.1407929,self.localization_gps.altitude >= 101.3139425)
        if(self.localization_gps.latitude >= 47.642141 and self.localization_gps.longitude >= -122.1402203 and self.localization_gps.altitude >= 101.5238980):
            print("Has acabado el recorrido")
            return True
        
        else:
            return False

        
    

    def algorithm(self,error,perception,current_state,centroid,error_angle):
        global n_steps,state


       
        while(not self.is_exit_lane(error,centroid,error_angle) and (not self.is_finish_route()) and (current_state != STATE_TERMINAL)):
            t0 = time.time()
            
            #print("Time: " +str(time.time() - self.lastTime))
            
           
            if time.time() - self.lastTime > 5.0:
                state = 5
                break
            

           
            action,id_action = qlearning.chooseAction(current_state)

            qlearning.execute_action(action)
            
            
            time.sleep(0.1)
            out_image,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
            
            

            self.client_airsim.simPause(True)
            #print("Pausado el simulador")
            #t1 = time.time()
           
            if (out_image is not None):
                perception.drawStates(out_image)
                cv2.circle(out_image,(cx,280),3,(0,0,0),-1)
                cv2.imshow("image",out_image)
                cv2.waitKey(3)
           


            #t4 = time.time()    
            reward = qlearning.reward_function(cx,angle_orientation)
            next_state = qlearning.getState(cx)

           
            qlearning.functionQLearning(current_state,next_state,id_action,reward)
            #t5 = time.time()
            self.client_airsim.simPause(False)
            #print("Tiempo de parar el simulador evaluar todo y reanudarlo: " + str(time.time() - t1))
            #print("Reanudado el simulador")
            #print("Current_state: " + str(current_state) + " , Next_state: " + str(next_state))
            current_state = next_state
            #print(current_state > 0 and current_state < 10)
            n_steps += 1
            
            error = (WIDTH/2 - cx) 
            error_angle = 0.0 - angle_orientation
            centroid = cx
            t1 = time.time()

        print("FPS train: " + str(1/(t1 - t0)))
            #print("Centroide: " + str(cx))
            #print("Error center : " + str(error) + " ,Error angle: " + str(error_angle))
            
           

        state = 4



class Perception():
    def __init__(self):
        self.bridge = CvBridge()
        self.tiempo_ultimo_callback = time.time()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback_img)
        self.cv_image = Image()
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

        self.address = ""

       

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
        global coefficients_left_global,is_not_detected_left
        """
        Calculate line thorugh lineal regression

        Args: 
                points_cluster: Numpy array, points cluster

        Return: 
            Line : numpy array,points line 
        """

        valuesX = np.arange(MIN_VALUE_X,MAX_VALUE_X) 
        punto = np.array([290, 0])
        line = None

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
                
                    #coefficients_left_global = mean_coeff

                    self.counter_it_left += 1

                    if(self.counter_it_left  > 5):
                      self.list_left_coeff_a.clear()
                      self.list_left_coeff_b.clear()
                      self.list_left_coeff_c.clear()
                    
                      self.counter_it_left = 0

                    self.left_fit = coefficients
                except np.RankWarning:
                    print("Polyfit may be poorly conditioned")
                    is_not_detected_left = True
                    #coefficients_left_global = mean_coeff
        except:
            print("He fallado")
            is_not_detected_left = True
            #mean_coeff = coefficients_left_global
        
        if(is_not_detected_left is False):
            values_fy = np.polyval(mean_coeff,valuesX).astype(int)
        
            fitLine_filtered = [(x, y) for x, y in zip(valuesX, values_fy) if 0 <= y <= 319]
            line = np.array(fitLine_filtered)

        return line
    

    def calculate_right_regression(self,points_cluster):
        global coefficients_right_global,is_not_detected_right
        """
        Calculate line thorugh lineal regression

        Args: 
                points_cluster: Numpy array, points cluster

        Return: 
            Line : numpy array,points line 
        """

        valuesX = np.arange(MIN_VALUE_X,MAX_VALUE_X) 
        punto = np.array([290,319])
        line = None

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
                    is_not_detected_right = True
                    #mean_coeff = coefficients_right_global
        except:
            print("He fallado")
            is_not_detected_right = True
            #mean_coeff = coefficients_right_global
        
        if(is_not_detected_right is False):
        
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
            
           
            if (len(left_clusters) == 0  or len(final_right_clusters) == 0):
                return None,None,-1
            
            else:

                return left_clusters,final_right_clusters,0
        
        else:
            return None,None,-1

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

        cross_product_z = PAx * PBy - PAy * PBx

        if (cross_product_z < 0 and angle_in_degrees > 1):
            self.address = "LEFT"
            
        elif(cross_product_z > 0 and angle_in_degrees > 1):
            self.address = "RIGHT"
            

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
        
            
            cx = 0 
            cy = 0
            orientation_angle = 0
            
           
            if(left_clusters and right_clusters):
                left = np.concatenate(left_clusters,axis=0)
                right = np.concatenate(right_clusters,axis=0)

                

                cvimage[left[:,0],left[:,1]] = [0,0,255]
                cvimage[right[:,0],right[:,1]] = [0,255,0]

                
                points_line_right =  self.calculate_right_regression(right)
                points_line_left = self.calculate_left_regression(left)


                if (points_line_right is None or points_line_left is None):
                    print("Error en la regresion")
                    cx = -1 
                    cy = -1
                    orientation_angle = -1
                    cvimage = None
                
                else:            
                    img_line_left,img_line_right = self.dilate_lines(points_line_left,points_line_right)

                    cvimage[img_line_left == 1] = [255,255,255]
                    cvimage[img_line_right == 1] = [255,255,255]

                    points_beetween_lines = self.interpolate_lines(cvimage,points_line_left,points_line_right)

                    cvimage[points_beetween_lines[:,0],points_beetween_lines[:,1]] = [255,0,0]

                    
                    cx,cy = self.calculate_mass_centre_lane(points_beetween_lines)
                    orientation_angle = self.calculate_angle([cx,cy],[160,cy],cvimage)

                #cv2.circle(cvimage, (self.cx,self.cy), radius=10, color=(0, 0, 0),thickness=-1)
               
                #cv2.line(cvimage,(160,320),(self.cx,self.cy),(0,0,0),3)
               

            else:
                print("Error de los puntos numpy")
                print(left_clusters,right_clusters)
                cvimage = None
                cx = -1 
                cy = -1 
                orientation_angle = -1

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


    def calculate_lane(self,cv_image):

        global is_not_detected_left,is_not_detected_right
        out_img = None
        cx = 0
        cy = 0
        orientation_angle = 0
        images_yolop = self.infer_yolop(cv_image)
        #print("Tiempo YOLOP: " + str(1/(time.time() - self.t1)))
        mask_cvimage = self.draw_region(cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))
        mask = self.draw_region(images_yolop[1])
        
        left_clusters,right_clusters,state_clusters = self.clustering(mask,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))
       
        #print("Clustering: " + str(1 / (time.time() - self.t1)))

        if state_clusters == -1:
            print("Error en clustering")
            cx = -1
            cy = -1 
            orientation_angle = -1 
            is_not_detected_left = True
            is_not_detected_right = True

        
        else:
           
            out_img,cx,cy,orientation_angle = self.calculate_margins_points(left_clusters,right_clusters,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR),images_yolop[0])
            

            #print("Regresion + interpolate + centroid: " + str(1/(time.time() - self.t1)))
        return out_img,cx,cy,orientation_angle
        
        
    def callback_img(self, data):
        

        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 


    def correct_initial_position(self,error,qlearning):
        global state

        while(abs(error) > 0):
            if time.time() - qlearning.lastTime > 5.0:
                state = 5
                break
            qlearning.height_velocity_controller()
            qlearning.set_mode_client.call(qlearning.set_mode_offboard)
               
        
            if perception.address == "LEFT":
                qlearning.velocity.angular.z = abs((0.009 * error))
            else:
                qlearning.velocity.angular.z = (0.009 * error)

            qlearning.local_raw_pub.publish(qlearning.velocity)
            qlearning.prev_error_height = qlearning.error
            _,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
            error  = WIDTH/2 - cx
            #print(error)
        
        _,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
        error  = WIDTH/2 - cx
        #print("Error of final: " + str(error))
        state = 2

        
if __name__ == '__main__':
    
    rospy.init_node("RL_node_py")
    
    perception = Perception()
    qlearning = QLearning()

    cx = 0
    cy = 0
    angle_orientation = 0
    current_state = 0

    list_ep_it = []
    list_ep_epsilon = []
    list_ep_accumulate_reward = []

    error = 0
    n_episode = 0

    rate = rospy.Rate(20)
    is_landing = False
    t_initial = time.time()
    t_counter_ep = 0

    t2 = 0.0

    

   
    while (not rospy.is_shutdown()):
        try:
            if(state == 0):
                t_episode = time.time()
                if(qlearning.extended_state.landed_state == STATE_ON_GROUND):
                    qlearning.take_off()
                
                elif(qlearning.extended_state.landed_state == STATE_IN_AIR):
                    state = 1

            if(state == 1):
                _,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
                #angle_error = 0.0 - angle_orientation
                error = (WIDTH/2 - cx)

                perception.correct_initial_position(error,qlearning)

            if(state == 2):

                if(n_episode < qlearning.MAX_EPISODES):
                    n_episode += 1
                    state = 3
                else:
                    state = 5
            
            if(state == 3):

                _,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)

                error = (WIDTH/2 - cx)
                error_angle = 0.0 - angle_orientation
                #print("Error antes de hacer avanzar, en la correccion" + str(error))
                current_state = qlearning.getState(cx)
             
                t2 = time.time()
                qlearning.algorithm(error,perception,current_state,cx,error_angle)
                
            if(state == 4):
                
               
                if(not is_landing):
                    print("Tiempo final: " + str(time.time() - t2))
                    cv2.destroyAllWindows()
                    qlearning.stop()
                    if (qlearning.set_mode_client.call(qlearning.set_mode_hold).mode_sent is True):
                            rospy.loginfo("HOLD")
                    time.sleep(2)
                    qlearning.land()
                    #time.sleep(1)
                    
                    is_landing = True

                else:
                    time.sleep(6)
                    if(qlearning.extended_state.landed_state == STATE_ON_GROUND and qlearning.current_state.armed is False):
                        if (qlearning.set_mode_client.call(qlearning.set_mode_hold).mode_sent is True):
                            rospy.loginfo("HOLD")

                        qlearning.reset_position()
                        qlearning.has_armed = False
                        qlearning.has_taken_off = False
                        is_landing = False
                        is_not_detected_left = False
                        is_not_detected_right = False
                        exit = False

                        print("ID_EPISODE: " + str(n_episode) + " N_STEPS: " + str(n_steps) + " epsilon: " + str(qlearning.epsilon))
                        if(n_episode < 900):
                            qlearning.epsilon = qlearning.epsilon_initial - (n_episode * (qlearning.epsilon_initial / 900))
                        else:
                            qlearning.epsilon = 0
                        print("Tiempo por episodio: " + str(time.time() - t_episode))
                        t_counter_ep +=time.time() - t_episode
                        
                        
                        list_ep_it.append([n_episode,n_steps])
                        list_ep_epsilon.append([n_episode,qlearning.epsilon])
                        list_ep_accumulate_reward.append([n_episode,qlearning.accumulatedReward])
                        n_steps = 0
                        qlearning.accumulatedReward = 0
                        state = 0
                        

                        if(n_episode ==  qlearning.MAX_EPISODES):
                            state = 5

            if(state == 5):
               
                break

            rate.sleep()

        except rospy.ROSInterruptException:
            print("Interrumpido C el nodo")
            break

        except rospy.service.ServiceException:
            print("Parado el servicio")
            break

        except KeyboardInterrupt:
            print("Control C")
            break
        except OSError:
            print("Se produjo un error inesperado capturado")
            break
        except RuntimeError:
            print("Se produjo un error runtime")
            break
        except AttributeError:
            print("Se produjo un error attributeError")
            break


    print("Ha acabado")  
    print("Tiempo de entrenamiento: " + str(time.time() - t_initial))
    print("Media de tiempo por episodio: " + str(t_counter_ep/qlearning.MAX_EPISODES))
    print(qlearning.QTable)  



    
    #fecha = datetime.datetime.now().strftime('%d-%B')

    
    #carpeta = f'/home/bb6/pepe_ws/src/qlearning/trainings/{fecha}'

    
    #os.makedirs(carpeta, exist_ok=True)

    
   
    with open('/home/bb6/pepe_ws/src/qlearning/trainings/07-febrero/episodes-iterations.csv', 'w') as file:
        wtr = csv.writer(file, delimiter= ' ')
        wtr.writerows(list_ep_it)

    with open('/home/bb6/pepe_ws/src/qlearning/trainings/07-febrero/episodes-epsilon.csv', 'w') as file:
        wtr = csv.writer(file, delimiter= ' ')
        wtr.writerows(list_ep_epsilon)

    with open('/home/bb6/pepe_ws/src/qlearning/trainings/07-febrero/episodes-accumulated-reward.csv', 'w') as file:
        wtr = csv.writer(file, delimiter= ' ')
        wtr.writerows(list_ep_accumulate_reward)

    df = pd.DataFrame(qlearning.QTable)
    df.to_csv('/home/bb6/pepe_ws/src/qlearning/trainings/07-febrero/q_table.csv')
    
    
    
    
      
   
    
 


