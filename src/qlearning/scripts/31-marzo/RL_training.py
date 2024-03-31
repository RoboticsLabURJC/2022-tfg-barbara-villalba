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
import threading


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

is_first = False

cx_global = 0.0
cy_global = 0.0

exit = False

is_not_detect_lane = False
is_first_time = True


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
    (231, 241),
]

STATE_TERMINAL = len(STATES)

counter_actions = 0


ACTIONS = []
speeds_actions = np.linspace(1.45,2.45,11, dtype=float)
angular_speeds = np.linspace(-0.25, 0.25, 21)

def build_actions():

    left_angular_speeds = angular_speeds[:10]
    right_angular_speeds = np.flip(angular_speeds[-10:])
    central_angular_speed = angular_speeds[10]
    speeds = speeds_actions[:10]
    central_speed = speeds_actions[10]

    for i in range(len(speeds)):
        #print(round(speeds[i],2),round(right_angular_speeds[i],2))
        ACTIONS.append([round(speeds[i],3),round(left_angular_speeds[i],3)])

    ACTIONS.append([round(central_speed,3),0.0])

    for  i in reversed (range(len(speeds))):
        ACTIONS.append([round(speeds[i],3),round(right_angular_speeds[i],3)])

build_actions()
print(ACTIONS)

MAX_EXPLORATIONS = 900



class QLearning:
    def __init__(self):
    
        self.QTable = np.zeros((len(STATES)+1,len(ACTIONS)))
        #self.QTable = np.genfromtxt('/home/bb6/pepe_ws/src/qlearning/trainings/29-marzo/q_table.csv', delimiter=',',skip_header=1,usecols=range(1,22))
        self.accumulatedReward = 0
       

        self.MAX_EPISODES = rospy.get_param('~max_episodes')
        self.epsilon_initial = 0.95

        n_episode = rospy.get_param('~n_episode')

        if(n_episode == 0):
            self.epsilon = self.epsilon_initial
        elif(n_episode >= MAX_EXPLORATIONS):
            self.epsilon = 0
        else:
            self.epsilon = self.epsilon_initial - ((n_episode) * (self.epsilon_initial / MAX_EXPLORATIONS))
            print(self.epsilon,n_episode)


        self.alpha = 0.5 #--Between 0-1. 
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
        self.latitud = 0.0
        self.longitud = 0.0
        
        self.state_sub = rospy.Subscriber(STATE_SUB, State, callback=self.state_cb)
        self.localization_gps_sub = rospy.Subscriber("/mavros/global_position/global",NavSatFix,callback=self.localization_gps_cb)
        self.extended_state = ExtendedState()
        self.extend_state_sub = rospy.Subscriber(EXTENDED_STATE,ExtendedState,callback=self.extended_state_cb)

        
        self.set_home_position = CommandHomeRequest()
        self.set_home_position.current_gps = False

        self.FIRST_LOCALIZATION = 1
        self.SECOND_LOCALIZATION = 2

        self.client_airsim = airsim.MultirotorClient(ip="192.168.1.16")
        self.client_airsim.confirmConnection()
        self.client_airsim.enableApiControl(True)

        #--Points to road final
        self.point_A_vertex = [47.642184,-122.140238]
        self.point_B_vertex = [47.642187,-122.140186]
        self.point_C_vertex = [47.642198,-122.140241]
        self.point_D_vertex = [47.642201,-122.140188]

       

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

        parte_entera = int(self.localization_gps.latitude)
        parte_decimal = int((self.localization_gps.latitude - parte_entera) * 1e6)  # Multiplicamos por 1e6 para obtener los 6 dígitos decimales
        # Combinamos la parte entera y la parte decimal sin el último dígito
        self.latitude = parte_entera + parte_decimal / 1e6

        parte_entera2 = int(self.localization_gps.longitude)
        parte_decimal2 = int((self.localization_gps.longitude - parte_entera2) * 1e6)  # Multiplicamos por 1e6 para obtener los 6 dígitos decimales
        # Combinamos la parte entera y la parte decimal sin el último dígito
        self.longitude = parte_entera2 + parte_decimal2 / 1e6
        

        

    def state_cb(self, msg):
        self.current_state = msg

        self.lastTime = time.time()

    def extended_state_cb(self,msg):
        self.extended_state = msg
       

    def lidar_cb(self,cloud_msg):
        z_points = [point[0] for point in sensor_msgs.point_cloud2.read_points(cloud_msg, field_names=("z"), skip_nans=True)]
        self.distance_z = sum(z_points) / len(z_points) 

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
        is_exploration = False
       
        print("Numero random para escoger la accion: " + str(n))
        #--Exploration
        if n < self.epsilon:
            id_action = np.random.choice(len(ACTIONS))
            is_exploration = True
            print("Exploracion: " + str(is_exploration))
            #print("Exploracion,Accion : " + str(id_action))
            return ACTIONS[id_action],id_action
        #--Explotation
        else:
            id_action = np.argmax(self.QTable[state,:])
            print("Exploracion: " + str(is_exploration))
            #print("Explotacion,Accion : " + str(id_action))
            return ACTIONS[id_action],id_action
        

        
    def functionQLearning(self,state,next_state,action,reward):

        #print("Table, State: " + str(state) + " Action: " + str(action) + "Next_State: " + str(next_state))
        #print(self.QTable[state, action])
        #print(np.argmax(self.QTable[next_state, action]))

        self.QTable[state, action] = self.QTable[state, action] + self.alpha * (reward + self.gamma * np.max(self.QTable[next_state]) - self.QTable[state, action])
        #print("QTable[" + str(state) + "," + str(action) + "] : " + str(self.QTable[state, action]) + "reward: " + str(reward))
        
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
        


    def execute_action(self,action,id):

        
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

    def reward_function(self,cx):
       
        
        reward = 0
        error_lane_center = (WIDTH/2 - cx)
        

        MIN_ERROR = 0
        MAX_ERROR = 160
        
        if (self.is_exit_lane(cx)):
            
            reward = -10

        else:

            
            normalise_error = (abs(error_lane_center) - MIN_ERROR) / (MAX_ERROR - MIN_ERROR)
            reward = 1 - normalise_error
            
        return reward
    
    def is_exit_lane(self,cx):

        status = False
        if (cx != -1) and ((is_not_detected_left is False ) or (is_not_detected_right is False)) \
        and (is_not_detect_lane is False):
          
           status = False
       
        else:
           
           status = True


       
        return status
        

    def is_finish_route(self):

        if (self.point_A_vertex[0] <= self.latitude <= self.point_C_vertex[0]) and \
           (self.point_B_vertex[0] <= self.latitude <= self.point_D_vertex[0]) and \
           (self.point_B_vertex[1] >= self.longitude >=self.point_A_vertex[1]) and \
           (self.point_D_vertex[1] >= self.longitude >=self.point_C_vertex[1]):
           
            print("¡Recorrido completado!")
            return True

       
        
        else:
            return False

        
    

    def algorithm(self,perception):
        global n_steps,state,is_not_detected_left,is_not_detected_right,is_first,counter_actions,is_not_detect_lane,is_first_time

        image = perception.cv_image
        _,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)

        
        current_state = self.getState(cx)
        
        is_first = True

        exit_lane = self.is_exit_lane(cx)

        while(not exit_lane and (not self.is_finish_route())):
            #t0 = time.time()
            
            #print("Time: " +str(time.time() - self.lastTime))

            
            action,id_action = qlearning.chooseAction(current_state)
            print("Estado: "+ str(current_state))
            print("Accion: "+str(id_action))
     
            print("Valor actual de Q(S" + str(current_state) + ",A" + str(id_action) + "): " + str(self.QTable[current_state,id_action]))
            

            qlearning.execute_action(action,id_action)
            #counter_actions +=1
            
            time.sleep(0.05)
            #print(cx)

            out_image,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)

            
            print("Error del centro en pixeles: " + str((WIDTH/2 - cx)) + "Centroide: " + str(cx))
            print("is_not_detect_lane: " + str(is_not_detect_lane) + "," + str(is_first_time))
           
          
            """
            
            #-- Comprobación de cuando se ejecuta las primeras acciones
            if(is_not_detect_lane and is_first_time):
                init = time.time()
                rate_aux = rospy.Rate(20)
                
                while (time.time() - init <= 1.2):
                    print("bucle")
                    print(time.time() - init )
                    self.height_velocity_controller()
                    self.local_raw_pub.publish(self.velocity)
                    rate_aux.sleep()
                   
                is_first_time = False
                is_not_detect_lane = False
            """

            

            print("is_not_detect_lane: " + str(is_not_detect_lane) + "," + str(is_first_time))
            #--Comprobación si falla la percepción
            if(cx == -1):
                print("Error por la percepción")
                init = time.time()
                rate_aux = rospy.Rate(30)
                while (time.time() - init <= 0.5):
                    self.height_velocity_controller()
                    self.local_raw_pub.publish(self.velocity)
                    rate_aux.sleep()
                        

                is_not_detected_left = False
                is_not_detected_right = False
                is_not_detect_lane = False
                out_image,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
            
            

            self.client_airsim.simPause(True)
            #print("Pausado el simulador")
            #t1 = time.time()
           

            #t4 = time.time()
            is_first = False    
            reward = qlearning.reward_function(cx)
            print("Recompensa: " + str(reward))
            next_state = qlearning.getState(cx)

           
            qlearning.functionQLearning(current_state,next_state,id_action,reward)
            print("Valor nuevo de Q(S" + str(current_state) + ",A" + str(id_action) + "): " + str(self.QTable[current_state,id_action]))
            print("Valor de maximo de Q en el siguiente estado MaxQ(S" + str(next_state) + "): " + str(np.max(self.QTable[next_state])))
            print("------------------------------------------------------")
            #t5 = time.time()
            self.client_airsim.simPause(False)
            #print("Tiempo de parar el simulador evaluar todo y reanudarlo: " + str(time.time() - t1))
            #print("Reanudado el simulador")
            #print("Current_state: " + str(current_state) + " , Next_state: " + str(next_state))


            current_state = next_state
            #print(current_state > 0 and current_state < 10)
            n_steps += 1

            if (out_image is not None):
                image = out_image
                perception.drawStates(out_image)

                
             
                
                cv2.putText(
                        out_image, 
                        text = "V: " + str(action[0]),
                        org=(0, 15),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA
                )

                
                cv2.putText(
                        out_image, 
                        text = "W: " + str(action[1]),
                        org=(0, 45),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA
                )
                cv2.putText(
                        out_image, 
                        text = "Action: " + str(id_action),
                        org=(0, 85),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA
                )

                cv2.putText(
                        out_image, 
                        text = "State: " + str(current_state),
                        org=(0, 65),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA
                )

                

                cv2.circle(out_image,(cx,280),3,(0,0,0),-1)
                cv2.imshow("image",out_image)
                cv2.waitKey(3)
           
            
            error = (WIDTH/2 - cx) 
            error_angle = 0.0 - angle_orientation
            is_first = True
            exit_lane = self.is_exit_lane(cx)
            cx_prev = cx 
            #centroid = cx
            #t1 = time.time()

        file = "/home/bb6/pepe_ws/src/qlearning/trainings/29-marzo/fotos/foto" + str(n_steps) + ".jpg"
        cv2.imwrite(file ,image)
        #print("FPS train: " + str(1/(t1 - t0)))
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
        extrem_point_line = None 

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    points_cluster = np.vstack((points_cluster,punto*15))
                  
                    coefficients = np.polyfit(points_cluster[:,0],points_cluster[:,1],2)
                    

                    self.list_left_coeff_a.append(coefficients[0])
                    self.list_left_coeff_b.append(coefficients[1])
                    self.list_left_coeff_c.append(coefficients[2])
##
                    a = np.mean(self.list_left_coeff_a[-5:])
                    b = np.mean(self.list_left_coeff_b[-5:])
                    c = np.mean(self.list_left_coeff_c[-5:])

                    mean_coeff = np.array([a,b,c])
                
                    #coefficients_left_global = mean_coeff

                    self.counter_it_left += 1
#
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

            max_y_index = np.argmax(line[:,0])
            extrem_point_line = line[max_y_index]

        return line,extrem_point_line
    

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
        extrem_point_line = None

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    points_cluster = np.vstack((points_cluster,punto))
                    coefficients = np.polyfit(points_cluster[:,0],points_cluster[:,1],2)

                    self.list_right_coeff_a.append(coefficients[0])
                    self.list_right_coeff_b.append(coefficients[1])
                    self.list_right_coeff_c.append(coefficients[2])
##
                    a = np.mean(self.list_right_coeff_a[-5:])
                    b = np.mean(self.list_right_coeff_b[-5:])
                    c = np.mean(self.list_right_coeff_c[-5:])
#
                    mean_coeff = np.array([a,b,c])


                    #coefficients_right_global = mean_coeff

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

            max_y_index = np.argmax(line[:,0])
            extrem_point_line = line[max_y_index]

        return line,extrem_point_line

        
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
        counter_right_points = 0
        counter_left_points = 0

        final_right_clusters = []
        final_left_clusters = []

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
                    left_clusters.append((points_cluster,centroid))
                    #print("Izquierda: " + str(len(points_cluster)))
                    img[points_cluster[:,0], points_cluster[:,1]] = [0,0,255]
                elif centroid[1] > 160 and centroid[1] < 300:  # right lane
                    right_clusters.append((points_cluster, centroid))
                    #print(centroid)
                    cv2.circle(cv_image, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)
                    img[points_cluster[:,0], points_cluster[:,1]] = [0,255,0]
                    #print("Derecha: " + str(len(points_cluster)))
               
            # Now, among the closest clusters, select the one with the highest density
           
           
            if left_clusters:
                left_clusters = [max(left_clusters, key=lambda x: self.score_cluster(x, center))]
           
            
           
            if right_clusters:
                right_clusters = [max(right_clusters, key=lambda x: self.score_cluster(x, center))]

            
            for points_cluster, centroid in left_clusters:
               
                counter_left_points +=len(points_cluster)
                final_left_clusters.append(points_cluster)
                #color = self.colors[cluster % len(self.colors)]
                cv_image[points_cluster[:,0], points_cluster[:,1]] = [0,255,0]
                #cv2.circle(cv_image, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)
            
            for points_cluster, centroid in right_clusters:
               
                counter_right_points +=len(points_cluster)
                final_right_clusters.append(points_cluster)
                
                #color = self.colors[cluster % len(self.colors)]
                cv_image[points_cluster[:,0], points_cluster[:,1]] = [0,0,255]
                #
            
           
            if (counter_left_points == 0  or counter_right_points == 0):
                print("No hay puntos")
                print(counter_left_points,counter_right_points)
                return None,None,-1
            
            else:

                return final_left_clusters,final_right_clusters,0
        
        else:
            print("No hay puntos en la calle")
            print(left_clusters,right_clusters)
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
            global is_not_detect_lane,is_not_detected_left,is_not_detected_right,counter_actions
            
            cx = 0 
            cy = 0
            orientation_angle = 0
            
           
            if(left_clusters and right_clusters):
                left = np.concatenate(left_clusters,axis=0)
                right = np.concatenate(right_clusters,axis=0)

                

                cvimage[left[:,0],left[:,1]] = [0,0,255]
                cvimage[right[:,0],right[:,1]] = [0,255,0]

                
                points_line_right,extrem_point_line_right =  self.calculate_right_regression(right)
                points_line_left,extrem_point_line_left = self.calculate_left_regression(left)


                if (points_line_right is None or points_line_left is None):
                    print("Error en la regresion")
                    cx = -1 
                    cy = -1
                    orientation_angle = -1
                    cvimage = None
                    is_not_detect_lane = True

                
                else:            
                    img_line_left,img_line_right = self.dilate_lines(points_line_left,points_line_right)

                    cvimage[img_line_left == 1] = [255,255,255]
                    cvimage[img_line_right == 1] = [255,255,255]

                    points_beetween_lines = self.interpolate_lines(cvimage,points_line_left,points_line_right)

                    cvimage[points_beetween_lines[:,0],points_beetween_lines[:,1]] = [255,0,0]

                    
                    cx,cy = self.calculate_mass_centre_lane(points_beetween_lines)


                    orientation_angle = self.calculate_angle([cx,cy],[160,cy],cvimage)


                    #print("LAB " + str(extrem_point_line_left[1]) + ", RAB: " + str(extrem_point_line_right))

                    max_y_index = np.argmin(points_line_right[:,0])
                    extrem_point_line2 = points_line_right[max_y_index]
                        
                    max_y_left_index = np.argmin(points_line_left[:,0])
                    extrem_point_left_line2 = points_line_left[max_y_left_index]


                        #--Detect right turn
                    if (extrem_point_line_left[1] > 130 and extrem_point_line_right[1] > WIDTH/2):
                        is_not_detect_lane = True
                        cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

                    #--Detect left turn
                    elif(extrem_point_line_left[1] < WIDTH/2 and extrem_point_line_right[1] < 170):

                        if(extrem_point_line2[1] - extrem_point_left_line2[1] <= 2):
                            is_not_detect_lane = True
                            cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

                        else:
                            is_not_detect_lane = True
                            cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

                
                    else:

                        is_not_detect_lane = False

                       
                        


                #cv2.circle(cvimage, (self.cx,self.cy), radius=10, color=(0, 0, 0),thickness=-1)
               
                #cv2.line(cvimage,(160,320),(self.cx,self.cy),(0,0,0),3)
               

            else:
                print("Error de los puntos numpy")
                print(left_clusters,right_clusters)
                cvimage = None
                cx = -1 
                cy = -1 
                orientation_angle = -1
                is_not_detected_left = True
                is_not_detected_right = True
                

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

def spin():
    rospy.spin() 


def save_files(n_episode, epsilon, steps, acummulateReward):

    with open('/home/bb6/pepe_ws/src/qlearning/trainings/29-marzo/episodes-iterations.csv', mode="a", newline="") as file_steps:
        writer_steps = csv.writer(file_steps)
        writer_steps.writerow([n_episode, steps])

    with open('/home/bb6/pepe_ws/src/qlearning/trainings/29-marzo/episodes-epsilon.csv', mode="a", newline="") as file_epsilon:
        writer_epsilon = csv.writer(file_epsilon)
        writer_epsilon.writerow([n_episode, epsilon])

    with open('/home/bb6/pepe_ws/src/qlearning/trainings/29-marzo/episodes-accumulated-reward.csv', mode="a", newline="") as file_reward:
        writer_reward = csv.writer(file_reward)
        writer_reward.writerow([n_episode, acummulateReward])

    df = pd.DataFrame(qlearning.QTable)
    df.to_csv('/home/bb6/pepe_ws/src/qlearning/trainings/29-marzo/q_table.csv')





    

        
if __name__ == '__main__':
    
    rospy.init_node("RL_node_py")

    thread_spin = threading.Thread(target=spin)
    thread_spin.start()
    
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
    n_episode = rospy.get_param('~n_episode')

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
                    time.sleep(3)
                    state = 1

            if(state == 1):
                _,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
                #angle_error = 0.0 - angle_orientation
                error = (WIDTH/2 - cx)

                perception.correct_initial_position(error,qlearning)
                qlearning.stop()
                if (qlearning.set_mode_client.call(qlearning.set_mode_hold).mode_sent is True):
                        rospy.loginfo("HOLD")
                time.sleep(3)

            if(state == 2):

                if(n_episode < qlearning.MAX_EPISODES):
                    n_episode += 1
                    state = 3
                    time.sleep(4)
                else:
                    state = 5
            
            if(state == 3):

               
             
                t2 = time.time()
                qlearning.algorithm(perception)
                
            if(state == 4):
                
               
                if(not is_landing):
                    #print("Tiempo final: " + str(time.time() - t2))
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
                        is_not_detect_lanes = False
                        is_first_time = True
                        exit = False
                        counter_actions = 0

                        
                       
                        #print("Tiempo por episodio: " + str(time.time() - t_episode))
                        t_counter_ep +=time.time() - t_episode
                        
                        if n_steps > 0:
                            print("ID_EPISODE: " + str(n_episode) + " N_STEPS: " + str(n_steps) + " epsilon: " + str(qlearning.epsilon))

                            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)


                        if n_steps == 0:
                            state = 5
                        if(n_episode < MAX_EXPLORATIONS):
                            qlearning.epsilon = qlearning.epsilon_initial - (n_episode * (qlearning.epsilon_initial / MAX_EXPLORATIONS))
                        else:
                            qlearning.epsilon = 0
                        n_steps = 0
                        qlearning.accumulatedReward = 0
                        state = 0
                        time.sleep(10)
                        

                        if(n_episode ==  qlearning.MAX_EPISODES):
                            state = 5

            if(state == 5):
               
                break

            rate.sleep()

        except rospy.ROSInterruptException:
            print("Interrumpido C el nodo")
            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)
            break

        except rospy.service.ServiceException:
            print("Parado el servicio")
            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)
            break

        except KeyboardInterrupt:
            print("Control C")
            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)
            break
        except OSError:
            print("Se produjo un error inesperado capturado")
            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)
            break
        except RuntimeError:
            print("Se produjo un error runtime")
            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)
            break

       
        except AttributeError:
            print("Se produjo un error attributeError")
            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)
            break
        

        """
        
        except IndexError:
            print("IndexError")
            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)
            break
        """
        


    print("Ha acabado")  
    print("Tiempo de entrenamiento: " + str(time.time() - t_initial))
    print("Media de tiempo por episodio: " + str(t_counter_ep/qlearning.MAX_EPISODES))

   
    
    
    
    
      
   
    
 

