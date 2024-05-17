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
import statistics


IMAGE_TOPIC = '/airsim_node/Drone/front_center_custom/Scene'

ROUTE_MODEL = "/home/bb6/YOLOP/weights/yolop-320-320.onnx"

MIN_VALUE_X = 185
MAX_VALUE_X = 320

HEIGH = 320
WIDTH = 320

X_PER_PIXEL = 3.0 #--Width road 

LIDAR = "/airsim_node/Drone/lidar/LidarCustom"


coefficients_left_global = np.array([])
coefficients_right_global = np.array([])

is_not_detected_left = False
is_not_detected_right = False

is_first = False

cx_global = 0.0
cy_global = 0.0

exit = False

is_not_detect_lane = False
is_first_time = False

size_blue = 0
i = 0

n_steps = 0

state = 0

counter_photo = 0


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

t1 = 0.0
t2 = 0.0


ACTIONS = []
speeds_actions = np.linspace(0.1,2.0,11, dtype=float)
angular_speeds = np.linspace(-25,25, 21)

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
    
        #self.QTable = np.zeros((len(STATES)+1,len(ACTIONS)))
        #self.CounterActionsMatrix = np.zeros((len(STATES)+1,len(ACTIONS)))
        self.CounterActionsMatrix = np.genfromtxt('/home/bb6/pepe_ws/src/qlearning/trainings/airsim/17-mayo/counter_actions_matrix.csv', delimiter=',')
        self.QTable = np.genfromtxt('/home/bb6/pepe_ws/src/qlearning/trainings/airsim/17-mayo/q_table.csv', delimiter=',',skip_header=1,usecols=range(1,22))
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

        self.FIRST_LOCALIZATION = 1
        self.SECOND_LOCALIZATION = 2

        self.client_airsim = airsim.MultirotorClient(ip="192.168.2.16")
        self.client_airsim.confirmConnection()
        self.client_airsim.enableApiControl(True)

        self.vz = 0

       
        self.localization_gps_sub = rospy.Subscriber("/airsim_node/Drone/global_gps",NavSatFix,callback=self.localization_gps_cb)
        #--Points to road final-- antiguos
        #self.point_A_vertex = [47.642184,-122.140238]
        #self.point_B_vertex = [47.642187,-122.140186]
        #self.point_C_vertex = [47.642198,-122.140241]
        #self.point_D_vertex = [47.642201,-122.140188]

        #--Puntos nuevos del final de la calle
        #self.point_A_vertex = [47.642711,-122.140800]
        #self.point_B_vertex = [47.642704,-122.140762]
        #self.point_C_vertex = [47.642726,-122.140785]
        #self.point_D_vertex = [47.642715,-122.140747]

        #--Punto del centro del circulo para comprobar el final del recorrido
        self.point_P = [47.642715,-122.140777]




        self.lidar_sub = rospy.Subscriber(LIDAR,PointCloud2,callback=self.lidar_cb)
        self.distance_z = 0.0

        self.last_req = 0.0
        self.pepe = False

        self.KP_v = 0.5
        self.KD_v = 1.2

        self.prev_error_height = 0
        self.error = 0

        self.lastTime = 0.0

    def lidar_cb(self,cloud_msg):
        z_points = [point[0] for point in sensor_msgs.point_cloud2.read_points(cloud_msg, field_names=("z"), skip_nans=True)]
        self.distance_z = sum(z_points) / len(z_points) 
        #print("Distancia: " + str(self.distance_z))

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
            print("Exploracion,Accion : " + str(id_action))
            return ACTIONS[id_action],id_action
        #--Explotation
        else:
            id_action = np.argmax(self.QTable[state,:])
            print("Exploracion: " + str(is_exploration))
            print("Explotacion,Accion : " + str(id_action))
            return ACTIONS[id_action],id_action

        
        

        
    def functionQLearning(self,state,next_state,action,reward):

        #print("Table, State: " + str(state) + " Action: " + str(action) + "Next_State: " + str(next_state))
        #print(self.QTable[state, action])
        #print(np.argmax(self.QTable[next_state, action]))

        self.QTable[state, action] = self.QTable[state, action] + self.alpha * (reward + self.gamma * np.max(self.QTable[next_state]) - self.QTable[state, action])
        #print("QTable[" + str(state) + "," + str(action) + "] : " + str(self.QTable[state, action]) + ",reward: " + str(reward) + ",Maximo valor del siguiente estado: " + 
            #str(np.max(self.QTable[next_state])) + ",Siguiente estado: " + str(next_state))
        
        self.accumulatedReward += reward
        #print(self.accumulatedReward,reward)

                
    def reset_position(self):

        number_position = random.randint(1,3)
        x = 0
        y = 0
        z = 0

        pitch = 0.0
        yaw = 0.0
        roll = 0.0

        self.client_airsim.moveByVelocityBodyFrameAsync(
            vx = 0,
            vy = 0,
            vz = self.vz,
            duration = 2.0,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode = airsim.YawMode(is_rate =True, yaw_or_rate=0)
            ).join()
        

    
        if(number_position == self.FIRST_LOCALIZATION):
            x = 18.658248901367188
            y = 11.574799537658691
            z = -0.276497483253479

            yaw = 0.4893928006316498
    

        elif(number_position == 2):
            x = 47.886112213134766
            y = 21.24800682067871
            z = -0.2762098014354706

            yaw =  0.07393334743888746

        else:
            x = 86.50939178466797
            y = -7.812751293182373
            z =  -0.2759387493133545

            yaw =  -1.3658072306973006

        
        

        position = airsim.Vector3r(x,y,z)
        orientation = airsim.to_quaternion(pitch,roll,yaw)
        pose = airsim.Pose(position,orientation)

        self.client_airsim.simSetVehiclePose(pose, True,"Drone")
        self.client_airsim.enableApiControl(True)
        self.client_airsim.armDisarm(False)
       
        print("Reinicie")



    def height_velocity_controller(self):
        
        self.error = round((2.80 - self.distance_z),2)
        derr = self.error - self.prev_error_height

        self.vz = -((self.KP_v * self.error) + (self.KD_v *derr))
        


    def execute_action(self,action,id):

        self.height_velocity_controller()
        
    
        self.client_airsim.moveByVelocityBodyFrameAsync(
            vx = action[0],
            vy = 0,
            vz = self.vz,
            duration = 0.1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode = airsim.YawMode(is_rate =True, yaw_or_rate=action[1])
            ).join()
        self.prev_error_height = self.error


    def reward_function(self,cx,angle):

        reward = 0
        target_heading = 0
        error_lane_center = (WIDTH/2 - cx)
        heading_difference = (target_heading - angle) 
        
        MIN_ERROR = 0
        MAX_ERROR = 80

        MIN_ANGLE = 0
        MAX_ANGLE = 70

        CENTRE_WEIGHT = 0.85
        ANGLE_WEIGHT = 0.15
        
        if (self.is_exit_lane(cx)):
            
            reward = -10

        else:

            
            normalise_error_centre = (abs(error_lane_center) - MIN_ERROR) / (MAX_ERROR - MIN_ERROR)
            reward_centre = 1 - normalise_error_centre

            
            normalise_error_angle = (abs(heading_difference) - MIN_ANGLE) / (MAX_ANGLE - MIN_ANGLE)
            reward_angle = 1 - normalise_error_angle


            reward = (reward_centre * CENTRE_WEIGHT) + (reward_angle * ANGLE_WEIGHT)
            
        return reward
    
    def is_exit_lane(self,cx):
        global is_not_detect_lane
        status = False
        if (cx != -1) and ((is_not_detected_left is False ) or (is_not_detected_left is False)) and (is_not_detect_lane is False):
          
           status = False
       
        else:
           print("Me sali")
           print(cx,is_not_detected_left,is_not_detected_left,is_not_detect_lane)
           status = True


       
        return status
        

    def is_finish_route(self):


        #--Condicion de si esta al final de la calle con un rectangulo
        """
        
        if (self.point_A_vertex[0] <= self.latitude <= self.point_C_vertex[0]) and \
           (self.point_B_vertex[0] <= self.latitude <= self.point_D_vertex[0]) and \
           (self.point_B_vertex[1] >= self.longitude >=self.point_A_vertex[1]) and \
           (self.point_D_vertex[1] >= self.longitude >=self.point_C_vertex[1]):
           
            print("¡Recorrido completado!")
            return True

         else:
            return False
        """
        #----------------------------------------------------------------

        #-- Para comprobar que estamos en el final del recorrido definiremos un circulo con radio 3,
        #--calcularemos con la distancia de Pitagoras si se encuentra dentro del circulo del final del recorrido

        distance_latitude = (self.latitude - self.point_P[0])**2
       
        distance_longitude = (self.longitude - self.point_P[1])**2
        
        distance = float(format(math.sqrt(distance_latitude + distance_longitude),'.15f'))
        
        if(distance <= 0.0000158):
            print("¡Recorrido completado!")
            return True

        else:
            return False
            

        
    def update_counter_actions(self,state,action):

        #print(state,action)
        self.CounterActionsMatrix[state,action] += 1
        print(state,action)
        print(self.CounterActionsMatrix[state,action])

    

    def algorithm(self,perception):
        global n_steps,state,is_not_detected_left,is_not_detected_right,is_first,counter_actions,is_not_detect_lane,size_blue,t1,t2,is_first_time,counter_photo,i

        pepe = True
        image = perception.cv_image
        
       
        out,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
        #perception.drawStates(out)
        #cv2.circle(out,(cx,280),3,(0,0,0),-1)
        #cv2.imshow("image-antes-percepcion",out)
        #cv2.waitKey(3)

        
        current_state = self.getState(cx)
        #print("Cx: " + str(cx) + ", State: " + str(current_state))
        is_first = True
        exit_lane = self.is_exit_lane(cx)

        counter_fail_percepcion = 0
        counter_fail_exit_lane = 0

        t1 = time.time()
        
       
      
        while(not exit_lane and (not self.is_finish_route())):
          
            
            action,id_action = qlearning.chooseAction(current_state)
            print("Estado: "+ str(current_state))
            print("Accion: "+str(id_action))
     
            print("Valor actual de Q(S" + str(current_state) + ",A" + str(id_action) + "): " + str(self.QTable[current_state,id_action]))
            

            self.client_airsim.simPause(False)

            te = time.time()
            qlearning.execute_action(action,id_action)
            #time.sleep(0.1)
            #print("Tiempo: " + str(time.time() - te))
            self.client_airsim.simPause(True)
          
           
            #t0 = time.time()
            out_image,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
            #print("Cx: " + str(cx))

           #--Percepcion fallida
            if((cx == -1) or (is_not_detected_left is True) or(is_not_detected_right is True)):
                print("Fallo en la percepcion")
                
                print("Centroide: " + str(cx))
                counter_fail_percepcion += 1
                print("Iteraciones consecutivas: " + str(counter_fail_percepcion))
                if(counter_fail_percepcion < 5):
                    continue

            else:
                
                if(counter_fail_percepcion == 0):
                   counter_fail_percepcion = 0

                else:
                
                    counter_fail_percepcion -= 1 

            #--Salida del carril
            if(is_not_detect_lane is True):
                print("Me he salido del carril")
                print("Centroide: " + str(cx))
                counter_fail_exit_lane += 1
                print("Iteraciones consecutivas de la salida del carril: " + str(counter_fail_exit_lane))
                if(counter_fail_exit_lane < 3):
                    continue

            else:
                #print("Iteraciones consecutivas de la salida del carril: " + str(counter_fail_exit_lane))
                #print("Error del centro en pixeles: " + str((WIDTH/2 - cx)) + "Centroide: " + str(cx))
                if(counter_fail_exit_lane == 0):
                   counter_fail_exit_lane = 0

                else:
                
                    counter_fail_exit_lane -= 1 

            
            
          
            reward = qlearning.reward_function(cx,angle_orientation)

            #--Depuracion
            qlearning.update_counter_actions(current_state,id_action)

            print("Recompensa: " + str(reward))
            next_state = qlearning.getState(cx)

           
            qlearning.functionQLearning(current_state,next_state,id_action,reward)
            print("Valor nuevo de Q(S" + str(current_state) + ",A" + str(id_action) + "): " + str(self.QTable[current_state,id_action]))
          
            print("------------------------------------------------------")
            
           


            current_state = next_state
            #print(current_state > 0 and current_state < 10)
            n_steps += 1

            if (out_image is not None):
                image = out_image
                

                
             
                if((cx != -1 )and(is_not_detected_left is False)and(is_not_detected_right is False)):
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

                    cv2.putText(
                            out_image, 
                            text = "Size_blue: " + str(size_blue),
                            org=(0, 105),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.6,
                            color=(255, 255, 255),
                            thickness=2,
                            lineType=cv2.LINE_AA
                    )

                

                    cv2.circle(out_image,(cx,280),3,(0,0,0),-1)
                cv2.imshow("image",out_image)
                cv2.waitKey(3)

            exit_lane = self.is_exit_lane(cx)

            #centroid = cx
            #t3 = time.time()
            t2 = time.time()

            
            if((t2 - t1 >= 2.5) and (pepe is True)):
                is_first_time = True
                pepe = False

            #print("Pausado")
            #input()
            #print("Reanudo")
          
            
        self.client_airsim.simPause(False)
        file = "/home/bb6/pepe_ws/src/qlearning/trainings/airsim/17-mayo/fotos/foto-episodio" + str(counter_photo) + ".jpg"
        cv2.imwrite(file ,image)
        #print("Rate train: " + str(int(1/(t3 - t0))))
            #print("Centroide: " + str(cx))
            #print("Error center : " + str(error) + " ,Error angle: " + str(error_angle))
            
        
        
        state = 3



class Perception():
    def __init__(self):
        self.bridge = CvBridge()
        self.tiempo_ultimo_callback = time.time()
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback_img,queue_size=1)
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
    
    def calculate_radius_curvature(self,a, b, x):
        """
        Calculate the radius of curvature at a given x in a quadratic function.

        Args:
            a (float): The second degree coefficient of the quadratic function.
            b (float): The first degree coefficient of the quadratic function.
            x (float): The x value at which to calculate the radius of curvature.

        Returns:
            R (float): The radius of curvature.
        """
        # First derivative
        y_prime = 2*a*x + b

        # Second derivative
        y_double_prime = 2*a
        
    

        # Radius of curvature
        R = (1 + y_prime**2)**1.5 / np.abs(y_double_prime)

        return R
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
        a = 0
        b = 0
        c = 0

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

            #x_centroid = (-mean_coeff[1]/(2*mean_coeff[0])).astype(int)
#
            #y = (mean_coeff[0]*x_centroid)**2 + mean_coeff[1] * x_centroid + mean_coeff[2]

            print("Coeficiente A: "  + str(mean_coeff[0]))

            ##center_5_x_values = line[(len(line) // 2) - 2 : (len(line) // 2) + 3, 1]
            #print("Radio en la regresion derecha: " + str(self.calculate_radius_curvature(a, b, x_centroid)))

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
        center = np.array([220,160])
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
                    #print("Izquierda: " + str(centroid[1]))
                    img[points_cluster[:,0], points_cluster[:,1]] = [0,0,255]
                elif centroid[1] >= 160:  # right lane
                    right_clusters.append((points_cluster, centroid))
                    #print(centroid)
                    cv2.circle(cv_image, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)
                    img[points_cluster[:,0], points_cluster[:,1]] = [0,255,0]
                    #print("Derecha: " + str(len(points_cluster)))
               
            # Now, among the closest clusters, select the one with the highest density
           
           
            if len(left_clusters) != 0:
                left_clusters = [max(left_clusters, key=lambda x: self.score_cluster(x, center))]
           
            
           
            if len(right_clusters) != 0:
                right_clusters = [max(right_clusters, key=lambda x: self.score_cluster(x, center))]

            
            for points_cluster, centroid in left_clusters:
               
                counter_left_points +=len(points_cluster)
                final_left_clusters.append(points_cluster)
                #color = self.colors[cluster % len(self.colors)]
                #cv_image[points_cluster[:,0], points_cluster[:,1]] = [0,255,0]
                #cv2.circle(cv_image, (centroid[1], centroid[0]), 5, [0, 0, 0], -1)
            
            for points_cluster, centroid in right_clusters:
               
                counter_right_points +=len(points_cluster)
                final_right_clusters.append(points_cluster)
                
                #color = self.colors[cluster % len(self.colors)]
                #cv_image[points_cluster[:,0], points_cluster[:,1]] = [0,0,255]
                #
            
           
            if (counter_left_points == 0  or counter_right_points == 0):
                #print("No hay puntos")
                #print(counter_left_points,counter_right_points)
                return None,None,-1,img
            
            else:

                return final_left_clusters,final_right_clusters,0,img
        
        else:
            #print("No hay puntos en la calle")
            #print(left_clusters,right_clusters)
            return None,None,-1,img

    def draw_region(self,img):


        mask = np.zeros_like(img) 

        cv2.fillPoly(mask, self.vertices,(255,255,255))
        masked_image = cv2.bitwise_and(img, mask)

        return masked_image
    
    
    
    def calculate_mass_centre_lane(self,points_lane):

        if (len(points_lane) > 0): 
            # Supongamos que todos los puntos tienen la misma masa
            m_i = 1

            # Calcula la masa total
            m_total = m_i * len(points_lane)

            # Calcula la suma de las posiciones de los puntos ponderadas por su masa
            r_i_sum = np.sum(points_lane * m_i, axis=0)

            # Calcula la posición del centro de masas
            r_CM = r_i_sum / m_total

            return int(r_CM[1]),int(r_CM[0])

        else:
            return -1,-1


            
      

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
            global is_not_detect_lane,is_not_detected_left,is_not_detected_right,size_blue,is_first_time
            
            cx = 0 
            cy = 0
            orientation_angle = 0
            
           
            if(left_clusters and right_clusters):
                left = np.concatenate(left_clusters,axis=0)
                right = np.concatenate(right_clusters,axis=0)

                

                cvimage[left[:,0],left[:,1]] = [0,0,255]
                cvimage[right[:,0],right[:,1]] = [0,255,0]

                
                points_line_right,extrem_point_line_right=  self.calculate_right_regression(right)
                points_line_left,extrem_point_line_left = self.calculate_left_regression(left)


                if (points_line_right is None or points_line_left is None):
                    #print("Error en la regresion")
                    #print(points_line_right,points_line_left)
                    #cx = -1 
                    cy = -1
                    orientation_angle = -1
                    
                   

                
                else:            
                    img_line_left,img_line_right = self.dilate_lines(points_line_left,points_line_right)

                    cvimage[img_line_left == 1] = [255,255,255]
                    cvimage[img_line_right == 1] = [255,255,255]

                    points_beetween_lines = self.interpolate_lines(cvimage,points_line_left,points_line_right)

                    size_blue = len(points_beetween_lines)

                    cvimage[points_beetween_lines[:,0],points_beetween_lines[:,1]] = [255,0,0]

                    
                    cx,cy = self.calculate_mass_centre_lane(points_beetween_lines)


                    orientation_angle = self.calculate_angle([cx,cy],[160,cy],cvimage)


                    #print("LAB " + str(extrem_point_line_left[1]) + ", RAB: " + str(extrem_point_line_right))

                    max_y_index = np.argmin(points_line_right[:,0])
                    extrem_point_line2 = points_line_right[max_y_index]
                        
                    max_y_left_index = np.argmin(points_line_left[:,0])
                    extrem_point_left_line2 = points_line_left[max_y_left_index]


                    if(is_first_time):
                        #--Detect right turn
                        if (extrem_point_line_left[1] > 150 and extrem_point_line_right[1] > WIDTH/2):
                            #print("Detenccion de salida por la derecha")
                            #print(extrem_point_line_left[1],extrem_point_line_right[1])
                            is_not_detect_lane = True
                            cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

                        #--Detect left turn
                        elif(extrem_point_line_left[1] < WIDTH/2 and extrem_point_line_right[1] < 170):
                            
                            if(extrem_point_line2[1] - extrem_point_left_line2[1] <= 5):
                                #print("Detenccion de salida por la izquierda con distancia de extremos")
                                is_not_detect_lane = True
                                cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

                            else:
                                #print("Detenccion de salida por la izquierda")
                                is_not_detect_lane = True
                                cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)

                    
                        else:
                            #print("Distancia maxima: " + str(extrem_point_line2[1] - extrem_point_left_line2[1]))
                            
                            
                            
                            #print("Paso los 2 segundos")
                            if(len(points_beetween_lines) >= 34800):
                                #print("El tamaño es muy grande de la zona azul")
                                cv2.circle(cvimage, (10,50), radius=10, color=(0, 0, 255),thickness=-1)
                                is_not_detect_lane = True

                            else:
                                is_not_detect_lane = False

                    
                            

                        
                        
                       
                        


                #cv2.circle(cvimage, (self.cx,self.cy), radius=10, color=(0, 0, 0),thickness=-1)
               
                #cv2.line(cvimage,(160,320),(self.cx,self.cy),(0,0,0),3)
               

            else:
                #print("Error de los puntos numpy")
                #print(left_clusters,right_clusters)
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
        is_not_detected_left = False
        is_not_detected_right = False
        images_yolop = self.infer_yolop(cv_image)
        #print("Tiempo YOLOP: " + str(time.time() - self.t1))
        mask_cvimage = self.draw_region(cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))
        mask = self.draw_region(images_yolop[1])
        
        t_clustering = time.time()
        left_clusters,right_clusters,state_clusters,img = self.clustering(mask,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR))
       
        #print("Clustering solo : " + str(time.time() - t_clustering))
        #print("Clustering + YOLOP : " + str(time.time() - self.t1))

        if state_clusters == -1:
            
            #print("Error en clustering")
            cx = -1
            cy = -1 
            orientation_angle = -1 
            #is_not_detected_left = True
            #is_not_detected_right = True
            out_img = img

        
        else:
           
            t_regresion = time.time()
            out_img,cx,cy,orientation_angle = self.calculate_margins_points(left_clusters,right_clusters,cv2.resize(cv_image,(WIDTH,HEIGH),cv2.INTER_LINEAR),images_yolop[0])
            

            #print("Regresion + interpolate + centroid solo: " + str(time.time() - t_regresion))
            #print("Clustering + YOLOP + Regresion + interpolate + centroid solo: " + str((time.time() - self.t1)))
        return out_img,cx,cy,orientation_angle
        
        
    def callback_img(self, data):
        
        
        self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

            

    def correct_initial_position(self,error,qlearning):
        global state

        while(abs(error) > 1):
            qlearning.height_velocity_controller()
            print(qlearning.vz)
            qlearning.client_airsim.moveByVelocityBodyFrameAsync(
            vx = 0,
            vy = 0,
            vz = qlearning.vz,
            duration = 0.080,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode = airsim.YawMode(is_rate =True, yaw_or_rate= ((0.009 * error)))
            ).join()
           
            
            _,cx,cy,angle_orientation = perception.calculate_lane(perception.cv_image)
            error  = WIDTH/2 - cx

        qlearning.vz = 0
        state = 2

def spin():
    rospy.spin() 


def save_files(n_episode, epsilon, steps, acummulateReward,QTable,CounterActionsMatrix):

    
    with open('/home/bb6/pepe_ws/src/qlearning/trainings/airsim/17-mayo/episodes-iterations.csv', mode="a", newline="") as file_steps:
        writer_steps = csv.writer(file_steps)
        writer_steps.writerow([n_episode, steps])

    with open('/home/bb6/pepe_ws/src/qlearning/trainings/airsim/17-mayo/episodes-epsilon.csv', mode="a", newline="") as file_epsilon:
        writer_epsilon = csv.writer(file_epsilon)
        writer_epsilon.writerow([n_episode,epsilon])

    with open('/home/bb6/pepe_ws/src/qlearning/trainings/airsim/17-mayo/episodes-accumulated-reward.csv', mode="a", newline="") as file_reward:
        writer_reward = csv.writer(file_reward)
        writer_reward.writerow([n_episode, round(acummulateReward,2)])

    
    df = pd.DataFrame(QTable)
    df.to_csv('/home/bb6/pepe_ws/src/qlearning/trainings/airsim/17-mayo/q_table.csv')

    np.savetxt('/home/bb6/pepe_ws/src/qlearning/trainings/airsim/17-mayo/counter_actions_matrix.csv',CounterActionsMatrix,fmt="%d",delimiter=",")





    

        
if __name__ == '__main__':
    
    rospy.init_node("RL_node_py")

    #thread_spin = threading.Thread(target=spin)
    #thread_spin.start()
    
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
                qlearning.client_airsim.armDisarm(True)
                qlearning.client_airsim.moveToZAsync(-2.3, 1).join()
                time.sleep(2)
                state = 1

            

            if(state == 1):

                if(n_episode < qlearning.MAX_EPISODES):
                    n_episode += 1
                    counter_photo = n_episode
                    state = 2
                else:
                    state = 5
            
            if(state == 2):

               
             
                time.sleep(3)
                qlearning.algorithm(perception)
                
            if(state == 3):
                
               
                
                cv2.destroyAllWindows()
               
                qlearning.reset_position()
                perception.cv_image = None
                
                is_not_detected_left = False
                is_not_detected_right = False
                is_not_detect_lane = False
                is_first_time = False
                exit = False
                counter_actions = 0
                perception.list_left_coeff_a.clear()
                perception.list_left_coeff_b.clear()
                perception.list_left_coeff_c.clear()

                perception.list_right_coeff_a.clear()
                perception.list_right_coeff_b.clear()
                perception.list_right_coeff_c.clear()

                perception.counter_it_right = 0
                perception.counter_it_left = 0


                #print("ID_EPISODE: " + str(n_episode) + " N_STEPS: " + str(n_steps) + " epsilon: " + str(qlearning.epsilon) + "accumulatedReward: " + str(qlearning.accumulatedReward))  
                       
               
               
                t_counter_ep +=time.time() - t_episode
                
                if n_steps > 0:
                    print("ID_EPISODE: " + str(n_episode) + " N_STEPS: " + str(n_steps) + " epsilon: " + str(qlearning.epsilon) + "accumulatedReward: " + str(qlearning.accumulatedReward))

                    save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward,qlearning.QTable,qlearning.CounterActionsMatrix)

                else:
                    state = 5

                if(n_episode < MAX_EXPLORATIONS):
                    qlearning.epsilon = qlearning.epsilon_initial - (n_episode * (qlearning.epsilon_initial / MAX_EXPLORATIONS))
                else:
                    qlearning.epsilon = 0
                n_steps = 0
                qlearning.accumulatedReward = 0
               
                state = 0
                time.sleep(3)
                

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

        """
        
        except IndexError:
            print("IndexError")
            save_files(n_episode,qlearning.epsilon ,n_steps,qlearning.accumulatedReward)
            break
        """
        


    print("Ha acabado")  
    print("Tiempo de entrenamiento: " + str(time.time() - t_initial))
    print("Media de tiempo por episodio: " + str(t_counter_ep/qlearning.MAX_EPISODES))

    
    
    
      
   
    
 

