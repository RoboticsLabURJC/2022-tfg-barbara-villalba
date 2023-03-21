---
title: "Month 1-2.Teleop Drone"
last_modified_at: 2023-03-20T13:05:00
categories:
  - Blog
tags:
  - ROS Noetic
  - PX4
  - Mavros
  - Mavlink
  - Gazebo
  - openCV
  - PyQt5
---

These months developed simple teleop for iris drone.  

## Month 1 
In the first few months it investigates how to load the sdf iris model into the PX4 package.

To load the iris model in the launch 'mavros_posix_sitl.launch' it was enough with change sdf field for model iris: 

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/mavros_posix_sitl.launch.png" alt="mavros_posix_sitl" width="500"/>
</p>

But I found the problem that PX4 was not able to load the model well and could not find the exact folder where it was located.
During a week of research and reading the PX4 forum, I tried to load the model by passing it as a parameter to launch itself, that is, passing the destination path to launch so that PX4 could find it:

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/load_vehicle.launch.png" alt="load_vehicle" width="500"/>
</p>

This was a possible solution to be able to load the desired model.

From here I already developed the teleoperator for the drone that will consist of two nodes:

- The interface where we can see the image of the drone's camera and command different behaviors for the movement of the drone

- The teleoperator that will consist of reading the data that we are receiving from the interface and processing it correctly.

The first approximation was to do everything in the same node but having to launch it with PX4 and using the Pyqt5 library would have conflicts, which led me to separate it into two nodes.

## Month 2

In the second month I focused on developing the previously mentioned node and develop a launch that launches the iris model with the world and both nodes. 

First of all we will talk about the interface node and after the teleoperate node. 

### Interface node 

To develop the interface for us to command the drone we will use the Pyqt5 library.

The node consists of two classes, one for the image that we are going to process from the camera and another class for the interface with the different buttons and sliders.

#### Camara Image 
To process the image we will subscribe to the topic '/iris/usb_cam/image_raw'. In order to use the image with the Pyqt5 library, we first need to convert the topic to openCV.
After having it in opencv format, we must transform it into QT format, which will be an image with Pixmap format.

The camera image also shows the current FPS. This will be done simply by measuring the time elapsed between one image and another and calculating the frequency of the image (it is the inverse of the period). Said calculation will make an average with all the measurements and we will update the FPS every second: 

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/ImageCapture.png" alt="ImageCapture" width="500"/>
</p>

Finally with opencv we can use a function called putText to average the FPS.

### Controls Fuctions
For the drone commands we will have different buttons and sliders to be able to do it:

- The buttons are for LAND, TAKE_OFF,POSITION AND VELOCITY. When these buttons are pressed, a String message will be sent to the topic '/commands/mode'.
POSITION AND VELOCITY are modes in how we want to command the drone, whether in position control or speed control.
LAND and TAKEOFF are modes to land and take off the drone.

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/BottomsInterface.png" alt="BottomsInterface" width="500"/>
</p>

- The sliders are to send position, speed and orientation to the drone depending on which control the user has chosen. And the topics '/commands/control_position' and '/commands/control_velocity' will be used, filling the messages with type PoseStamped and Twist

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/Sliders.png" alt="Sliders" width="500"/>
</p>

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/AddSliders.png" alt="AddSliders" width="500"/>
</p>


### Teleop node 

The teleoperated node will consist of subscribing to the interface node issues, processing them (saying if we are in position and speed control, or taking off or landing) and commanding the drone.

I will use the OFFBOARD flight mode, which consists of a mode to control the movement and attitude of the vehicle, establishing the position, the speed. acceleration...etc In order for the commands for the drone to work, we have to make sure that the vehicle is armed and for this we will use a service to be able to say that the vehicle is ready. Without the armed vehicle, the drone would not be able to fly.

For each control I will subscribe to the topics of the interface node to obtain the messages and later to be able to process them.

In order to control the modes in which we find ourselves, we will use some simple checks:

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/Modes.png" alt="Modes" width="500"/>
</p>

#### Control Position
The position control will consist of commanding the drone positions in the gazebo world and it will go towards them, in addition we will also command turns in the z axis in radians from 0 to pi.
This control will only work only when the drone has taken off.

To make the turns in the z axis we have had to subscribe to the topic '/mavros/local_position/pose' to obtain the local position of the drone and obtain what orientations it has since when we publish the angle we are working in quaternions not in angles which we will use Two methods of the tfs euler_from_quaternion and quaternion_from_euler to be able to transform the euler angles (in this case we are interested in the z axis) into quaternions 

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/ControlPosition.png" alt="ControlPosition" width="500"/>
</p>

#### Control Velocity
The position control will consist of commanding the drone speeds in the gazebo world and it will go towards them, in addition we will also command angular speed in the z axis in radians/seconds from 0 to pi.
This control will only work only when the drone has taken off.

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/ControlVelocity.png" alt="ControlVelocity" width="500"/>
</p>

#### Mode Land
In order to land the drone we will use a service called "/mavros/cmd/land" and when the user presses the LAND button the drone will land where it is 

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/LAND.png" alt="LAND" width="500"/>
</p>

#### Mode Take Off
In order to take off the drone and make it fly, we send the height on the z axis by position.

### Results 

<p align="center">
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/Teleop-Interface.png" alt="Teleop" width="500"/>
</p>

### Conclusions 
It is a very simple teleoperator and a contact to know the controls that we can command a drone. Both the interface and the teleoperator could be improved by having other types of behaviors such as commanding speeds, angular speeds, turns and positions in the 3 axes.






