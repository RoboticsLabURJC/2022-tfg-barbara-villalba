# mavros/mavlink and PX4 Installation Guide



# Introduction
In this page, it's show installation guide for Ubuntu 20.04.  

The following steps are shown in the installation guide

# 1. Install ROS noetic and Gazebo 11
In my case, I'm working with the distribution of ROS noetic (you can install any version to suit your needs). 

First, in order to install ROS noetic we will go to the oficial ROS website: http://wiki.ros.org/noetic/Installation/Ubuntu


Once ROS noetic has been installed, we will install Gazebo 11. In order to install Gazebo 11 we will follow the following commands: 

   ~~~
   sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
   ~~~

If you want to see the installed version of gazebo: 

~~~
gazebo --version
~~~

# 2. Install Mavros and Mavlink 

## 2.1 Install mavros according to the ROS distribution you have, in my case i will do it for ROS noetic

~~~
sudo apt-get install ros-noetic-mavros ros-noetic-mavros-extras
~~~

## 2.2 Then install GeographicLib datasets by running the install_geographiclib_datasets.sh script

~~~
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh \ 
./install_geographiclib_datasets.sh
~~~

## 2.3 Install common dependencies 

~~~
sudo apt-get update -y
sudo apt-get install git zip qtcreator cmake \
    build-essential genromfs ninja-build exiftool \
    python3-pip python3-dev python-is-python3 -y
~~~

## 2.4 Install mavlink

~~~
rosinstall_generator --rosdistro noetic mavlink | tee /tmp/mavros.rosinstall
~~~

# 3. Install PX4 

For install PX4, we will go to the official PX4 repository on github(https://github.com/PX4/PX4-Autopilot) and download the version v1.11.3: 

~~~
mkdir ~/PX4
cd PX4
git clone --recursive https://github.com/PX4/PX4-Autopilot.git -b v1.11.3
~~~

## 3.1 Run PX4 installation script

~~~
cd ~/PX4/PX4-Autopilot/Tools/setup/
bash ubuntu.sh --no-nuttx --no-sim-tools
~~~

## 3.2 Install gstreamer

~~~
sudo apt install libgstreamer1.0-dev
sudo apt install gstreamer1.0-plugins-bad
~~~

## 3.3 Build PX4 

~~~
cd ~/repos/PX4-Autopilot
DONT_RUN=1 make px4_sitl gazebo
~~~

In my case, as I had ROS and ROS2 on the same system. I had problems installing PX4. The provisonal solution was to create a virtual machine exclusively with ROS as i had o version conflict. 

## 3.4 Export environment variables

~~~
echo 'export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:~/PX4/PX4-Autopilot/build/px4_sitl_default/build_gazebo' >> ~/.bashrc
echo 'export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/PX4/PX4-Autopilot/Tools/sitl_gazebo/models' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/PX4/PX4-Autopilot/build/px4_sitl_default/build_gazebo' >> ~/.bashrc    
echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4/PX4-Autopilot' >> ~/.bashrc
echo 'export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4/PX4-Autopilot/Tools/sitl_gazebo' >> ~/.bashrc
    
source ~/.bashrc
~~~

# 4. If you would like to try PX4

~~~
roslaunch px4 mavros_posix_sitl.launch
~~~

<p align="center"> 
<img src="https://github.com/RoboticsLabURJC/2022-tfg-barbara-villalba/blob/main/docs/images/Try-PX4.png" width="80%" height="80%"> 
</p align> 







