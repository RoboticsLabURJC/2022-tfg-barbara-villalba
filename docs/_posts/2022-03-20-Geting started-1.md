---
title: "Week 1-10. Getting started"
last_modified_at: 2023-03-20T19:43:00
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

This weeks were to create the workspace, install all needs and start developing a simple teleoperate for drone.

## Weeks 1-8 
In the first two months I investigated how to install mavros and mavlink on my personal computer.

Previously I had ROS installed, the non-etic distribution and gazebo, which I did not have to install.

I found problems to be able to work with PX4 on my computer. The problem was that I had ROS and ROS2 installed, which had a conflict between both versions.
Which I opted for a simple solution and was to create a virtual machine with a Linux 22.04 operating system and install ROS no etic, Gazebo, mavros, mavlink and I was able to launch the PX4 package without any problem

## Week 8-10

The next weeks, I started developing a simple teleoperate for Iris drone. I tried some libraries (opencv and pyqt5). Finalliced I chooice Pyqt5 because I worked with hem and it was simple. 

The teleoperator will consist of commanding positions and speeds to the iris drone through some sliders, we will also command orientations and angular speeds.

It is a first contact with a drone since my TFG will try to focus on reinforcement learning through a drone. 

