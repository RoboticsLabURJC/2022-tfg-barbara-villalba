---
title: "Perception"
last_modified_at: 2023-03-20T13:05:00
categories:
  - Blog
tags:
  - DBSCAN
  - Pytorch
  - Onnx 
---

## Introduction
In this post, we will talking about Perception.

## Perception
In the previous post we talked about YOLOP and which model to choose to get the best result. From this neural network, we will keep the detected lines of the lanes of the road and we will perform an unsupervised learning algorithm called clustering (DBSCAN) to choose the group of lines that we are interested in the lane to follow, for more details of this algorithm you can visit the following page [DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan). 
From this, a quadratic regression will be performed on both groups of lines chosen to represent them in curvilinear lines since the lines of the scenario road are not straight at all.  Finally, once both regressions have been performed, an interpolation will be performed to know which points are within the 2 regressions and to be able to represent it as a mass of points and calculate the centroid of the lane. 

The result is as follows: 

<p align="center">
<img src="/2022-tfg-barbara-villalba/images/perception1.png" alt="Perception-result" width="500"/>
</p>

<p align="center">
<img src="/2022-tfg-barbara-villalba/images/perception2.png" alt="Perception-resultII" width="500"/>
</p>

When the insight was obtained, a simple PID controller was made to see how the insight worked. The result is as follows: 














