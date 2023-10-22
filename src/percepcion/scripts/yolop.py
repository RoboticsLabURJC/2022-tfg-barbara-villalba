#! /usr/bin/env python

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchshow as ts


normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

IMAGE = "/home/oem/images/image2.jpg"

image = cv2.imread(IMAGE)

#-- Resize image size 640x640 pixels
resized = cv2.resize(image, (640,640), interpolation = cv2.INTER_AREA)

#--Load model
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

#--Transform image for proceesing model
img = torch.unsqueeze(transform(resized), dim=0)

det_out, da_seg_out,ll_seg_out = model(img)

if len(ll_seg_out.shape) == 4 and ll_seg_out.shape[0] == 1:
    result_lanes = np.squeeze(ll_seg_out, axis=0)

if len(da_seg_out.shape) == 4 and ll_seg_out.shape[0] == 1:
    result_da_seg = np.squeeze(da_seg_out, axis=0)


image_lane_array = result_lanes.detach().cpu().numpy()
image_lane_array = np.transpose(image_lane_array, (1, 2, 0))



image_result_da_seg_array = result_da_seg.detach().cpu().numpy()
image_result_da_seg_array = np.transpose(image_result_da_seg_array, (1, 2, 0))

cv2.imshow("lane", image_lane_array[:,:,1])
cv2.imshow("da_seg", image_result_da_seg_array[:,:,1])
cv2.waitKey(0)



