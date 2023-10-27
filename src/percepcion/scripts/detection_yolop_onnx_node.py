#! /usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from torchvision import transforms
import torch
import time
import onnxruntime as ort

IMAGE_TOPIC = '/airsim_node/PX4/front_center_custom/Scene'

ROUTE_MODEL = 'hustvl/yolop'
MODEL = 'yolop'


class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/yolop/image',Image,queue_size=1)
        self.image_sub = rospy.Subscriber(IMAGE_TOPIC, Image, self.callback)

    def resize_unscale(self,image,new_shape):

        shape = image.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        canvas = np.zeros((new_shape[0], new_shape[1], 3))
        canvas.fill(114)
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
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_AREA)

        canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = image

        return canvas, r, dw, dh, new_unpad_w, new_unpad_h  # (dw,dh)
    
    def infer_yolop(self,image):
        height, width, _ = image.shape
        ort_session = ort.InferenceSession("/home/oem/YOLOP/weights/yolop-640-640.onnx")

        outputs_info = ort_session.get_outputs()
        inputs_info = ort_session.get_inputs()

        canvas, r, dw, dh, new_unpad_w, new_unpad_h = self.resize_unscale(image, (640,640))

        img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
        img /= 255.0
        img[:, :, 0] -= 0.485
        img[:, :, 1] -= 0.456
        img[:, :, 2] -= 0.406
        img[:, :, 0] /= 0.229
        img[:, :, 1] /= 0.224
        img[:, :, 2] /= 0.225

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, 0)  # (1, 3,640,640)

        # inference: (1,n,6) (1,2,640,640) (1,2,640,640)
        _, da_seg_out, ll_seg_out = ort_session.run(
            ['det_out', 'drive_area_seg', 'lane_line_seg'],
            input_feed={"images": img}
        )

            # select da & ll segment area.
        da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
        ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

        da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
        ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)
        print(da_seg_mask.shape)
        print(ll_seg_mask.shape)

        color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
        color_area[da_seg_mask == 1] = [0, 255, 0]
        color_area[ll_seg_mask == 1] = [255, 0, 0]
        color_seg = color_area

        # convert to BGR
        color_seg = color_seg[..., ::-1]
        color_mask = np.mean(color_seg, 2)
        img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
        img_merge = img_merge[:, :, ::-1]

        # merge: resize to original size
        img_merge[color_mask != 0] = \
            img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
        img_merge = img_merge.astype(np.uint8)
        img_merge = cv2.resize(img_merge, (width, height),
                           interpolation=cv2.INTER_LINEAR)



        return img_merge



    def callback(self, data):

        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8") 

        image_result = self.infer_yolop(cv_image)

        image_message = self.bridge.cv2_to_imgmsg(image_result, encoding="bgr8")
        
        self.image_pub.publish(image_message)
        
        
class ImageViewer:
    def __init__(self):
        self.subscriber = ImageSubscriber()

    def start(self):
        try:
          rospy.spin()
        except rospy.ROSInterruptException:
              pass
            

if __name__ == '__main__':
    rospy.init_node("det_node_py")
    image_viewer = ImageViewer()
    image_viewer.start()