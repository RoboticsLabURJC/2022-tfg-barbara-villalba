import rospy
from PyQt5.QtWidgets import *
import sys
import message_filters
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import numpy
import time
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
import cv2

class Interface(QWidget):
    def __init__(self):
        super(Interface, self).__init__()
        self.w = QWidget()
        self.layout = QGridLayout()
        self.w.resize(500, 500)
        self.w.setWindowTitle('Interfaz')
        self.disply_width = 500
        self.display_height = 500
        self.lastTime = time.time()


        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.layout.addWidget(self.image_label,0,0)

        self.w.setLayout(self.layout)
        self.w.show()


    def callback_image_chasis(self, rgb_msg, camera_info):

            rgb_image = CvBridge().imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            fps = 1/(time.time() - self.lastTime)
            cv2.putText(rgb_image,"FPS:" + str(round(fps,2)) ,(40,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            camera_info_K = numpy.array(camera_info.K).reshape([3, 3])
            camera_info_D = numpy.array(camera_info.D)
            rgb_undist = cv2.undistort(rgb_image, camera_info_K, camera_info_D)

            h, w, ch = rgb_undist.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QtGui.QImage(rgb_undist.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap(convert_to_Qt_format))
            
            self.w.show()

def main():
    app = QApplication(sys.argv)
    rospy.init_node("interface_node_py")
    interface = Interface()

    sub_camera = message_filters.Subscriber(
        '/iris/usb_cam/image_raw', Image)

    sub_camera_info = message_filters.Subscriber(
        '/iris/usb_cam/camera_info', CameraInfo)
    ts = message_filters.ApproximateTimeSynchronizer(
        [sub_camera, sub_camera_info], 10, 0.2)
    ts.registerCallback(interface.callback_image_chasis)

    sys.exit(app.exec_())



if __name__ == '__main__':
    main()