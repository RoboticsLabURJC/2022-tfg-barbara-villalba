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
from std_msgs.msg import String

MAX_HEIGHT = 500
MAX_WIDTH = 500


class Interface(QWidget):
    def __init__(self):
        super(Interface, self).__init__()
        self.w = QWidget()
        self.layout = QGridLayout()
        self.w.resize(MAX_HEIGHT, MAX_WIDTH)
        self.w.setWindowTitle('Interfaz')
        self.disply_width = MAX_HEIGHT
        self.display_height = MAX_WIDTH
        self.lastTime = time.time()

        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)

        self.layout.addWidget(self.image_label, 0, 0)

        self.btn_land = QPushButton(self.w)
        self.btn_land.setText('LAND')
        self.btn_land.clicked.connect(self.land_clicked)
        self.layout.addWidget(self.btn_land, 5, 0)

        self.btn_take_off = QPushButton(self.w)
        self.btn_take_off.setText('TAKE OFF')
        self.btn_take_off.clicked.connect(self.take_off_clicked)
        self.layout.addWidget(self.btn_take_off, 7, 0)

        self.btn_row = QPushButton(self.w)
        self.btn_row.setText('ROW')
        self.btn_row.clicked.connect(self.row)
        self.layout.addWidget(self.btn_row, 1, 0)

        self.btn_pitch = QPushButton(self.w)
        self.btn_pitch.setText('PITCH')
        self.btn_pitch.clicked.connect(self.pitch)
        self.layout.addWidget(self.btn_pitch, 2, 0)

        self.btn_yaw = QPushButton(self.w)
        self.btn_yaw.setText('YAW')
        self.btn_yaw.clicked.connect(self.yaw)
        self.layout.addWidget(self.btn_yaw, 3, 0)

        self.pub = rospy.Publisher(
            '/interfaces/comander', String, queue_size=10)

        self.pub.publish('')

        self.w.setLayout(self.layout)
        self.w.show()

    def land_clicked(self):
        self.pub.publish('LAND')

    def take_off_clicked(self):
        self.pub.publish('TAKE OFF')

    def row(self):
        self.pub.publish('ROW')

    def pitch(self):
        self.pub.publish('PITCH')

    def yaw(self):
        self.pub.publish('YAW')

    def callback_image(self, rgb_msg, camera_info):

        fps = 1/(time.time() - self.lastTime)
        rgb_image = CvBridge().imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        
        cv2.putText(rgb_image, "FPS:" + str(round(fps, 2)), (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        camera_info_K = numpy.array(camera_info.K).reshape([3, 3])
        camera_info_D = numpy.array(camera_info.D)
        rgb_undist = cv2.undistort(rgb_image, camera_info_K, camera_info_D)

        h, w, ch = rgb_undist.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_undist.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap(convert_to_Qt_format))
        
        self.lastTime = time.time()
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
    ts.registerCallback(interface.callback_image)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
