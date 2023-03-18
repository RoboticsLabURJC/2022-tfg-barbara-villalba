import rospy
from PyQt5.QtWidgets import *
from geometry_msgs.msg import PoseStamped,Twist
import sys
import message_filters
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import numpy
import time
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5 import QtGui
import cv2
from std_msgs.msg import String

MAX_HEIGHT = 200
MAX_WIDTH = 800

IMAGE_TOPIC = '/iris/usb_cam/image_raw'

MODE_TOPIC = '/commands/mode'

POSITION_TOPIC = '/commands/control_position'
VALUE_POSITION_TOPIC = '/commands/control_position/value'

VELOCITY_TOPIC = '/commands/control_velocity'

class ImageCapture():
    def __init__(self,interface):
        self.sub_camera = rospy.Subscriber(IMAGE_TOPIC, Image,callback=self.callback_image)
        self.w = interface.window
        self.image_label = interface.label
        self.bridge = CvBridge()
        self.lastTime = time.time()
        self.list = []

    def Calculate_fps(self):
        fps = 1/(time.time() - self.lastTime)
        self.list.append(fps)
        return sum(self.list)/len(self.list)
    
    def callback_image(self,image):
        rgb_image =  self.bridge .imgmsg_to_cv2(image, desired_encoding="rgb8")
        fps = self.Calculate_fps()
        cv2.putText(rgb_image, "FPS:" + str(int(fps)), (40, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap(convert_to_Qt_format))
        self.lastTime = time.time()
        

class Interface():
    def __init__(self):
        self.app = QApplication([])
        self.window = QWidget()
        self.layout = QGridLayout()
        self.window.setWindowTitle('Interfaz')
        self.label = QLabel("Camera View")
        self.window.resize(MAX_WIDTH,MAX_HEIGHT)
        self.pub_mode = rospy.Publisher(MODE_TOPIC, String, queue_size=10)
        self.pub_position = rospy.Publisher(POSITION_TOPIC, String, queue_size=10)
        self.pub_velocity = rospy.Publisher(VELOCITY_TOPIC, Twist, queue_size=10)
        self.pub_value_position = rospy.Publisher(VALUE_POSITION_TOPIC, PoseStamped, queue_size=10)
        self.pub_position.publish('')
        self.msg_v = Twist()
        self.msg_position = PoseStamped()

    def addImages(self):
        self.layout.addWidget(self.label,0,0)
        

    def addTitle(self):
        self.label_title_position = QLabel("Control position")
        self.layout.addWidget(self.label_title_position,1,1)

        self.label_title_velocity = QLabel("Control velocity")
        self.layout.addWidget(self.label_title_velocity,1,3)


    def addButtons(self):
        self.btn_land = QPushButton(self.window)
        self.btn_land.setText('LAND')
        self.btn_land.clicked.connect(self.land_clicked)
        self.layout.addWidget(self.btn_land,0,1)

        self.btn_take_off = QPushButton(self.window)
        self.btn_take_off.setText('TAKE OFF')
        self.btn_take_off.clicked.connect(self.take_off_clicked)
        self.layout.addWidget(self.btn_take_off,0,3)

        
        self.btn_row_position = QPushButton(self.window)
        self.btn_row_position.setText('ROW')
        self.btn_row_position.clicked.connect(self.row_position)
        self.layout.addWidget(self.btn_row_position, 4, 1)

        self.btn_pitch_position = QPushButton(self.window)
        self.btn_pitch_position.setText('PITCH')
        self.btn_pitch_position.clicked.connect(self.pitch_position)
        self.layout.addWidget(self.btn_pitch_position, 5, 1)

        self.btn_yaw_position = QPushButton(self.window)
        self.btn_yaw_position.setText('YAW')
        self.btn_yaw_position.clicked.connect(self.yaw_position)
        self.layout.addWidget(self.btn_yaw_position, 6, 1)

       
    def land_clicked(self):
        self.pub_mode.publish('LAND') 

    def take_off_clicked(self):
        self.pub_mode.publish('TAKE OFF') 

    def row_position(self):
        self.pub_position.publish('ROW')

    def pitch_position(self):
        self.pub_position.publish('PITCH')

    def yaw_position(self):
        self.pub_position.publish('YAW')

    def comanderPosition(self):
        self.ValuePosition.setText( "Position in exe x : " + str(self.PositionSlider.value()))
        self.layout.addWidget(self.ValuePosition,7,1)
        self.msg_position.pose.position.x = self.PositionSlider.value()
        self.pub_value_position.publish(self.msg_position)

    def comanderVelocity(self):
        
        self.ValueVelocity.setText( "Velocity in exe x : " + str(self.VelocitySlider.value()))
        self.layout.addWidget(self.ValueVelocity,4,4)
        self.msg_v.linear.x = self.VelocitySlider.value()
        self.pub_velocity.publish(self.msg_v)

    def comanderVelocityRawX(self):
        self.ValueRawX.setText( "Row : " + str(self.Raw_xSlider.value()))
        self.layout.addWidget(self.ValueRawX,5,4)
        self.msg_v.angular.x = self.Raw_xSlider.value()
        self.pub_velocity.publish(self.msg_v)

    def comanderVelocityRawY(self):
        self.ValueRawY.setText( "Pitch : " + str(self.Raw_ySlider.value()))
        self.layout.addWidget(self.ValueRawY,6,4)
        self.msg_v.angular.y = self.Raw_ySlider.value()
        self.pub_velocity.publish(self.msg_v)

    def comanderVelocityRawZ(self):
        self.ValueRawZ.setText( "Yaw : " + str(self.Raw_zSlider.value()))
        self.layout.addWidget(self.ValueRawZ,7,4)
        self.msg_v.angular.z = self.Raw_zSlider.value()
        self.pub_velocity.publish(self.msg_v)

    def addSliders(self):
        self.PositionSlider = QSlider(QtCore.Qt.Horizontal)
        self.PositionSlider.setMinimum(-5)
        self.PositionSlider.setMaximum(5)
        self.PositionSlider.setValue(0)
        self.PositionSlider.setTickPosition(QSlider.TicksBelow)
        self.PositionSlider.setTickInterval(5)
        self.layout.addWidget(self.PositionSlider,8,1)
        self.ValuePosition = QLabel("Value")
        self.ValuePosition.setText( "Position in exe x : " + str(self.PositionSlider.value()))
        self.layout.addWidget(self.ValuePosition,7,1)
        self.PositionSlider.valueChanged.connect(self.comanderPosition)
        
        self.VelocitySlider = QSlider(QtCore.Qt.Horizontal)
        self.VelocitySlider.setMinimum(0)
        self.VelocitySlider.setMaximum(5)
        self.VelocitySlider.setValue(0)
        self.VelocitySlider.setTickPosition(QSlider.TicksBelow)
        self.VelocitySlider.setTickInterval(5)
        self.layout.addWidget(self.VelocitySlider,4,3)
        self.ValueVelocity = QLabel("Value")
        self.ValueVelocity.setText( "Velocity in exe x : " + str(self.VelocitySlider.value()))
        self.layout.addWidget(self.ValueVelocity,4,4)
        self.VelocitySlider.valueChanged.connect(self.comanderVelocity)

        self.Raw_xSlider = QSlider(QtCore.Qt.Horizontal)
        self.Raw_xSlider.setMinimum(0)
        self.Raw_xSlider.setMaximum(5)
        self.Raw_xSlider.setValue(0)
        self.Raw_xSlider.setTickPosition(QSlider.TicksBelow)
        self.Raw_xSlider.setTickInterval(5)
        self.layout.addWidget(self.Raw_xSlider,5,3)
        self.ValueRawX = QLabel("Value")
        self.ValueRawX.setText( "Row : " + str(self.Raw_xSlider.value()))
        self.layout.addWidget(self.ValueRawX,5,4)
        self.Raw_xSlider.valueChanged.connect(self.comanderVelocityRawX)


        self.Raw_ySlider = QSlider(QtCore.Qt.Horizontal)
        self.Raw_ySlider.setMinimum(0)
        self.Raw_ySlider.setMaximum(5)
        self.Raw_ySlider.setValue(0)
        self.Raw_ySlider.setTickPosition(QSlider.TicksBelow)
        self.Raw_ySlider.setTickInterval(5)
        self.layout.addWidget(self.Raw_ySlider,6,3)
        self.ValueRawY = QLabel("Value")
        self.ValueRawY.setText( "Pitch : " + str(self.Raw_ySlider.value()))
        self.layout.addWidget(self.ValueRawY,6,4)
        self.Raw_ySlider.valueChanged.connect(self.comanderVelocityRawY)


        self.Raw_zSlider = QSlider(QtCore.Qt.Horizontal)
        self.Raw_zSlider.setMinimum(0)
        self.Raw_zSlider.setMaximum(5)
        self.Raw_zSlider.setValue(0)
        self.Raw_zSlider.setTickPosition(QSlider.TicksBelow)
        self.Raw_zSlider.setTickInterval(5)
        self.layout.addWidget(self.Raw_zSlider,7,3)
        self.ValueRawZ = QLabel("Value")
        self.ValueRawZ.setText( "Yaw : " + str(self.Raw_zSlider.value()))
        self.layout.addWidget(self.ValueRawZ,7,4)
        self.Raw_zSlider.valueChanged.connect(self.comanderVelocityRawZ)


    def executeInterface(self):
        self.window.setLayout(self.layout)
        self.window.show()
        self.app.exec_()

rospy.init_node("interface_node_py")
interface = Interface()
interface.addTitle()
interface.addButtons()
interface.addImages()
interface.addSliders()

image = ImageCapture(interface)

def main():
    interface.executeInterface()
   
if __name__ == '__main__':
    main()
