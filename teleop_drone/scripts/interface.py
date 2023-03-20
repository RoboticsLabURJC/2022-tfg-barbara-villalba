import rospy
from PyQt5.QtWidgets import *
from geometry_msgs.msg import PoseStamped,Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import time
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5 import QtGui
import cv2
from std_msgs.msg import String

MAX_HEIGHT = 100
MAX_WIDTH = 700

MAX_POSITION_X = 5
MIN_POSITION_X = -5

MAX_ANGLE_YAW = 180
MIN_ANGLE_YAW = 0

MAX_VELOCITY_X = 5
MIN_VELOCITY_X = 0

MAX_SECONDS = 1

# Topics
IMAGE_TOPIC = '/iris/usb_cam/image_raw'
MODE_TOPIC = '/commands/mode'
POSITION_TOPIC = '/commands/control_position'
VELOCITY_TOPIC = '/commands/control_velocity'


class ImageCapture():
    def __init__(self,interface):
        self.sub_camera = rospy.Subscriber(IMAGE_TOPIC, Image,callback=self.callback_image)
        self.w = interface.window
        self.image_label = interface.label
        self.bridge = CvBridge()
        self.lastTime = time.time()
        self.counter_time = 0.0
        self.isfirst = False 
        self.fps = 0
        self.list = []
        
    def calculate_fps(self):
        fps = 1/(time.time() - self.lastTime)
        self.list.append(fps)
        return sum(self.list)/len(self.list)

    def callback_image(self,image):
        rgb_image =  self.bridge .imgmsg_to_cv2(image, desired_encoding="rgb8")
        self.counter_time += (time.time() - self.lastTime)
        if(not self.isfirst):
            self.fps = self.calculate_fps()
            self.isfirst = True

        #--Update each 1 seconds
        if(self.counter_time > MAX_SECONDS):
            self.fps = self.calculate_fps()
            self.counter_time = 0.0

        cv2.putText(rgb_image, "FPS:" + str(int(self.fps)), (40, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        
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
        self.window.setWindowTitle('Teleop Drone')
        self.label = QLabel("Camera View")
        self.window.resize(MAX_WIDTH,MAX_HEIGHT)
        self.pub_mode = rospy.Publisher(MODE_TOPIC, String, queue_size=10)
        self.pub_velocity = rospy.Publisher(VELOCITY_TOPIC, Twist, queue_size=10)
        self.pub_position = rospy.Publisher(POSITION_TOPIC, PoseStamped, queue_size=10)
        self.msg_v = Twist()
        self.msg_position = PoseStamped()

    def addImages(self):
        self.layout.addWidget(self.label,0,0)

    def addButtons(self):
        self.btn_land = QPushButton(self.window)
        self.btn_land.setText('LAND')
        self.btn_land.clicked.connect(self.land_clicked)
        self.layout.addWidget(self.btn_land,0,1)

        self.btn_take_off = QPushButton(self.window)
        self.btn_take_off.setText('TAKE OFF')
        self.btn_take_off.clicked.connect(self.take_off_clicked)
        self.layout.addWidget(self.btn_take_off,0,3)

        self.btn_control_position = QPushButton(self.window)
        self.btn_control_position.setText('POSITION')
        self.btn_control_position.clicked.connect(self.position_clicked)
        self.layout.addWidget(self.btn_control_position,1,1)

        self.btn_control_velocity = QPushButton(self.window)
        self.btn_control_velocity.setText('VELOCITY')
        self.btn_control_velocity.clicked.connect(self.velocity_clicked)
        self.layout.addWidget(self.btn_control_velocity,1,3)

    def land_clicked(self):
        self.pub_mode.publish('LAND') 

    def take_off_clicked(self):
        self.pub_mode.publish('TAKE OFF') 

    def position_clicked(self):
        self.pub_mode.publish('POSITION') 

    def velocity_clicked(self):
        self.pub_mode.publish('VELOCITY') 

    def take_off_clicked(self):
        self.pub_mode.publish('TAKE OFF') 

    def comanderPosition(self):
        self.ValuePosition.setText( "Position in exe x : " + str(self.PositionSlider.value()))
        self.layout.addWidget(self.ValuePosition,5,1)
        self.msg_position.pose.position.x = self.PositionSlider.value()
        self.pub_position.publish(self.msg_position)

    def comanderAngleYawPosition(self):
        self.ValueYawPosition.setText( "Yaw: " + str(self.YawPositionSlider.value()) + " degrees")
        self.layout.addWidget(self.ValueYawPosition,2,1)
        self.msg_position.pose.orientation.z = self.YawPositionSlider.value()
        self.pub_position.publish(self.msg_position)

    def comanderVelocity(self):
        
        self.ValueVelocity.setText( "Velocity in exe x : " + str(self.VelocitySlider.value()))
        self.layout.addWidget(self.ValueVelocity,5,3)
        self.msg_v.linear.x = self.VelocitySlider.value()
        self.pub_velocity.publish(self.msg_v)


    def comanderVelocityRawZ(self):
        self.ValueRawZ.setText( "Yaw : " + str(self.Raw_zSlider.value()) + " degrees")
        self.layout.addWidget(self.ValueRawZ,2,3)
        self.msg_v.angular.z = self.Raw_zSlider.value()
        self.pub_velocity.publish(self.msg_v)

    def addSliders(self):
        self.PositionSlider = QSlider(QtCore.Qt.Horizontal)
        self.PositionSlider.setMinimum(MIN_POSITION_X)
        self.PositionSlider.setMaximum(MAX_POSITION_X)
        self.PositionSlider.setValue(0)
        self.PositionSlider.setTickPosition(QSlider.TicksBelow)
        self.PositionSlider.setTickInterval(5)
        self.layout.addWidget(self.PositionSlider,6,1)
        self.ValuePosition = QLabel("Value")
        self.ValuePosition.setText( "Position in exe x : " + str(self.PositionSlider.value()))
        self.layout.addWidget(self.ValuePosition,5,1)
        self.PositionSlider.valueChanged.connect(self.comanderPosition)

        self.YawPositionSlider = QSlider(QtCore.Qt.Horizontal)
        self.YawPositionSlider.setMinimum(MIN_ANGLE_YAW)
        self.YawPositionSlider.setMaximum(MAX_ANGLE_YAW)
        self.YawPositionSlider.setValue(0)
        self.YawPositionSlider.setTickPosition(QSlider.TicksBelow)
        self.YawPositionSlider.setTickInterval(5)
        self.layout.addWidget(self.YawPositionSlider,4,1)
        self.ValueYawPosition = QLabel("Value")
        self.ValueYawPosition.setText( "Yaw: " + str(self.YawPositionSlider.value()) + " degrees")
        self.layout.addWidget(self.ValueYawPosition,2,1)
        self.YawPositionSlider.valueChanged.connect(self.comanderAngleYawPosition)
        
        self.VelocitySlider = QSlider(QtCore.Qt.Horizontal)
        self.VelocitySlider.setMinimum(MIN_VELOCITY_X)
        self.VelocitySlider.setMaximum(MAX_VELOCITY_X)
        self.VelocitySlider.setValue(0)
        self.VelocitySlider.setTickPosition(QSlider.TicksBelow)
        self.VelocitySlider.setTickInterval(5)
        self.layout.addWidget(self.VelocitySlider,6,3)
        self.ValueVelocity = QLabel("Value")
        self.ValueVelocity.setText( "Velocity in exe x : " + str(self.VelocitySlider.value()))
        self.layout.addWidget(self.ValueVelocity,5,3)
        self.VelocitySlider.valueChanged.connect(self.comanderVelocity)

        self.Raw_zSlider = QSlider(QtCore.Qt.Horizontal)
        self.Raw_zSlider.setMinimum(MIN_ANGLE_YAW)
        self.Raw_zSlider.setMaximum(MAX_ANGLE_YAW)
        self.Raw_zSlider.setValue(0)
        self.Raw_zSlider.setTickPosition(QSlider.TicksBelow)
        self.Raw_zSlider.setTickInterval(5)
        self.layout.addWidget(self.Raw_zSlider,4,3)
        self.ValueRawZ = QLabel("Value")
        self.ValueRawZ.setText( "Yaw : " + str(self.Raw_zSlider.value()) + " degrees")
        self.layout.addWidget(self.ValueRawZ,2,3)
        self.Raw_zSlider.valueChanged.connect(self.comanderVelocityRawZ)


    def executeInterface(self):
        self.window.setLayout(self.layout)
        self.window.show()
        self.app.exec_()

rospy.init_node("interface_node_py")
interface = Interface()
interface.addButtons()
interface.addImages()
interface.addSliders()
image = ImageCapture(interface)

def main():
    interface.executeInterface()
   
if __name__ == '__main__':
    main()
