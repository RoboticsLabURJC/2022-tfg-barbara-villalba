
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest
from std_msgs.msg import String
import math

# Topics
SETPOINT_POS_PUB = "/mavros/setpoint_position/local"
STATE_SUB = "mavros/state"
COMANDER_POSITION_SUB = '/commands/control_position'
LOCAL_VEL_PUB = "/mavros/setpoint_velocity/cmd_vel_unstamped"
COMANDER_VELOCITY_SUB = '/commands/control_velocity'
LOCAL_POS__SUB = '/mavros/local_position/pose'
MODE_SUB = "/commands/mode"

# Services
ARMING_CLIENT = "/mavros/cmd/arming"
SET_MODE_CLIENT = "/mavros/set_mode"
LAND_CMD_SRV = "/mavros/cmd/land"
TAKE_OFF_CMD_CLIENT = "/mavros/cmd/takeoff"

# Modes
LAND = "LAND"
TAKE_OFF = "TAKE OFF"
POSITION = "POSITION"
VELOCITY = "VELOCITY"


MAX_HEIGH = 2

mode = ""

position = PoseStamped()
value_position = PoseStamped()
current_pos = PoseStamped()

raw = Twist()
value_velocity = Twist()
istakeoff = False

class Drone():
    def __init__(self):
        self.current_state = State()
        self.state_sub = rospy.Subscriber(STATE_SUB, State, callback=self.state_cb)
        self.mode_sub = rospy.Subscriber(MODE_SUB, String, callback=self.control_mode_cb)
        self.local_pos_pub = rospy.Publisher(SETPOINT_POS_PUB, PoseStamped, queue_size=10)
        self.value_position_sub = rospy.Subscriber(COMANDER_POSITION_SUB, PoseStamped, self.get_value_position_cb)
        self.value_velocity_sub = rospy.Subscriber(COMANDER_VELOCITY_SUB, Twist, self.get_value_velocity_cb)
        self.local_raw_pub = rospy.Publisher(LOCAL_VEL_PUB, Twist, queue_size=10)
        self.current_pos_sub = rospy.Subscriber(LOCAL_POS__SUB, PoseStamped, callback=self.current_pos_cb)
        self.orientation_list = []
        

    def state_cb(self, msg):
        self.current_state = msg

    def check_to_fly(self, offb_set_mode, arm_cmd, last_req, arming_client, set_mode_client):

        '''
        Check before to fly. The dron have a mode offboard and it's should armed before to fly
        '''

        if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if (set_mode_client.call(offb_set_mode).mode_sent is True):
                rospy.loginfo("OFFBOARD enabled")

            last_req = rospy.Time.now()

        else:
            if (not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if (arming_client.call(arm_cmd).success is True):
                    rospy.loginfo("Vehicle armed")

                last_req = rospy.Time.now()

    def current_pos_cb(self,pose):

        '''
        Callback, save current position for orientation in control position
        '''
        global current_pos
        current_pos = pose

    def control_mode_cb(self, msg):
        '''
        Callback, save mode: position,velocity,land or take off
        '''
        global mode
        mode = msg.data

    def get_value_position_cb(self, msg):

        '''
        Callback, save value for control position
        '''
        global value_position
        value_position = msg

    def get_value_velocity_cb(self, msg):

        '''
        Callback, save value for control velocity
        '''
        global value_velocity
        value_velocity = msg
    def control_position(self):

        '''
        Control position, position and orientation in radians in exe z (Yaw)

        Only works if the dron is not on the ground
        '''
        if (istakeoff):
            
            self.orientation_list = [current_pos.pose.orientation.x,
                    current_pos.pose.orientation.y,
                    current_pos.pose.orientation.z,
                    current_pos.pose.orientation.w]
            
            roll, pitch, yaw = euler_from_quaternion(self.orientation_list)
            
            quaternion = quaternion_from_euler(roll, pitch, yaw + math.radians(value_position.pose.orientation.z))

        
            position.pose.orientation.x = quaternion[0]
            position.pose.orientation.y = quaternion[1]
            position.pose.orientation.z = quaternion[2]
            position.pose.orientation.w = quaternion[3]

            position.pose.position.x = value_position.pose.position.x

    def control_velocity(self):
        
        '''
        Control velocity, velocity lineal and angular in exe z

        Only works if the dron is not on the ground
        '''

        if(istakeoff):
            raw.linear.x = value_velocity.linear.x
            raw.angular.z = math.radians(value_velocity.angular.z)


if __name__ == "__main__":
    rospy.init_node("teleop_node_py")
    drone = Drone()

    rospy.wait_for_service(ARMING_CLIENT)
    arming_client = rospy.ServiceProxy(ARMING_CLIENT, CommandBool)

    rospy.wait_for_service(SET_MODE_CLIENT)
    set_mode_client = rospy.ServiceProxy(SET_MODE_CLIENT, SetMode)

    rospy.wait_for_service(LAND_CMD_SRV)
    land_client = rospy.ServiceProxy(LAND_CMD_SRV, CommandTOL)

    rospy.wait_for_service(TAKE_OFF_CMD_CLIENT)
    take_off_client = rospy.ServiceProxy(TAKE_OFF_CMD_CLIENT, CommandTOL)

    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while (not rospy.is_shutdown() and not drone.current_state.connected):
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    land_cmd = CommandTOLRequest()
    land_cmd.min_pitch = 0
    land_cmd.yaw = 0
    land_cmd.latitude = 0
    land_cmd.longitude = 0
    land_cmd.altitude = 0

    last_req = rospy.Time.now()

    while (not rospy.is_shutdown()):
        drone.check_to_fly(offb_set_mode, arm_cmd, last_req,
                           arming_client, set_mode_client)

        if(mode == TAKE_OFF):
            position.pose.position.z = MAX_HEIGH
            drone.local_pos_pub.publish(position)
            istakeoff = True

        elif (mode == POSITION):
            drone.control_position()
            drone.local_pos_pub.publish(position)

        elif (mode == VELOCITY):
            drone.control_velocity()
            drone.local_raw_pub.publish(raw)

        elif(mode == LAND):
            if(rospy.Time.now() - last_req) > rospy.Duration(5.0):
                response = land_client.call(land_cmd)
                last_req = rospy.Time.now()
        
        rate.sleep()
