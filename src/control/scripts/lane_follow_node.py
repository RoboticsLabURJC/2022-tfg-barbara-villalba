#! /usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest, CommandTOL, CommandTOLRequest
from yolop.msg import MassCentre

#--Topics
STATE_SUB = "mavros/state"
MODE_SUB = "/commands/mode"
LOCAL_VEL_PUB = "/mavros/setpoint_velocity/cmd_vel_unstamped"
MASS_CENTRE = "/yolop/detection_lane/mass_centre_lane"

#--Services
ARMING_CLIENT = "/mavros/cmd/arming"
SET_MODE_CLIENT = "/mavros/set_mode"
TAKE_OFF_CLIENT = "/mavros/cmd/takeoff"

#--Modes
HOLD = "AUTO.LOITER"
OFFBOARD = "OFFBOARD"


class LaneFollow():
    def __init__(self):
        self.current_state = State()
        self.mass_centre = MassCentre()
        self.state_sub = rospy.Subscriber(STATE_SUB, State, callback=self.state_cb) 
        self.local_raw_pub = rospy.Publisher(LOCAL_VEL_PUB, Twist, queue_size=10)
        self.mass_centre_sub = rospy.Subscriber(MASS_CENTRE,MassCentre,callback=self.mass_centre_cb)
        rospy.wait_for_service(SET_MODE_CLIENT)
        self.set_mode_client = rospy.ServiceProxy(SET_MODE_CLIENT, SetMode)
        self.prev_error = 0
        self.error = 0

        self.velocity = Twist()
        self.KP_w = 0.018
        self.KP_v = 0.00025
      

    def state_cb(self, msg):
        self.current_state = msg

    def mass_centre_cb(self,msg):
        self.mass_centre = msg


    def controller(self,mass_centre):

        self.error = 160.0 - mass_centre.cx

        print(self.error)
    
        self.velocity.angular.z = (self.KP_w * self.error)
        self.velocity.linear.y = (self.KP_v * self.error)

if __name__ == '__main__':
    rospy.init_node("lane_follow_node_py")

    lane_follow = LaneFollow()
    rate = rospy.Rate(20)

    rospy.wait_for_service(ARMING_CLIENT)
    arming_client = rospy.ServiceProxy(ARMING_CLIENT, CommandBool)

    rospy.wait_for_service(TAKE_OFF_CLIENT)
    take_off_client = rospy.ServiceProxy(TAKE_OFF_CLIENT, CommandTOL)

    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while (not rospy.is_shutdown() and not lane_follow.current_state.connected):
        rate.sleep()

    set_mode = SetModeRequest()
    set_mode.custom_mode = 'OFFBOARD'

    last_req = rospy.Time.now()

    lane_follow.velocity.linear.x = 1.5
    lane_follow.velocity.linear.y = 0
    lane_follow.velocity.linear.z = 0
    lane_follow.velocity.angular.z = 0

    while (not rospy.is_shutdown()):

        if (lane_follow.current_state.mode != OFFBOARD and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if (lane_follow.set_mode_client.call(set_mode).mode_sent is True):
                rospy.loginfo("OFFBOARD enabled")
            #last_req = rospy.Time.now()

        
        lane_follow.controller(lane_follow.mass_centre)

        lane_follow.local_raw_pub.publish(lane_follow.velocity)

        #lane_follow.prev_error = lane_follow.error

        rate.sleep()









    