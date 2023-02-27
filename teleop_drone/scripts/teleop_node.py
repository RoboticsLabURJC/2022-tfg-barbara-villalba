#! /usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from std_msgs.msg import String

# Topics
LOCAL_POS_PUB = "mavros/setpoint_position/local"
STATE_SUB = "mavros/state"

# Services
ARMING_CLIENT = "/mavros/cmd/arming"
SET_MODE_CLIENT = "/mavros/set_mode"

LAND = "LAND"
TAKE_OFF = "TAKE OFF"
ROW = "ROW"
PITCH = "PITCH"
YAW = "YAW"


behaviour = " "


class Drone():
    def __init__(self):
        self.current_state = State()

    def state_cb(self, msg):
        self.current_state = msg

    def check_to_fly(self, offb_set_mode, arm_cmd, last_req, arming_client, set_mode_client):

        if (self.current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if (set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

            last_req = rospy.Time.now()
        else:
            if (not self.current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                if (arming_client.call(arm_cmd).success == True):
                    rospy.loginfo("Vehicle armed")

                last_req = rospy.Time.now()

    def manage_behavior(self, msg):

        global behaviour
        behaviour = msg.data


    def check_behavior(self,pose):

        if (behaviour == LAND):
            pose.pose.position.x = 0
            pose.pose.position.y = 0
            pose.pose.position.z = 2

        elif(behaviour == TAKE_OFF):
            pose.pose.position.x = 0
            pose.pose.position.y = 0
            pose.pose.position.z = 0

        elif(behaviour == ROW and pose.pose.position.z != 0):
            pose.pose.orientation.x = 1
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0

        elif(behaviour == PITCH and pose.pose.position.z != 0):
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 1
            pose.pose.orientation.z = 0

        elif(behaviour == YAW and pose.pose.position.z != 0):
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 1


        return pose


if __name__ == "__main__":
    rospy.init_node("teleop_node_py")
    drone = Drone()

    state_sub = rospy.Subscriber(STATE_SUB, State, callback=drone.state_cb)

    sub = rospy.Subscriber('/interfaces/comander',
                           String, drone.manage_behavior)

    local_pos_pub = rospy.Publisher(LOCAL_POS_PUB, PoseStamped, queue_size=10)

    rospy.wait_for_service(ARMING_CLIENT)
    arming_client = rospy.ServiceProxy(ARMING_CLIENT, CommandBool)

    rospy.wait_for_service(SET_MODE_CLIENT)
    set_mode_client = rospy.ServiceProxy(SET_MODE_CLIENT, SetMode)

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while (not rospy.is_shutdown() and not drone.current_state.connected):
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    pose = PoseStamped()
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 0

    last_req = rospy.Time.now()

    while (not rospy.is_shutdown()):
        drone.check_to_fly(offb_set_mode, arm_cmd, last_req,
                           arming_client, set_mode_client)

        pose = drone.check_behavior(pose)
        local_pos_pub.publish(pose)

        rate.sleep()
