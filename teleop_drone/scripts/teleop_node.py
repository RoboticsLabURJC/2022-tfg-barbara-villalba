#! /usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

# Topics
LOCAL_POS_PUB = "mavros/setpoint_position/local"
STATE_SUB = "mavros/state"

# Services
ARMING_CLIENT = "/mavros/cmd/arming"
SET_MODE_CLIENT = "/mavros/set_mode"


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


if __name__ == "__main__":
    #--app = QApplication([])
    rospy.init_node("offb_node_py")
    drone = Drone()

    state_sub = rospy.Subscriber(STATE_SUB, State, callback=drone.state_cb)

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

    # -- Move with location local

    """
    pose = PoseStamped()
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 2
    """

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    while (not rospy.is_shutdown()):
        drone.check_to_fly(offb_set_mode, arm_cmd, last_req,
                           arming_client, set_mode_client)

        rate.sleep()

