#! /usr/bin/env python
import rospy
import numpy as np
from op3_ros_utils import getWalkingParams, Robot
from copy import copy
import sys
from std_msgs.msg import Int32, String, Float32, Float64, Bool, Float32MultiArray



rospy.init_node("action_test")

robot = Robot()


robot.setGeneralControlModule("action_module")
robot.playMotion(7, wait_for_end=True)
rospy.spin()