import rospy
from op3_ros_utils import getWalkingParams, Robot
import cv2
import numpy as np
import time
import math
from scipy import optimize
from sensor_msgs.msg import Imu

rospy.init_node("fira_Archery")

robot = Robot()

class States:
    INIT = -1
    READY = 0 
    WALK_FORWARD = 1
    CORRECT_ANGLE = 2
    TURN_AROUND = 3
    INIT_SHOOTING_POSITION = 4
    CORRECT_POSITION = 5
    CALCULATE_TARGET = 6
    END = 99


robot.setGeneralControlModule("action_module")
robot.playMotion(200 , wait_for_end=True)
time.sleep(2)

robot.setGeneralControlModule("action_module")
robot.playMotion(202 , wait_for_end=True)
time.sleep(2)

robot.setGeneralControlModule("action_module")
robot.playMotion(203 , wait_for_end=True)
time.sleep(2)

