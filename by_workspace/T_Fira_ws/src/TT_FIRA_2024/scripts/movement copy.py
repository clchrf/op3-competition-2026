#! /usr/bin/python
import rospy
from op3_ros_utils import Robot
import sys

import cv2
import numpy as np
import math
import time   

# init rosnode
rospy.init_node("fira_Archery_PF")
# rospy.init_node("fira_Archery_PF", anonymous=True)

# Create robot
robot = Robot()
#rospy.init_node("fira_Archery_PF")     #change to line 13
rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages
               
if __name__ == '__main__':
    # while not rospy.is_shutdown():

    # robot.setGeneralControlModule("action_module")
    # robot.playMotion(188, wait_for_end=True)
    # time.sleep(10)

    # robot.setGeneralControlModule("action_module")
    # robot.playMotion(189, wait_for_end=True)

    # robot.setGeneralControlModule("action_module")
    # robot.playMotion(195, wait_for_end=True)
    # time.sleep(7)

    # robot.setGeneralControlModule("action_module")
    # robot.playMotion(196, wait_for_end=True)

    robot.setGeneralControlModule("action_module")
    robot.playMotion(197, wait_for_end=True)
    time.sleep(10)

    robot.setGeneralControlModule("action_module")
    robot.playMotion(198, wait_for_end=True)