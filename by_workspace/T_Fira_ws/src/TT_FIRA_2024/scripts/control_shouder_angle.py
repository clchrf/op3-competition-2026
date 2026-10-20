#! /usr/bin/python

import rospy
import cv2

from op3_ros_utils import getWalkingParams, Robot
from copy import copy
import sys

DEBUG_MODE = True # Show the detected image

class States:
    INIT = -1
    READY = 0 # Waits for start button
    WALK_FORWARD = 1
    #WALK_FORWARD = 1 # Moves the head, looking for the ball
    #WALK_SLOWLY_FORWARD = 2
    #WALK_PREBACKWARDS = 3
    #WALK_BACKWARDS = 4
    #WALK_BACKWARDS_2 = 5
    END = 99
rospy.init_node("fira_sprint")

# Create robot
robot = Robot()

rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages

def init():
    # Set ctrl modules of all actions to joint, so we can reset robot position 
    robot.setGeneralControlModule("action_module")
    # robot.setGrippersPos(left=0.0, right=0.0)
    #robot.playMotion(9, wait_for_end=True)#stand up

    #rospy.sleep(1)
    robot.playMotion(229, wait_for_end=True)#archery_set, put bow and arrow
    # Call initial robot position
    #robot.playMotion(230, wait_for_end=True)#play set-up motion of archery 

    # Set ctrl module to walking, this actually only sets the legs
    robot.setGeneralControlModule("online_walking_module")
    
    # Set joint modules of head joints to none so we can control them directly
    robot.setJointsControlModule(["head_pan", "l_sho_pitch"], ["none", "none"])

    robot.setJointPos(["l_sho_pitch"], [-0.2])

    rospy.sleep(1.0)

# TODO remember to put to 0
STEP_LEVEL = 0



tickrate = 60
rate = rospy.Rate(tickrate)

# TODO remember to put to 0
STEP_LEVEL = 0

currState = States.INIT

count = 0
while not rospy.is_shutdown():
    
    STEP_LEVEL = 0

    if currState == States.INIT:
        print("[INIT]")
        init()

        # Transition
        tick_count = 0

        direction = False
        currState = States.READY
    
    elif currState == States.READY:
        print("[READY]")
        if robot.buttonCheck("start"):
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.2])
            tick_count = 0
            
            currState = States.WALK_FORWARD

    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD]")
        pan_angle = robot.joint_pos["head_pan"]
        print(pan_angle)
        if pan_angle > 0.1:
            theta = 5
        elif pan_angle < -0.1:
            theta = -5
        else:
            theta = 0

        robot.walkVelocities(x=32, th=theta, balance=True, hip_pitch=6)
        tick_count += 1

    elif currState == States.END:
        print("[END]")
        robot.walkStop()


    count = count + 1
    rate.sleep()