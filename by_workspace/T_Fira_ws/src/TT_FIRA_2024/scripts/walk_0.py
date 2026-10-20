#! /usr/bin/python

import rospy
import numpy as np
import cv2

from op3_ros_utils import getWalkingParams, Robot
import sys

DEBUG_MODE = True # Show the detected image

cap = cv2.VideoCapture(0)

class States:
    INIT = -1
    READY = 0 # Waits for start button
    WALK_FORWARD = 1 # Moves the head, looking for the ball
    END = 99

rospy.init_node("walk_op3")

# Create robot
robot = Robot()

rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages

def init():
    # Set ctrl modules of all actions to joint, so we can reset robot position
    robot.setGeneralControlModule("action_module")

    # Call initial robot position
    robot.playMotion(1, wait_for_end=True)

    # Set ctrl module to walking, this actually only sets the legs
    robot.setGeneralControlModule("walking_module")
    
    # Set joint modules of head joints to none so we can control them directly
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])

    robot.setJointPos(["head_pan", "head_tilt"], [0, -0.3])

    rospy.sleep(1.0)

tickrate = 60
rate = rospy.Rate(tickrate)

# TODO remember to put to 0
STEP_LEVEL = 0

currState = States.INIT
while not rospy.is_shutdown():
    
    ret, frame = cap.read()

    cv2.waitKey(3)
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
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.5])
            tick_count = 0
            robot.walkStart()
            currState = States.WALK_FORWARD
        
    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD]")
        tilt_angle = robot.joint_pos["head_tilt"]
        pan_angle = robot.joint_pos["head_pan"]
        print(pan_angle)
        print(tilt_angle)
        # if center_x > image_center_x:
        #     theta = 0.3
        # elif center_x < image_center_x:
        #     theta = -0.3
        # else :
        #     theta = 0
     
        robot.walkVelocities(x=5, th=0, balance=True, hip_pitch=7)

        tick_count += 1
        #rospy.sleep(8)
        #currState = States.PICK_BAR
        # TODO change this
        # if obj_area > CROSS_AREA:
        #      currState = States.PICK_BAR
        print(tilt_angle)
        if tick_count == 1000:
            robot.walkStop()
            # robot_weightlift.onlineWalkCommand(direction="turn_left", start_leg="right", step_num=1,
            #         front_length=0.4, step_angle=15,step_time=0.5)
        
            currState = States.END
 

    elif currState == States.END:
        print("[END]")
        
    rate.sleep()
