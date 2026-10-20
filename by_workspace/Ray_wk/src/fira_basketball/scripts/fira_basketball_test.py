#! /usr/bin/python

import rospy
import numpy as np
import cv2

from op3_ros_utils import getWalkingParams, Robot
from op3_utils import Robot_weightlift
from vision import *
from copy import copy
import sys

# Forgive me Sasuke
#from pygame import mixer

DEBUG_MODE = True # Show the detected image

MIN_AREA = 300 # Minimum area of objects to consider for vision
BALL_PICKUP_SIZE = 11000
BASKET_DUNK_SIZE = 28500
SIDE_STEP_TIME = 3.5
HEAD_SEARCH_SPEED = 0.065

class States:
    INIT = -1
    READY = 0 # Waits for start button
    WALK_BACKWARDS = 1 # Walk backwards for a bit
    FOCUS_BALL = 2
    FIND_BALL = 3 # Moves the head, looking for the ball
    PICK_BALL = 4 # Picks the ball
    
    FIND_BASKET = 5 # Moves head, looking for the basket
    FOCUS_BASKET = 6 # Center heads on basket
    FACE_BASKET = 7 # Rotates in place to face the basket
    WALK_TO_BASKET = 8 # Rotates in place to face the basket
    WALK_SIDEWAYS_BASKET = 9
    DUNK = 10
    END = 99

# Functions to be passed to vision system
func1 = detectSingleColor
func2 = detect2Color
args1 = ((np.array([13, 90, 150]), np.array([25, 255, 255])),)
args2 = ((np.array([0, 130, 70]), np.array([10, 255, 255])),
         (np.array([170, 130, 70]), np.array([180, 255, 255])))

# Create vision system
vision = VisionSystem(pipeline_funcs=[func1, func2],
                      pipeline_args=[args1, args2], debug=DEBUG_MODE, verbose=0)

# Subscribe to cv_camera topic with vision system
rospy.Subscriber("/cv_camera/image_raw", Image, vision.read, queue_size=1)

# Create robot
robot = Robot()


# Iinitialize Node
rospy.init_node("fira_basketball")

rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages

def init():
    # Set ctrl modules of all actions to joint, so we can reset robot position
    robot.setGeneralControlModule("action_module")

    '''robot.setGrippersPos(left=100.0, right=0.0)'''

    # Call initial robot position
    robot.playMotion(1, wait_for_end=True)

    # Set ctrl module to walking, this actually only sets the legs
    robot.setGeneralControlModule("walking_module")
    
    # Set joint modules of head joints to none so we can control them directly
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])

    robot.setJointPos(["head_tilt"], [-0.8])


    rospy.sleep(1.0)

def center_head_on_object(obj_pos):
    cx, cy = 0.5, 0.5 # Center of image
    obj_x, obj_y = obj_pos

    dist_x = obj_x - cx
    dist_y = obj_y - cy

    head_curr_x = robot.joint_pos["head_pan"]
    head_curr_y = robot.joint_pos["head_tilt"]

    kp = 0.5
    new_head_x = head_curr_x + kp * -dist_x
    new_head_y = head_curr_y + kp * -dist_y

    robot.setJointPos(["head_tilt", "head_pan"], [new_head_y, new_head_x])

tickrate = 60
rate = rospy.Rate(tickrate)

first_look_direction = ''

currState = States.INIT
while not rospy.is_shutdown():
    if robot.buttonCheck("mode"):
        currState = States.INIT

    if robot.buttonCheck("user"):
        currState = States.PICK_BALL

    if DEBUG_MODE:
        cv2.imshow("Image", vision.img_buffer[-1])
        cv2.imshow("Func1", vision.debug_img[0])
        cv2.imshow("Func2", vision.debug_img[1])
        if vision.status[0]:
            print("Area 0: {}".format(vision.results[0][1]))
        if vision.status[1]:
            print("Area 1: {}".format(vision.results[1][1]))
            vision.debug_img[0]
        cv2.waitKey(1)

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
            currState = States.WALK_BACKWARDS
        
    elif currState == States.WALK_BACKWARDS:
        print("[WALK_BACKWARDS]")
        robot.walkVelocities(x=-13, y=0, balance=True, hip_pitch=4)
        rospy.sleep(5)
        tick_count = 0
        robot.walkStop()
        rospy.sleep(1)
        robot.walkStart()
        currState = States.FOCUS_BALL

    elif currState == States.FOCUS_BALL:
        print("[FOCUS_BALL]")
        # Retrieve results of first function
        status = vision.status[0]
        if status == True and vision.results[0][1] > MIN_AREA:
            tick_count += 1
            center, area = vision.results[0]
            center_head_on_object(center)
        else:
            tick_count = 0
            currState = States.FIND_BALL

    elif currState == States.FIND_BALL:
        print("[FIND_BALL]")
        center, area = vision.results[0]
        center_head_on_object(center)
        tilt_angle = robot.joint_pos["head_tilt"]
        pan_angle = robot.joint_pos["head_pan"]
        print(pan_angle)
        print(tilt_angle)
        if pan_angle > 0.1:
            theta = 5
        elif pan_angle < -0.1:
            theta = -5
        else:
            theta = 0

        robot.walkVelocities(x=0.6, th=theta, balance=True, hip_pitch=5)

        tick_count += 1

        if tilt_angle <= -0.9:
            robot.walkStop()
            # robot_weightlift.onlineWalkCommand(direction="turn_left", start_leg="right", step_num=1,
            #         front_length=0.4, step_angle=15,step_time=0.5)
            currState = States.PICK_BALL
 
    elif currState == States.PICK_BALL:
        rospy.loginfo("[PICK_BALL]")
        # TODO testing
        #rospy.sleep(2)
        center, area = vision.results[0]
        center_head_on_object(center)
        robot.setGeneralControlModule("none")
        rospy.sleep(2)
        robot.setGeneralControlModule("action_module")
        robot.playMotion(254, wait_for_end=True)
        rospy.sleep(1.0)
        currState = States.FIND_BASKET

    elif currState == States.FIND_BASKET:
        print("[FIND_BASKET]")
        # Move head to find the ball
        head_curr_x = robot.joint_pos["head_pan"]
        if direction == False:
            new_x = head_curr_x + HEAD_SEARCH_SPEED
        else:
            new_x = head_curr_x - HEAD_SEARCH_SPEED

        if new_x > 1.0:
            direction = True
        elif new_x < -1.0:
            direction = False

        robot.setJointPos(["head_pan", "head_tilt"], [new_x, 0.0])

        # Retrieve results of first function
        status = vision.status[1]
        if status == True and vision.results[1][1] > MIN_AREA:
            tick_count += 1
        else:
            tick_count = 0

        if tick_count > tickrate // 3:
            tick_count = 0

            currState = States.FOCUS_BASKET

    elif currState == States.FOCUS_BASKET:
        print("[FOCUS_BASKET]")
        # Retrieve results of first function
        status = vision.status[1]
        if status == True and vision.results[1][1] > MIN_AREA:
            tick_count += 1
            center, area = vision.results[1]
            center_head_on_object(center)
        else:
            tick_count = 0
            currState = States.FIND_BASKET

        print(area)
        print(tick_count)

        if tick_count > tickrate:
            # Transition
            tick_count = 0
            if robot.joint_pos["head_pan"] > -0.2 and robot.joint_pos["head_pan"] < 0.2:
                robot.walkStart()
                currState = States.WALK_TO_BASKET
            else:
                robot.walkStart()
                currState = States.FACE_BASKET

    elif currState == States.FACE_BASKET:
        print("[FACE_BALL]")
        center, area = vision.results[1]
        center_head_on_object(center)
        pan = robot.joint_pos["head_pan"]

        if pan > 0.05:
            th = 4.0
        elif pan < -0.05:
            th = -4.0
        else:
            th = 0.0
        robot.walkVelocities(x=-2.0, th=th)

        if pan > -0.2 and pan < 0.2:
            tick_count += 1

            if tick_count > tickrate:
                # Transition
                tick_count = 0

                currState = States.WALK_TO_BASKET

    elif currState == States.WALK_TO_BASKET:
        print("[WALK_TO_BASKET]")

        center, area = vision.results[1]
        center_head_on_object(center)
        pan = robot.joint_pos["head_pan"]

        print("Obj X: {}\t Area: {}\t Pan: {}".format(center[0], area, pan))

        if pan > 0.05:
            th = 4.0
        elif pan < -0.05:
            th = -4.0
        else:
            th = 0.0
        robot.walkVelocities(x=6.5, th=th)

        if area > BASKET_DUNK_SIZE:
            tick_count += 1

            if tick_count > tickrate//2:
                # Transition
                tick_count = 0

                robot.walkStop()

                # TODO: testing making head centered
                robot.setJointPos(["head_pan"], [0.0])

                robot.onlineWalkSetup(x=0.02, z=-0.045, foot_dist=0.08, foot_height=0.03)

                currState = States.WALK_SIDEWAYS_BASKET

    elif currState == States.WALK_SIDEWAYS_BASKET:
        print("[WALK_SIDEWAYS]")
        #mixer.music.stop()

        center, area = vision.results[1]
        detected = vision.status[1]

        print("X position of object in image: {} -- Y: {} -- Status: {}".format(center[0], center[1], detected))

        if center[0] > 0.14:
            robot.onlineWalkCommand(direction="right", start_leg="right", step_num=2,
                side_length=0.04, step_time=0.4)
            rospy.sleep(SIDE_STEP_TIME)
        elif center[0] < 0.08:
            robot.onlineWalkCommand(direction="left", start_leg="left", step_num=2,
                side_length=0.02, step_time=0.4)
            rospy.sleep(SIDE_STEP_TIME)
        else:
            tick_count += 1

            if tick_count > tickrate:
                # Transition
                currState = States.DUNK

    elif currState == States.DUNK:
        # I think this will stop the robot from "jerking" when changing the module
        #robot.online_walk_balance_pub.publish("balance_off")
        # TODO testing if robot doesnt jump anymore
        rospy.sleep(4)
        robot.setGeneralControlModule("none")
        rospy.sleep(1)
        robot.setGeneralControlModule("action_module")
        robot.playMotion(74)
        rospy.sleep(5.25)
        # mixer.music.load('/home/robotis/sound/slam_verse.mp3')
        # mixer.music.play()
        '''robot.setGrippersPos(left=50.0)'''

        rospy.sleep(5)

        currState = States.END

    elif currState == States.END:
        print("[END]")
        robot.walkStop()

    rate.sleep()
