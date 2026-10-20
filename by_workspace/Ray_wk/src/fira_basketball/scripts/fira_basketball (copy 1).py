#! /usr/bin/python

import rospy
import numpy as np
import cv2

from op3_ros_utils import getWalkingParams, Robot
from vision import *
from copy import copy
import sys

# Forgive me Sasuke
#from pygame import mixer

DEBUG_MODE = True # Show the detected image

MIN_AREA = 100 # Minimum area of objects to consider for vision
BALL_PICKUP_SIZE = 9000
BASKET_DUNK_SIZE = 52000
SIDE_STEP_TIME = 3.5
HEAD_SEARCH_SPEED = 0.065

class States:
    INIT = -1
    READY = 0 # Waits for start button
    FIND_BALL = 1 # Moves the head, looking for the ball
    WALK_BACKWARDS = -5 # Walk backwards for a bit
    FOCUS_BALL = 2 # Center heads on the ball, waits some time to make sure detection is corret
    FACE_BALL = 3 # Walk rotate in place to face the ball
    WALK_FORWARD = 4 # Walk towards the ball
    WALK_SIDEWAYS = 5 # Walk sideways, repositioning to grab the ball
    PICK_BALL = 6 # Picks the ball
    FIND_BASKET = 7 # Moves head, looking for the basket
    FOCUS_BASKET = 8 # Center heads on basket
    FACE_BASKET = 9 # Rotates in place to face the basket
    WALK_TO_BASKET = 10 # Rotates in place to face the basket
    WALK_SIDEWAYS_BASKET = 11
    DUNK = 12
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

    robot.setJointPos(["head_tilt"], [-0.5])

    # Play some intense music
    # mixer.init()
    # mixer.music.load('/home/robotis/sound/windowsXP_startup.mp3')
    # mixer.music.play()

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
        cv2.waitKey(1)

    if currState == States.INIT:
        print("[INIT]")
        init()

        # Transition
        tick_count = 0
        direction = False
        #currState = States.FOCUS_BASKET
        currState = States.READY

    elif currState == States.READY:
        print("[READY]")
        if robot.buttonCheck("start"):

            finished_cycle = False
            currState = States.FIND_BALL
    
    elif currState == States.FIND_BALL:
        print("[FIND_BALL]")
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
            finished_cycle = True

        robot.setJointPos(["head_pan", "head_tilt"], [new_x, -0.6])


        # Retrieve results of first function
        status = vision.status[0]
        if finished_cycle:
            tick_count = 0
            currState = States.WALK_BACKWARDS
        elif status == True and vision.results[0][1] > MIN_AREA:
            tick_count += 1
        else:
            tick_count = 0
            currState = States.FIND_BALL

        if tick_count > tickrate // 3:
            tick_count = 0
            currState = States.FOCUS_BALL

    elif currState == States.WALK_BACKWARDS:
            print("[WALK_BACKWARDS]")
            robot.walkStart()
            robot.walkVelocities(x=-6.5, th=0)
            rospy.sleep(3)
            robot.walkStop()
            direction = False
            finished_cycle = False
            # Go back to looking for the ball
            currState = States.FIND_BALL

    elif currState == States.FOCUS_BALL:
        print("[FOCUS_BALL]")
        # Retrieve results of first function
        status = vision.status[0]
        if status == True and vision.results[0][1] > MIN_AREA:
            tick_count += 1
            center, area = vision.results[0]
            center_head_on_object(center)
            print("[A]")
        else:
            tick_count = 0
            currState = States.FIND_BALL
            print("[B]")

        if tick_count > tickrate:
            # Transition
            tick_count = 0
            if robot.joint_pos["head_pan"] > -0.2 and robot.joint_pos["head_pan"] < 0.2:
                if area > 4400:
                    print("[C]")
                    robot.walkStart()
                    currState = States.WALK_FORWARD
                else:
                    if robot.joint_pos["head_pan"] > 0.0:
                        first_look_direction = 'left'
                    elif robot.joint_pos["head_pan"] <= 0.0:
                        first_look_direction = 'right'

                    robot.walkStart()
                    print("[D]")
                    currState = States.WALK_FORWARD
            else:
                robot.walkStart()
                print("[E]")
                currState = States.FACE_BALL

    elif currState == States.FACE_BALL:
        print("[FACE_BALL]")
        center, area = vision.results[0]
        center_head_on_object(center)
        pan = robot.joint_pos["head_pan"]

        if pan > 0.05:
            th = 4.0
        elif pan < -0.05:
            th = -4.0
        else:
            th = 0.0
        robot.walkVelocities(x=-5.0, th=th)

        if pan > -0.2 and pan < 0.2:
            tick_count += 1

            if tick_count > tickrate*1.5:
                # Transition
                tick_count = 0

                currState = States.WALK_FORWARD

    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD]")

        center, area = vision.results[0]
        center_head_on_object(center)
        pan = robot.joint_pos["head_pan"]
        tilt_angle = robot.joint_pos["head_tilt"]
        print("Obj X: {}\t Area: {}\t Pan: {}".format(center[0], area, pan))

        if pan > 0.05:
            th = 4.0
            yamp = -1.0
        elif pan < -0.05:
            th = -3.0
            yamp = 0
        else:
            th = 0.0
            yamp = 1.0
        robot.walkVelocities(x=3.0, y=yamp, th=th)

        if tilt_angle <= -0.89:
            tick_count += 1

            if tick_count > tickrate//2:
                # Transition
                tick_count = 0

                robot.walkStop()

                #robot.onlineWalkSetup(x=0.02, z=-0.045, foot_dist=0.08, foot_height=0.03)

                # TODO: testing making head centered
                robot.setJointPos(["head_pan"], [0.0])
                currState = States.PICK_BALL


    # elif currState == States.WALK_SIDEWAYS:
    #     print("[WALK_SIDEWAYS]")

    #     center, area = vision.results[0]
    #     detected = vision.status[0]

    #     print("X position of object in image: {} -- Y: {} -- Status: {}".format(center[0], center[1], detected))

    #     if center[0] > 0.29:
    #         tick_count = 0
    #         robot.onlineWalkCommand(direction="right", start_leg="right", step_num=2,
    #             side_length=0.04, step_time=0.4)
    #         #rospy.sleep(SIDE_STEP_TIME)
    #     elif center[0] < 0.205:
    #         tick_count = 0
    #         robot.onlineWalkCommand(direction="left", start_leg="left", step_num=2,
    #             side_length=0.02, step_time=0.4)
    #         #rospy.sleep(SIDE_STEP_TIME)
    #     else:
    #         tick_count += 1

    #         if tick_count > tickrate:
    #             # Transition
    #             currState = States.PICK_BALL

    elif currState == States.PICK_BALL:
        print("[PICK_BALL]")
        center, area = vision.results[0]

        # I think this will stop the robot from "jerking" when changing the module
        #robot.online_walk_balance_pub.publish("balance_off")
        # TODO testing jump thingie
        #rospy.sleep(3)
        robot.setGeneralControlModule("none")
        rospy.sleep(1)
        robot.setGeneralControlModule("action_module")
        robot.playMotion(255, wait_for_end=True)
        # robot.playMotion(72)
        rospy.sleep(0.3)
        '''robot.setGrippersPos(left=15.0)'''
        rospy.sleep(2)
        #robot.playMotion(71, wait_for_end=True)

        # Transition
        # Set ctrl module to walking, this actually only sets the legs
        robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch",
        "l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],
        ["walking_module"])
        # Set joint modules of head joints to none so we can control them directly
        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        
        robot.walkStart()
        robot.walkVelocities(x=-10, y=-5, th=0, hip_pitch=3)
        rospy.sleep(10.0)
        robot.walkStop()
        if first_look_direction == 'right':
            direction = True 
        elif first_look_direction == 'left':
            direction = False 
        else:
            print("Should never get here...")
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
        print("[FACE_BASKET]")
        center, area = vision.results[1]
        center_head_on_object(center)
        pan = robot.joint_pos["head_pan"]
        if pan > 0.05:
            th = 5.0
        elif pan < -0.05:
            th = -4.0
        else:
            th = 0.0
        robot.walkVelocities(x=-2.0, th=th, hip_pitch=5)
        robot.walkStop()
        rospy.sleep(2)
        robot.walkStart()
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
        robot.walkVelocities(x=5.5, th=th, hip_pitch=6)

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
        rospy.sleep(0.5)
        for i in range (0, 12):
            robot.onlineWalkCommand(direction="left", start_leg="left", step_num=2,
                    side_length=0.03, step_time=0.4)
            rospy.sleep(2.5)


        print("X position of object in image: {} -- Y: {} -- Status: {}".format(center[0], center[1], detected))

        # if center[0] > 0.14:
        #     robot.onlineWalkCommand(direction="right", start_leg="right", step_num=2,
        #         side_length=0.04, step_time=0.4)
        #     rospy.sleep(2.5)
        # elif center[0] < 0.08:
        #     robot.onlineWalkCommand(direction="left", start_leg="left", step_num=2,
        #         side_length=0.02, step_time=0.4)
        #     rospy.sleep(2.5)
        # else:
        #     tick_count += 1

        #     if tick_count > tickrate:
        #         # Transition
        #         currState = States.DUNK
        '''
         robot.onlineWalkCommand(direction="right", start_leg="right", step_num=2,
                  side_length=0.04, step_time=0.4)
         rospy.sleep(2.5)
         robot.onlineWalkCommand(direction="right", start_leg="right", step_num=2,
                  side_length=0.04, step_time=0.4)
         rospy.sleep(2.5)
         robot.onlineWalkCommand(direction="right", start_leg="right", step_num=2,
                  side_length=0.04, step_time=0.4)
         rospy.sleep(2.5)
         robot.onlineWalkCommand(direction="right", start_leg="right", step_num=2,
                  side_length=0.04, step_time=0.4)
         rospy.sleep(2.5)
         '''
        currState = States.DUNK
    
    elif currState == States.DUNK:
        # I think this will stop the robot from "jerking" when changing the module
        #robot.online_walk_balance_pub.publish("balance_off")
        # TODO testing if robot doesnt jump anymore
        rospy.sleep(1)
        robot.setGeneralControlModule("none")
        rospy.sleep(1)
        robot.setGeneralControlModule("action_module")
        robot.playMotion(251)
        rospy.sleep(5.25)
        # mixer.music.load('/home/robotis/sound/slam_verse.mp3')
        # mixer.music.play()
        '''robot.setGrippersPos(left=50.0)'''
        # rospy.sleep(1)
        # robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch",
        # "l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],
        # ["walking_module"])
        # robot.walkVelocities(x=-5, hip_pitch=3)
        # robot.walkStart()
        # rospy.sleep(2)
        # robot.walkStop()
        # rospy.sleep(3)
        currState = States.END

    elif currState == States.END:
        print("[END]")
        # robot.walkStop()

    rate.sleep()
