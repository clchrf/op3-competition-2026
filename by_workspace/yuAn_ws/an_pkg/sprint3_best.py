#! /usr/bin/python

import rospy
import numpy as np
import cv2

from op3_ros_utils1 import getWalkingParams, Robot
# from vision import *
from copy import copy
import sys
import time
from sensor_msgs.msg import Imu


DEBUG_MODE = True # Show the detected image
MIN_AREA = 1000
SLOWDOWN_AREA = 40000
CROSS_AREA = 45000

angoff = 5#5

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0
def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    # print("aa")
    z_gyro = z_gyro - angular_velocity.z/2 + z_gyro_offset_for_caculate
    # z_gyro = angular_velocity.z/2
    # print("z_gyro_offset_for_caculate:",z_gyro_offset_for_caculate)
    z_gyro_offset = angular_velocity.z/2

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

class States:
    INIT = -1
    READY = 0 # Waits for start button
    ACCELERATE = 1
    WALK_FORWARD = 2
    WALK_SLOWLY_FORWARD = 3
    STOP = 4
    WALK_PREBACKWARDS = 5
    WALK_BACKWARDS = 6
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

    # Call initial robot position
    robot.playMotion(1, wait_for_end=True)

    # Set ctrl module to walking, this actually only sets the legs
    robot.setGeneralControlModule("walking_module")
    
    # Set joint modules of head joints to none so we can control them directly
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])

    robot.setJointPos(["head_tilt"], [-0.2])

    rospy.sleep(1.0)

tickrate = 60
rate = rospy.Rate(tickrate)
time.time()

# TODO remember to put to 0
STEP_LEVEL = 0

currState = States.INIT
z_gyro = 0
z_gyro_offset = 0
theta = 0
while not rospy.is_shutdown():
    print("Zgyro, offset, theta")
    print(z_gyro)
    print(z_gyro_offset)
    print(theta)
    if robot.buttonCheck("mode"):
        STEP_LEVEL = 0
        currState = States.END  #INIT

    if currState == States.INIT:
        print("[INIT]")
        init()
        
        z_gyro = 0
        z_gyro_offset = 0
        currState = States.READY

    elif currState == States.READY:
        print("[READY]")
        
        if robot.buttonCheck("start"):
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.2])
            robot.walkVelocities(x=15, th=theta, balance=True, hip_pitch=8, para = '1')
            robot.walkStart()
            currState = States.ACCELERATE
            timestart = time.time()
            z_gyro = 0
            z_gyro_offset = 0

    elif currState == States.ACCELERATE:
        print("[ACCELERATE]")

        robot.walkVelocities(x=15, th=theta, balance=True, hip_pitch=8, feet_pitch=-3, para = '1') #15

        if time.time()>timestart+1:
            currState = States.WALK_FORWARD

    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD]")
        if z_gyro > 5:
            theta = 5
        elif z_gyro < -5:
            theta = -5
        else:
            theta = 0

        robot.walkVelocities(x=55, th=theta, balance=True, hip_pitch=8, feet_pitch=-2.3, para = '1') #55

        if time.time()>timestart+7:
            currState = States.WALK_SLOWLY_FORWARD


    elif currState == States.WALK_SLOWLY_FORWARD:
        print("[WALK_SLOWLY_FORWARD]")
        if z_gyro > 5-angoff:
            theta = 5
        elif z_gyro < -5-angoff:
            theta = -5
        else:
            theta = 0
        
        if time.time()<timestart+8:
            robot.walkVelocities(x=20, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=-1 , para = '1')#20
        elif time.time()<timestart+8.8:
            robot.walkVelocities(x=0, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=0 , para = '1')#0
        else:
            currState = States.WALK_PREBACKWARDS

    # elif currState == States.STOP:
    #     print("[STOP]")
    #     # robot.walkStop

    #     if time.time()>timestart+9:
    #         currState = States.WALK_PREBACKWARDS
    #         robot.walkStart

    elif currState == States.WALK_PREBACKWARDS:
        print("[WALK_PREBACKWARDS]")
        if z_gyro > 5-angoff:
            theta = 6
        elif z_gyro < -5-angoff:
            theta = -6
        else:
            theta = 0
        robot.walkVelocities(x=-20, y=0,th = theta, balance=True, hip_pitch=8, feet_pitch=0.2, para = '2') #-20 yamp -0.5
        # rospy.sleep(1.0)
        if time.time()>timestart+10:
            currState = States.WALK_BACKWARDS
            robot.walkStart()
            angoff+=5
    
    elif currState == States.WALK_BACKWARDS:
        print("[WALK_BACKWARDS]")
        if z_gyro > 5-angoff:
            theta = 6
        elif z_gyro < -5-angoff:
            theta = -6
        else:
            theta = 0

        robot.walkVelocities(x=-45, y=0, th = theta, balance=True, hip_pitch=8, feet_pitch=0.5, para = '2') #-45 yamp -0.5

    elif currState == States.END:
        print("[END]")
        robot.walkStop()
        break

    rate.sleep()
