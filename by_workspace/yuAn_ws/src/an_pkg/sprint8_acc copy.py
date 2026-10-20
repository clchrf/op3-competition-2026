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


# DEBUG_MODE = True # Show the detected image
# MIN_AREA = 1000
# SLOWDOWN_AREA = 40000
# CROSS_AREA = 45000

####adjust mode
angoff = 7 #5
pitoff =  0.4#-0.5

acc_time = 2.5#2.6
run_time = 6.2 #8.5#6.5
slow_time = 1
stop_time = 0.5#0.8
preback_time = 4.5#2
####

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0
def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    z_gyro = z_gyro - angular_velocity.z/2 + z_gyro_offset_for_caculate
    # z_gyro = angular_velocity.z/2
    z_gyro_offset = angular_velocity.z/2

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

class States:
    INIT = -1
    READY = 0 # Waits for start button
    ACCELERATE = 3.3
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
    if robot.buttonCheck("user"):#mode
        STEP_LEVEL = 0
        currState = States.END  #INIT END

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
            robot.walkVelocities(x=0, th=theta, z_move_amplitude=0.01, balance=True, hip_pitch=8, feet_pitch=-1+pitoff, para= '4')
            robot.walkStart()
            currState = States.ACCELERATE
            timestart = time.time()
            z_gyro = 0
            z_gyro_offset = 0

            run_spot = timestart+acc_time
            slow_spot = run_spot+run_time
            stop_spot = slow_spot+slow_time
            preback_spot = stop_spot+stop_time
            back_spot = preback_spot+preback_time
            tp = 0
            fh = 0
            spd = 0
            fp = 0
            hp = 0

    elif currState == States.ACCELERATE:
        print("[ACCELERATE]")
        if z_gyro > 5:
            theta = 5
        elif z_gyro < -5:
            theta = -5
        else:
            theta = 0
        
        tp = time.time()-timestart
        spd = tp*18
        fh = tp*0.012+0.01

        robot.walkVelocities(x=spd, th=theta, z_move_amplitude=fh, balance=True, hip_pitch=8, feet_pitch=-1.6+pitoff, para= '4')

        if time.time()-timestart>2.5:#run_spot:
            currState = States.WALK_FORWARD

    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD]")
        if z_gyro > 5:
            theta = 5
        elif z_gyro < -5:
            theta = -5
        else:
            theta = 0

        robot.walkVelocities(x=45, th=theta, z_move_amplitude=0.04, balance=True, hip_pitch=8.5, feet_pitch=-1.3+pitoff, para= '4')#8.5 -1.3

        if time.time()>slow_spot:
            currState = States.WALK_SLOWLY_FORWARD


    elif currState == States.WALK_SLOWLY_FORWARD:
        print("[WALK_SLOWLY_FORWARD]")
        if z_gyro > 5-angoff:
            theta = 5
        elif z_gyro < -5-angoff:
            theta = -5
        else:
            theta = 0
        tp = time.time()-slow_spot
        spd = 45-30*tp
        # fp = -0.9+1.2*tp#0.6*
        fp = -1.3+tp*0.7
        zm = 0.041+tp*0.002

        robot.walkVelocities(x=spd, y=0, th=theta, balance=True, z_move_amplitude=zm, hip_pitch=8, feet_pitch=fp+pitoff , para = '4')#20 -1

        if time.time()>preback_spot:
            currState = States.WALK_PREBACKWARDS

        # if time.time()<stop_spot:
        #     robot.walkVelocities(x=20, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=-0.5+pitoff , para = '4')#20 -1
        # elif time.time()<preback_spot:
        #     robot.walkVelocities(x=0, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=0+pitoff , para = '4')#0 0
        # else:
        #     currState = States.WALK_PREBACKWARDS

    elif currState == States.WALK_PREBACKWARDS:
        print("[WALK_PREBACKWARDS]")
        if z_gyro > 5-angoff:
            theta = 7
        elif z_gyro < -5-angoff:
            theta = -7
        else:
            theta = 0
        tp = time.time()-preback_spot
        spd = -8*tp-4
        # hp=8-tp

        # fh = tp*0.007+0.00185
        # spd = -tp*8-4
        robot.walkVelocities(x=spd, y=0,th = theta, balance=True, hip_pitch=8, feet_pitch=4-+pitoff, para = '5') #-20 yamp 0.5
        
        if time.time()>back_spot:
            currState = States.WALK_BACKWARDS
            robot.walkStart()
            angoff+=4.5
    
    elif currState == States.WALK_BACKWARDS:
        print("[WALK_BACKWARDS]")
        if z_gyro > 5-angoff:
            theta = 6
        elif z_gyro < -5-angoff:
            theta = -6
        else:
            theta = 0

        robot.walkVelocities(x=-45, y=0, th = theta, balance=True, hip_pitch=8, feet_pitch=3.2+pitoff, para = '5') #-45 yamp 0.6
        if(abs(z_gyro > 20)):
            currState = States.END

    elif currState == States.END:
        print("[END]")
        robot.walkVelocities(x=0, th=theta, z_move_amplitude=0.01, balance=True, hip_pitch=8, feet_pitch=-1, para= '4')
        robot.walkStop()
        print(time.time()-timestart)
        currState = States.INIT
        #break

    rate.sleep()
