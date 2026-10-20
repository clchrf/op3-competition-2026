#! /usr/bin/python

import rospy
import numpy as np
import cv2

from op3_ros_utils2 import getWalkingParams, Robot
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
pitoff = 0

acc_time = 2.6#1
run_time = 6.5 
slow_time = 2.4
stop_time = 2 #1
preback_time = 2.5
####

# IMU yaw correction settings.
# z_gyro is stored in degrees. The first samples are used to estimate gyro bias
# while the robot is standing still during initialization.
GYRO_CALIBRATION_SAMPLES = 120
GYRO_DEADBAND = 0.015 # rad/s
YAW_THRESHOLD = 5.0 # degrees

z_gyro = 0.0
z_gyro_offset = 0.0 # corrected angular velocity, deg/s, only for display
z_gyro_bias = 0.0
z_gyro_bias_sum = 0.0
z_gyro_bias_count = 0
imu_last_time = None
imu_ready = False

def imu_get_yaw_by_integral(data):
    global z_gyro, z_gyro_offset, z_gyro_bias, z_gyro_bias_sum, z_gyro_bias_count, imu_last_time, imu_ready

    now = time.time()
    raw_z = data.angular_velocity.z

    if z_gyro_bias_count < GYRO_CALIBRATION_SAMPLES:
        z_gyro_bias_sum += raw_z
        z_gyro_bias_count += 1
        z_gyro_bias = z_gyro_bias_sum / z_gyro_bias_count
        imu_last_time = now
        if z_gyro_bias_count == GYRO_CALIBRATION_SAMPLES:
            imu_ready = True
            print("[IMU] calibration finished, gyro z bias = {:.6f} rad/s".format(z_gyro_bias))
        return

    if imu_last_time is None:
        imu_last_time = now
        return

    dt = now - imu_last_time
    imu_last_time = now
    if dt <= 0 or dt > 0.1:
        return

    corrected_z = raw_z - z_gyro_bias
    if abs(corrected_z) < GYRO_DEADBAND:
        corrected_z = 0.0

    z_gyro_offset = corrected_z * 180.0 / np.pi
    z_gyro += -corrected_z * dt * 180.0 / np.pi

def reset_imu_yaw():
    global z_gyro, z_gyro_offset, imu_last_time
    z_gyro = 0.0
    z_gyro_offset = 0.0
    imu_last_time = time.time()

def get_straight_theta(turn_angle):
    if not imu_ready:
        return 0
    if z_gyro > YAW_THRESHOLD:
        return turn_angle
    elif z_gyro < -YAW_THRESHOLD:
        return -turn_angle
    return 0

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
rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

# Create robot
robot = Robot()

rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages


def reset_to_ready():
    """Stop the sprint immediately and return to READY without killing the node.

    Press MODE during the race when a foul happens.
    After this reset, place the robot back at the start line and press START to start again.
    """
    global theta

    print("[RESET] MODE pressed: stop walking and return to READY")

    # Send zero walking params first, then stop the walking module.
    robot.walkVelocities(x=0, y=0, th=0, z_move_amplitude=0.01, balance=True, hip_pitch=8, feet_pitch=-1, para='4')
    robot.walkStop()
    rospy.sleep(0.25)

    # Keep the robot in walking mode and reset head pose for the next run.
    robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [0, -0.2])

    # Reset run-time correction variables.
    reset_imu_yaw()
    theta = 0
    rospy.sleep(0.25)

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
theta = 0
while not rospy.is_shutdown():
    print("Zgyro={:.2f} deg, gyro_rate={:.2f} deg/s, theta={}".format(z_gyro, z_gyro_offset, theta))

    # Emergency race reset: MODE stops the robot and returns to READY.
    # START can then start a fresh run without restarting this python node.
    if robot.buttonCheck("mode"):
        STEP_LEVEL = 0
        reset_to_ready()
        currState = States.READY
        rate.sleep()
        continue

    if currState == States.INIT:
        print("[INIT]")
        init()
        reset_imu_yaw()
        currState = States.READY

    elif currState == States.READY:
        print("[READY]")

        if robot.buttonCheck("start"):
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.2])
            reset_imu_yaw()
            theta = 0
            robot.walkVelocities(x=0, th=theta, z_move_amplitude=0.01, balance=True, hip_pitch=8, feet_pitch=-1, para='4')
            robot.walkStart()
            currState = States.ACCELERATE
            timestart = time.time()

            run_spot = timestart+acc_time
            slow_spot = run_spot+run_time
            stop_spot = slow_spot+slow_time
            preback_spot = stop_spot+stop_time
            back_spot = preback_spot+preback_time

    elif currState == States.ACCELERATE:
        print("[ACCELERATE]")
        theta = get_straight_theta(5)

        if time.time()<timestart+1.1:#1.6
            robot.walkVelocities(x=1, th=theta, z_move_amplitude=0.01, balance=True, hip_pitch=8, feet_pitch=-1, para='4')
        elif time.time()<timestart+1.8:#2.6
            robot.walkVelocities(x=15, th=theta, z_move_amplitude=0.04, balance=True, hip_pitch=8, feet_pitch=-1.8, para='4')
        elif time.time()<run_spot:
            robot.walkVelocities(x=35, th=theta, z_move_amplitude=0.04, balance=True, hip_pitch=8, feet_pitch=-1.9, para='4')
        else:
            currState = States.WALK_FORWARD

    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD]")
        theta = get_straight_theta(5)
        robot.walkVelocities(x=45, th=theta, z_move_amplitude=0.04, balance=True, hip_pitch=8.5, feet_pitch=-1.3, para='4')

        if time.time()>slow_spot:
            currState = States.WALK_SLOWLY_FORWARD

    elif currState == States.WALK_SLOWLY_FORWARD:
        print("[WALK_SLOWLY_FORWARD]")

        if time.time()<slow_spot+0.8:
            theta = get_straight_theta(4)
            robot.walkVelocities(x=20, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=-0.8+pitoff, para='4')#20 -1
        elif time.time()<slow_spot+1.6:
            theta = get_straight_theta(3)
            robot.walkVelocities(x=10, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=-0.6+pitoff, para='4')#20 -1
        elif time.time()<stop_spot:
            robot.walkVelocities(x=3, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=-0.5+pitoff, para='6')#20 -1
        elif time.time()<stop_spot+1.0:
            robot.walkVelocities(x=1, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=-0.5+pitoff, para='6')#20 -1
        elif time.time()<preback_spot:
            robot.walkVelocities(x=-1, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=-0.5+pitoff, para='2')#0 0
        else:
            currState = States.WALK_PREBACKWARDS
            
    elif currState == States.WALK_PREBACKWARDS:
        print("[WALK_PREBACKWARDS]")
        if time.time()<preback_spot+0.9:
            robot.walkVelocities(x=-10, y=0, th=0, balance=True, hip_pitch=8, feet_pitch=-0.4+pitoff, para='2')
        elif time.time()<preback_spot+1.8:
            robot.walkVelocities(x=-15, y=0, th=0, balance=True, hip_pitch=8, feet_pitch=0+pitoff, para='2')
        elif time.time()<back_spot:
            theta = get_straight_theta(3)
            robot.walkVelocities(x=-20, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=0.2+pitoff, para='2')
        else:
            currState = States.WALK_BACKWARDS

    elif currState == States.WALK_BACKWARDS:
        print("[WALK_BACKWARDS]")
        theta = get_straight_theta(6)
        robot.walkVelocities(x=-30, y=0, th=theta, balance=True, hip_pitch=8, feet_pitch=0.5+pitoff, para='2') #-45 yamp 0.6  #-25

    elif currState == States.END:
        print("[END]")
        robot.walkVelocities(x=0, th=theta, z_move_amplitude=0.01, balance=True, hip_pitch=8, feet_pitch=-1, para='4')
        robot.walkStop()
        break

    rate.sleep()