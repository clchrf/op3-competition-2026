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

#### adjust mode
angoff = 7  # 5
pitoff = 0.4  # -0.5

acc_time = 2.5   # 2.6
run_time = 5         #7.5   # reduced a bit from 8.0
slow_time = 2
stop_time = 0.5  # 0.8
preback_time = 4.5  # 2
####

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0


def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    z_gyro = z_gyro - angular_velocity.z / 2 + z_gyro_offset_for_caculate
    # z_gyro = angular_velocity.z / 2
    z_gyro_offset = angular_velocity.z / 2


rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)


class States:
    INIT = -1
    READY = 0  # Waits for start button
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

rospy.sleep(3)  # Make sure every publisher has registered to their topic,
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
timestart = None
backward_start_time = None

last_log_time = 0.0
last_log_state = ""


def log_state(name, force=False):
    global last_log_time
    global last_log_state

    now = time.time()
    if force or name != last_log_state or (now - last_log_time) >= 0.10:
        last_log_time = now
        last_log_state = name
        print("[{}] Zgyro={:.3f}, offset={:.3f}, theta={:.3f}".format(
            name, z_gyro, z_gyro_offset, theta
        ))
        sys.stdout.flush()


while not rospy.is_shutdown():
    if robot.buttonCheck("mode"):  # mode
        STEP_LEVEL = 0
        currState = States.END  # INIT END

    if currState == States.INIT:
        log_state("INIT", force=True)
        init()

        z_gyro = 0
        z_gyro_offset = 0
        theta = 0
        currState = States.READY

    elif currState == States.READY:
        log_state("READY")

        if robot.buttonCheck("user"):
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.2])
            robot.walkVelocities(
                x=0,
                th=theta,
                z_move_amplitude=0.00,
                balance=True,
                hip_pitch=8,
                feet_pitch=-1.6 + pitoff,  # keep consistent with ACCELERATE
                para='4'
            )
            robot.walkStart()
            currState = States.ACCELERATE
            timestart = time.time()
            z_gyro = 0
            z_gyro_offset = 0
            theta = 0
            angoff = 7

            run_spot = timestart + acc_time
            slow_spot = run_spot + run_time
            stop_spot = slow_spot + slow_time
            preback_spot = stop_spot + stop_time
            back_spot = preback_spot + preback_time
            tp = 0
            fh = 0
            spd = 0
            fp = 0
            hp = 0

            log_state("ACCELERATE", force=True)

    elif currState == States.ACCELERATE:
        theta = np.clip(0.18 * z_gyro, -2.0, 2.0)

        log_state("ACCELERATE")

        tp = time.time() - timestart  # about 2.5s
        spd = tp * 18
        fh = tp * 0.014

        robot.walkVelocities(
            x=spd,
            th=theta,
            z_move_amplitude=fh,
            balance=True,
            hip_pitch=8,
            feet_pitch=-1.6 + pitoff,
            para='4'
        )

        if time.time() - timestart > acc_time:
            currState = States.WALK_FORWARD
            log_state("WALK_FORWARD", force=True)

    elif currState == States.WALK_FORWARD:
        theta = np.clip(0.18 * z_gyro, -2.0, 2.0)

        log_state("WALK_FORWARD")

        robot.walkVelocities(
            x=45,
            th=theta,
            z_move_amplitude=0.035,
            balance=True,
            hip_pitch=8.5,
            feet_pitch=-1.3 + pitoff,
            para='4'
        )

        if time.time() > slow_spot:
            currState = States.WALK_SLOWLY_FORWARD
            log_state("WALK_SLOWLY_FORWARD", force=True)

    elif currState == States.WALK_SLOWLY_FORWARD:
        theta = np.clip(0.18 * z_gyro, -2.0, 2.0)

        log_state("WALK_SLOWLY_FORWARD")

        tp = time.time() - slow_spot
        total_slow = slow_time + stop_time
        spd = 45 - (45 / total_slow) * tp
        fp = -1.3 + tp * 0.7
        zm = 0.041 + tp * 0.002

        robot.walkVelocities(
            x=spd,
            y=0,
            th=theta,
            balance=True,
            z_move_amplitude=zm,
            hip_pitch=8,
            feet_pitch=fp + pitoff,
            para='4'
        )

        if time.time() > preback_spot:
            currState = States.WALK_PREBACKWARDS
            log_state("WALK_PREBACKWARDS", force=True)

    elif currState == States.WALK_PREBACKWARDS:
        theta_back = np.clip(0.10 * z_gyro, -1.5, 1.5)
        theta = theta_back

        log_state("WALK_PREBACKWARDS")

        tp = time.time() - preback_spot
        blend_t = 0.7

        start_fp = 0.85
        fast_fp = 4.0 + pitoff   # = 4.4
        end_spd = -37.0

        if tp < blend_t:
            a = tp / blend_t
            spd = -4.0 * a
            fpb = start_fp + (fast_fp - start_fp) * a
        else:
            remain_t = preback_time - blend_t
            a = tp - blend_t
            spd = -4.0 - ((abs(end_spd) - 4.0) / remain_t) * a
            fpb = fast_fp

        robot.walkVelocities(
            x=spd,
            y=0,
            th=theta,
            balance=True,
            hip_pitch=8,
            feet_pitch=fpb,
            para='5'
        )

        if time.time() > back_spot:
            currState = States.WALK_BACKWARDS
            backward_start_time = time.time()
            robot.walkStart()
            angoff += 4.5
            log_state("WALK_BACKWARDS", force=True)

    elif currState == States.WALK_BACKWARDS:
        theta_back = np.clip(0.10 * z_gyro, -1.5, 1.5)
        theta = theta_back

        log_state("WALK_BACKWARDS")

        robot.walkVelocities(
            x=-37,
            y=0,
            th=theta,
            balance=True,
            hip_pitch=8,
            feet_pitch=3.2 + pitoff,
            para='5'
        )

        if backward_start_time is not None and (time.time() - backward_start_time) >= 6:  #10
            currState = States.END
            log_state("END", force=True)

    elif currState == States.END:
        log_state("END")

        robot.walkVelocities(
            x=0,
            th=theta,
            z_move_amplitude=0.01,
            balance=True,
            hip_pitch=8,
            feet_pitch=-1,
            para='4'
        )
        robot.walkStop()

        if timestart is not None:
            print("[END] total_time={:.3f}".format(time.time() - timestart))
            sys.stdout.flush()

        currState = States.INIT

    rate.sleep()