#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import cv2
import rospy
import numpy as np
from collections import deque
from cv_bridge import CvBridge, CvBridgeError
from op3_ros_utils import Robot
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32, Int8
from tf.transformations import euler_from_quaternion
from fira_basketball.msg import ThreeDouble


# ============================================================
# Hybrid obstacle control
# Basketball-style scan / turn / approach
# + Weightlifting-style Adjust_Mode pulse correction
# ============================================================

DEBUG_MODE = False

camera_ang = 60
camera_h = 44.0
hfov = 80.54
vfov = 52.1
hfov_rad = np.radians(hfov)
vfov_rad = np.radians(vfov)

IMU_RECORD = True
z_gyro = 0.0
pitch = 0.0
roll = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0
argsIMU = [68, True]

directionx = 0.0
directiony = 0.0
find_the_path = 0.0
closest_obs_z = 1e6
closest_z_buf = deque(maxlen=3)
obs_seen = False
last_seen_time = 0.0
SEEN_TIMEOUT = 1.5
oimg = None

args_red = ((np.array([0, 50, 37]), np.array([10, 255, 255])),
            (np.array([170, 50, 126]), np.array([179, 255, 255])))

# ============================================================
# Main obstacle parameters
# ============================================================
STOP_DIST = 50.0
NO_PATH_TH = 0.5
NO_PATH_STOP_DIST = 75.0

# Basketball approach-style normal forward walking
CRUISE_X = 3.6
CRUISE_Z = 0.022
CRUISE_Z_OFFSET = 0.030
CRUISE_HIP_PITCH = 7.0
PATH_Y_GAIN = 4.0
PATH_Y_CLAMP = 4.5
HEADING_KP = 0.05
HEADING_TH_CLAMP = 1.0

# Basketball-style stable turn
# Do NOT add WALK_TH_ZERO_OFFSET here. This file uses op3_ros_utils, not op3_ros_utils1.
TURN_SIGN = -1.0
OBS_TURN_TH = 1.7
OBS_TURN_TIME = 1.8
OBS_TURN_Z_MOVE_AMPLITUDE = 0.014
OBS_TURN_Z_OFFSET = 0.034
OBS_TURN_HIP_PITCH = 5.0
OBS_TURN_SETTLE = 0.90

# Weightlifting-style Adjust_Mode pulse
OBS_ADJ_SAMPLE_FRAMES = 7
OBS_ADJ_SAMPLE_DT = 0.05
OBS_ADJ_MAX_ITER = 5
OBS_ADJ_TURN_TH = 2.2
OBS_ADJ_TURN_TIME = 0.65
OBS_ADJ_SIDE_Y = 0.8
OBS_ADJ_Y_SIGN = 1.0
OBS_ADJ_Z_AMP = 0.018
OBS_ADJ_SETTLE = 0.90

# Passing motion. No rotate-back: pass, reset current heading, then keep going forward.
OBS_PASS_X = 2.8
OBS_PASS_Y = 11.0
OBS_PASS_TIME = 4.3
OBS_PASS_CLOSE_Z = 30.0
OBS_PASS_EMERGENCY_Z = 8.0
OBS_PASS_Z_MOVE_AMPLITUDE = 0.022
OBS_PASS_Z_OFFSET = 0.030
OBS_PASS_HIP_PITCH = 7.0

SCAN_ANGLES = [35, -35, 55, -55]
SCAN_WAIT = 0.90
SCAN_MIN_ANGLE = 20
SCAN_SCORE_MIN = -50.0

HEAD_PAN_SLEW = 4.0
_head_pan_cmd = 0.0

scan_index = 0
scan_start_time = None
scan_best_score = -999.0
scan_best_angle = 0
scan_best_dirx = 0.0
scan_best_diry = 0.0
scan_best_z = 1e6

obs_adjust_iter = 0
obs_adjust_turn_dir = 1
pass_start_time = None
last_turn_dir = 1


class States:
    INIT = -1
    READY = 0
    OBSTACLE_FORWARD = 1
    OBSTACLE_SCAN_PATH = 2
    OBSTACLE_ADJUST = 3
    OBSTACLE_PASS = 4
    END = 99


LEG_JOINTS = [
    "r_hip_yaw", "l_hip_yaw",
    "r_hip_roll", "l_hip_roll",
    "r_hip_pitch", "l_hip_pitch",
    "r_knee", "l_knee",
    "r_ank_pitch", "l_ank_pitch",
    "r_ank_roll", "l_ank_roll"
]


rospy.init_node("hybrid_obstacle_control")
robot = Robot()
rospy.sleep(2)


def read_IMU(data):
    global z_gyro, pitch, roll, z_gyro_offset, argsIMU

    if not IMU_RECORD:
        return

    angular_velocity = data.angular_velocity
    orientation = data.orientation
    orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
    roll, pitch, _ = euler_from_quaternion(orientation_list)

    pitch = np.degrees(pitch)
    roll = np.degrees(roll)

    z_gyro = z_gyro - angular_velocity.z / 2.0 + z_gyro_offset_for_caculate
    z_gyro_offset = angular_velocity.z / 2.0

    try:
        argsIMU[0] = pitch - np.degrees(robot.joint_pos["head_tilt"])
    except Exception:
        pass


def read_obstacle_vision_data(data):
    global directionx, directiony, find_the_path
    directionx = data.first
    directiony = data.second
    find_the_path = data.third


def read_closest_z(data):
    global closest_obs_z, obs_seen, last_seen_time
    v = float(data.data)

    if 0 < v < 10000:
        closest_z_buf.append(v)
        closest_obs_z = min(closest_z_buf)
        obs_seen = True
        last_seen_time = time.time()
    else:
        if time.time() - last_seen_time > SEEN_TIMEOUT:
            obs_seen = False
            closest_obs_z = 1e6
            closest_z_buf.clear()


def get_img(data):
    global oimg
    bridge = CvBridge()
    try:
        oimg = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))


def reset_imu():
    global z_gyro_offset_for_caculate, z_gyro
    z_gyro_offset_for_caculate = z_gyro_offset
    z_gyro = 0.0


def init_pose():
    robot.walkStop()
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [0, np.radians(pitch - camera_ang)])

    try:
        robot.setJointsControlModule(["l_sho_pitch", "l_sho_roll", "l_el"], ["none", "none", "none"])
        robot.setJointPos(["l_sho_pitch", "l_sho_roll", "l_el"], [0.2, 1.5, 0])
        robot.setJointsControlModule(["r_sho_pitch", "r_sho_roll", "r_el"], ["none", "none", "none"])
        robot.setJointPos(["r_sho_pitch", "r_sho_roll", "r_el"], [-0.2, -1.5, 0])
    except Exception:
        pass

    robot.setJointsControlModule(LEG_JOINTS, ["walking_module"] * len(LEG_JOINTS))
    robot.walkStop()


def set_head_pan_smooth(target_deg):
    global _head_pan_cmd

    target_deg = float(np.clip(target_deg, -60, 60))
    delta = float(np.clip(target_deg - _head_pan_cmd, -HEAD_PAN_SLEW, HEAD_PAN_SLEW))
    _head_pan_cmd += delta
    robot.setJointPos(["head_pan"], [np.radians(_head_pan_cmd)])
    return abs(target_deg - _head_pan_cmd) < 2.0


def set_head_to_forward():
    set_head_pan_smooth(np.clip(z_gyro, -60, 60))


def see_the_door():
    global oimg

    if oimg is None:
        return False

    img_copy = oimg.copy()
    lower1 = args_red[0][0]
    upper1 = args_red[0][1]
    lower2 = args_red[1][0]
    upper2 = args_red[1][1]

    hsv_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_img, lower1, upper1)
    mask2 = cv2.inRange(hsv_img, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    try:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except ValueError:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    for contour in contours:
        largest_area = max(largest_area, cv2.contourArea(contour))

    return largest_area > 20000


def obstacle_turn_pulse(th_cmd, duration):
    # Basketball-style stable body turn: ramp up th, then stop and re-sample.
    robot.walkVelocities(
        x=0.0,
        y=0.0,
        th=th_cmd * 0.45,
        z_move_amplitude=OBS_TURN_Z_MOVE_AMPLITUDE,
        balance=True,
        z_offset=OBS_TURN_Z_OFFSET,
        hip_pitch=OBS_TURN_HIP_PITCH
    )

    robot.walkStart()
    rospy.sleep(0.25)

    t0 = time.time()
    while time.time() - t0 < duration and not rospy.is_shutdown():
        elapsed = time.time() - t0

        if elapsed < 0.5:
            gain = 0.45 + 0.55 * (elapsed / 0.5)
        else:
            gain = 1.0

        robot.walkVelocities(
            x=0.0,
            y=0.0,
            th=th_cmd * gain,
            z_move_amplitude=OBS_TURN_Z_MOVE_AMPLITUDE,
            balance=True,
            z_offset=OBS_TURN_Z_OFFSET,
            hip_pitch=OBS_TURN_HIP_PITCH
        )
        rospy.sleep(0.05)

    robot.walkStop()
    rospy.sleep(OBS_TURN_SETTLE)


def sample_obstacle_stable(n=OBS_ADJ_SAMPLE_FRAMES, dt=OBS_ADJ_SAMPLE_DT):
    # Weightlifting-style multi-frame median sampling.
    dxs, dys, fs, zs = [], [], [], []

    for _ in range(n):
        dxs.append(float(directionx))
        dys.append(float(directiony))
        fs.append(float(find_the_path))

        if obs_seen:
            zs.append(float(closest_obs_z))
        else:
            zs.append(200.0)

        rospy.sleep(dt)

    return float(np.median(dxs)), float(np.median(dys)), float(np.median(fs)), float(np.median(zs))


def obstacle_adjust_pulse(turn_dir):
    # Weightlifting Adjust_Mode style: one short pulse -> stop -> settle -> sample again.
    th_cmd = TURN_SIGN * float(turn_dir) * OBS_ADJ_TURN_TH
    y_cmd = OBS_ADJ_Y_SIGN * float(turn_dir) * OBS_ADJ_SIDE_Y

    print("[OBS_ADJ_PULSE] turn_dir:", turn_dir,
          "x:", 0.0,
          "y:", round(y_cmd, 2),
          "th:", round(th_cmd, 2),
          "time:", OBS_ADJ_TURN_TIME)

    robot.walkStart()
    t0 = time.time()

    while time.time() - t0 < OBS_ADJ_TURN_TIME and not rospy.is_shutdown():
        robot.walkVelocities(
            x=0.0,
            y=y_cmd,
            th=th_cmd,
            z_move_amplitude=OBS_ADJ_Z_AMP,
            balance=True,
            z_offset=0.0
        )
        rate.sleep()

    robot.walkStop()
    rospy.sleep(OBS_ADJ_SETTLE)


def score_scan_sample(angle_deg, dx, dy, find_value, z_value):
    if abs(angle_deg) < SCAN_MIN_ANGLE:
        return -100.0

    z_seen = z_value
    if z_seen > 9000:
        z_seen = 200.0

    # Prefer wider/safer direction. Path existence is useful, but distance matters more.
    return z_seen + 20.0 * find_value + 5.0 * max(dx, 0.0) - 0.08 * abs(angle_deg)


rospy.Subscriber("/cv_camera/image_raw", Image, get_img, queue_size=1)
rospy.Subscriber("/robotis/open_cr/imu", Imu, read_IMU, queue_size=1)
rospy.Subscriber("obstacle_vision_data", ThreeDouble, read_obstacle_vision_data, queue_size=1)
rospy.Subscriber("obstacle_closest_z", Float32, read_closest_z, queue_size=1)
pub_head = rospy.Publisher("obstacle_head_angle", Int8, queue_size=1)

rospy.sleep(1.0)

tickrate = 60
rate = rospy.Rate(tickrate)
currState = States.INIT


while not rospy.is_shutdown():

    if robot.buttonCheck("mode"):
        currState = States.INIT

    try:
        pub_head.publish(int(round(argsIMU[0])))
    except Exception:
        pass

    if currState == States.INIT:
        print("[INIT]")
        pub_head.publish(camera_ang)
        init_pose()
        rospy.sleep(2.0)
        reset_imu()
        currState = States.READY
        print("[INIT_FINISH]")

    elif currState == States.READY:
        if robot.buttonCheck("start"):
            reset_imu()
            robot.walkStart()
            currState = States.OBSTACLE_FORWARD
            print("[START] -> OBSTACLE_FORWARD")

    elif currState == States.OBSTACLE_FORWARD:
        set_head_to_forward()

        if see_the_door():
            robot.walkStop()
            print("[DOOR_SEEN] obstacle stage finished -> END")
            currState = States.END
            continue

        # Stop before the obstacle and enter basketball-style scan.
        near_obstacle = obs_seen and closest_obs_z < STOP_DIST
        near_no_path = obs_seen and find_the_path < NO_PATH_TH and closest_obs_z < NO_PATH_STOP_DIST
        if find_the_path < NO_PATH_TH or (obs_seen and closest_obs_z < STOP_DIST):
            robot.walkStop()
            rospy.sleep(0.30)

            scan_index = 0
            scan_start_time = None
            scan_best_score = -999.0
            scan_best_angle = 0
            scan_best_dirx = directionx
            scan_best_diry = directiony
            scan_best_z = closest_obs_z
            obs_adjust_iter = 0

            print("[OBS_SCAN_START] z:", closest_obs_z,
                  "dirx:", directionx,
                  "diry:", directiony,
                  "find:", find_the_path)

            currState = States.OBSTACLE_SCAN_PATH
            continue

        # Basketball approach-style forward walking: smooth, small z_amp, continuous visual y correction.
        z_front = closest_obs_z if obs_seen else 200.0
        speed_scale = float(np.clip((z_front - STOP_DIST) / 90.0, 0.35, 1.0))

        x_cmd = np.clip(max(directionx, 0.25), 0.25, 1.0) * CRUISE_X * speed_scale
        y_cmd = np.clip(directiony * PATH_Y_GAIN, -PATH_Y_CLAMP, PATH_Y_CLAMP)
        th_cmd = np.clip(z_gyro * HEADING_KP, -HEADING_TH_CLAMP, HEADING_TH_CLAMP)

        robot.walkVelocities(
            x=x_cmd,
            y=y_cmd,
            th=th_cmd,
            z_move_amplitude=CRUISE_Z,
            balance=True,
            z_offset=CRUISE_Z_OFFSET,
            hip_pitch=CRUISE_HIP_PITCH
        )

    elif currState == States.OBSTACLE_SCAN_PATH:
        robot.walkStop()

        if scan_index >= len(SCAN_ANGLES):
            set_head_to_forward()

            if scan_best_score < SCAN_SCORE_MIN:
                obs_adjust_turn_dir = last_turn_dir
                print("[SCAN_WEAK] score:", scan_best_score,
                      "use last_turn_dir:", obs_adjust_turn_dir)
            else:
                obs_adjust_turn_dir = 1 if scan_best_angle > 0 else -1
                last_turn_dir = obs_adjust_turn_dir

            turn_cmd = TURN_SIGN * float(obs_adjust_turn_dir) * OBS_TURN_TH

            print("[OBS_TURN_TO_SIDE] best_angle:", scan_best_angle,
                  "turn_dir:", obs_adjust_turn_dir,
                  "turn_cmd:", round(turn_cmd, 2),
                  "best_z:", scan_best_z,
                  "score:", round(scan_best_score, 2))

            obstacle_turn_pulse(turn_cmd, OBS_TURN_TIME)

            # Important: after turning, define the current heading as the new forward direction.
            reset_imu()

            obs_adjust_iter = 0
            currState = States.OBSTACLE_ADJUST
            continue

        reached = set_head_pan_smooth(SCAN_ANGLES[scan_index])
        if not reached:
            scan_start_time = None
            rate.sleep()
            continue

        if scan_start_time is None:
            scan_start_time = time.time()

        if time.time() - scan_start_time >= SCAN_WAIT:
            angle_now = SCAN_ANGLES[scan_index]
            dx_now, dy_now, find_now, z_now = sample_obstacle_stable()
            score = score_scan_sample(angle_now, dx_now, dy_now, find_now, z_now)

            print("[SCAN_CHECK] angle:", angle_now,
                  "dirx:", round(dx_now, 2),
                  "diry:", round(dy_now, 2),
                  "find:", round(find_now, 2),
                  "z:", round(z_now, 2),
                  "score:", round(score, 2))

            if score > scan_best_score:
                scan_best_score = score
                scan_best_angle = angle_now
                scan_best_dirx = dx_now
                scan_best_diry = dy_now
                scan_best_z = z_now

            scan_index += 1
            scan_start_time = None

    elif currState == States.OBSTACLE_ADJUST:
        robot.walkStop()
        rospy.sleep(0.20)

        dx_now, dy_now, find_now, z_now = sample_obstacle_stable()

        print("[OBSTACLE_ADJUST] iter:", obs_adjust_iter,
              "dirx:", round(dx_now, 2),
              "diry:", round(dy_now, 2),
              "find:", round(find_now, 2),
              "z:", round(z_now, 2),
              "turn_dir:", obs_adjust_turn_dir)

        if find_now >= NO_PATH_TH and z_now > STOP_DIST + 5:
            pass_start_time = time.time()
            print("[OBS_ADJ_DONE] path clear -> PASS")
            currState = States.OBSTACLE_PASS
            continue

        if obs_adjust_iter >= OBS_ADJ_MAX_ITER:
            pass_start_time = time.time()
            print("[OBS_ADJ_MAX] force -> PASS")
            currState = States.OBSTACLE_PASS
            continue

        if abs(dy_now) > 0.20:
            obs_adjust_turn_dir = 1 if dy_now > 0 else -1
            last_turn_dir = obs_adjust_turn_dir

        obstacle_adjust_pulse(obs_adjust_turn_dir)

        # Important: every short pulse re-defines current direction as forward.
        reset_imu()

        obs_adjust_iter += 1
        continue

    elif currState == States.OBSTACLE_PASS:
        if pass_start_time is None:
            pass_start_time = time.time()

        elapsed = time.time() - pass_start_time
        side_sign = obs_adjust_turn_dir

        if obs_seen and closest_obs_z < OBS_PASS_EMERGENCY_Z:
            robot.walkStop()
            rospy.sleep(0.30)
            print("[PASS_EMERGENCY_RESCAN] z:", closest_obs_z)
            scan_index = 0
            scan_start_time = None
            scan_best_score = -999.0
            currState = States.OBSTACLE_SCAN_PATH
            continue

        if obs_seen and closest_obs_z < OBS_PASS_CLOSE_Z:
            x_cmd = 0.0
            y_cmd = side_sign * OBS_PASS_Y
        else:
            x_cmd = OBS_PASS_X
            y_cmd = side_sign * OBS_PASS_Y * 0.65

        th_cmd = np.clip(z_gyro * HEADING_KP, -HEADING_TH_CLAMP, HEADING_TH_CLAMP)

        robot.walkStart()
        robot.walkVelocities(
            x=x_cmd,
            y=y_cmd,
            th=th_cmd,
            z_move_amplitude=OBS_PASS_Z_MOVE_AMPLITUDE,
            balance=True,
            z_offset=OBS_PASS_Z_OFFSET,
            hip_pitch=OBS_PASS_HIP_PITCH
        )

        print("[PASS] t:", round(elapsed, 2),
              "x:", round(x_cmd, 2),
              "y:", round(y_cmd, 2),
              "z:", round(closest_obs_z, 2),
              "seen:", obs_seen)

        if elapsed >= OBS_PASS_TIME:
            robot.walkStop()
            rospy.sleep(0.60)

            # Do not rotate back. Current body direction becomes the new forward direction.
            reset_imu()
            pass_start_time = None

            print("[PASS_DONE] no rotate-back, reset heading -> FORWARD")
            currState = States.OBSTACLE_FORWARD
            continue

    elif currState == States.END:
        robot.walkStop()
        rospy.sleep(0.2)

    rate.sleep()

robot.walkStop()
