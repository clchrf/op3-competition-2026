#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time
import math

from op3_ros_utils_jeff import getWalkingParams, Robot
from vision import *
from copy import copy
from sensor_msgs.msg import Imu
import sys

DEBUG_MODE = False
# DEBUG_MODE = True
MIN_AREA = 300
CROSS_AREA = 1165
DEGREE2RADIAN = np.pi / 180

# ===== Post-pick / finish parameters =====
PICK_TO_LIFT_WALK_TIME = 14.0
LIFT_TO_FINISH_WALK_TIME = 15.0
FINISH_MOTION_ID = 185

# ===== Vision anti-background tuning =====
ROI_TOP = 180               # only use lower part of image
SIDE_MARGIN = 80            # ignore far left/right background
TARGET_MIN_AREA = 400
TARGET_MAX_AREA = 18000     # reject very large background block
CENTER_WEIGHT = 2.0         # prefer target near image center

LEG_JOINTS = [
    "r_hip_yaw", "l_hip_yaw",
    "r_hip_roll", "l_hip_roll",
    "r_hip_pitch", "l_hip_pitch",
    "r_knee", "l_knee",
    "r_ank_pitch", "l_ank_pitch",
    "r_ank_roll", "l_ank_roll"
]

class States:
    INIT = -1
    READY = 0
    WALK_FORWARD = 1
    PICK_BAR = 5
    WALK_WITH_BAR = 6
    LIFT_BAR = 7
    WALK_2_FINISH = 8
    CHECK_SLOPE = 9
    Adjust_Mode = 11
    Adjust_Mode_V2 = 13
    CHECK_SLOPE_V2 = 14
    CHECK_PAN = 10
    CHECK_TOUCH = 12
    Buffer = 20
    END = 99

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0

_count = 0
last_Px = 0.0
last_Py = 0.0
time_for_Pd = time.time()

go_back = 0
position_correct = []
walk_started = False


def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    z_gyro = z_gyro - (angular_velocity.z / 2.0) + z_gyro_offset_for_caculate
    z_gyro_offset = angular_velocity.z / 2.0


def compute_angle(width, height, angle):
    if angle < -90 or (width > height and angle < 0):
        return 90 + angle
    if width < height and angle > 0:
        return (90 - angle) * -1
    return angle


def detect_single_color(frame, lower_hsv, upper_hsv):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Wider red range for near-field detection
    lower_red1 = np.array([0, 80, 40])
    upper_red1 = np.array([12, 255, 255])

    lower_red2 = np.array([155, 80, 40])
    upper_red2 = np.array([179, 255, 255])

    mask_R1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_R2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_comb = cv2.bitwise_or(mask_R1, mask_R2)

    h, w = mask_comb.shape[:2]

    # ROI: keep lower-middle image, but do not crop too much
    roi_mask = np.zeros_like(mask_comb)
    roi_mask[ROI_TOP:h, 40:w - 40] = 255
    mask_comb = cv2.bitwise_and(mask_comb, roi_mask)

    # Fill broken red regions
    kernel = np.ones((5, 5), np.uint8)
    mask_comb = cv2.medianBlur(mask_comb, 5)
    mask_comb = cv2.morphologyEx(mask_comb, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_comb = cv2.dilate(mask_comb, kernel, iterations=1)

    _, contours, _ = cv2.findContours(mask_comb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        cv2.imshow("mask", mask_comb)
        return None, None, None

    frame_center_x = w / 2.0
    best_contour = None
    best_score = -1e18

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < TARGET_MIN_AREA:
            continue

        # Do not reject large target when robot is close to the bar
        # if area > TARGET_MAX_AREA:
        #     continue

        x, y, bw, bh = cv2.boundingRect(contour)

        # Prefer horizontal red target
        if bw < bh * 1.5:
            continue

        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        center_penalty = abs(cX - frame_center_x) * 1.0
        score = area - center_penalty

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is None:
        cv2.imshow("mask", mask_comb)
        return None, None, None

    M = cv2.moments(best_contour)
    if M["m00"] == 0:
        cv2.imshow("mask", mask_comb)
        return None, None, None

    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    rect = cv2.minAreaRect(best_contour)
    (x, y), (width, height), slope = rect
    angle = compute_angle(width, height, slope)

    area = cv2.contourArea(best_contour)
    bx, by, bw, bh = cv2.boundingRect(best_contour)

    print("bar area:", area, "bbox:", bw, bh, "cX:", cX, "cY:", cY, "angle:", angle)

    cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
    cv2.drawContours(frame, [best_contour], -1, (0, 255, 0), 2)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

    cv2.imshow("mask", mask_comb)

    return cX, cY, angle


def detect_single_color_V2(frame, lower_hsv, upper_hsv):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ratio = 2.0
    hor_contour = []

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (x, y), (width, height), slope = rect

        if width > height * ratio:
            hor_contour.append(contour)

    if hor_contour:
        largest_contour = max(hor_contour, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)

            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]
            width = rect[1][0]
            height = rect[1][1]
            if width < height:
                angle += 90

            cv2.imshow("frame", frame)
            return cX, cY, angle
        else:
            return None, None, None

    return None, None, None


def detect_line(frame, hsv_l, hsv_h):
    height, width = frame.shape[:2]
    point_x = width // 2
    point_y = height - 1 - 320

    if point_y < 0:
        point_y = height // 2

    color = frame[point_y, point_x]
    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)

    detected = cv2.inRange(hsv_color, hsv_l, hsv_h).any()
    circle_color = (0, 0, 0)
    cv2.circle(frame, (point_x, point_y), 5, circle_color, -1)
    return frame, detected


def camera_theta(cX, cY, frame_width, frame_height):
    global last_Px, last_Py, time_for_Pd

    Px = -1 * ((cX / float(frame_width)) - 0.5)
    Py = -1 * ((cY / float(frame_height)) - 0.5)

    pan_angle_old = np.degrees(robot.joint_pos["head_pan"])
    tilt_angle_old = np.degrees(robot.joint_pos["head_tilt"])

    dt = time.time() - time_for_Pd
    if dt <= 0:
        dt = 1e-6

    Dx = (Px - last_Px) / dt
    Dy = (Py - last_Py) / dt
    time_for_Pd = time.time()

    kp = 20
    kd = 0

    pan_angle = pan_angle_old + Px * kp + Dx * kd
    tilt_angle = tilt_angle_old + Py * kp + Dy * kd

    last_Px, last_Py = Px, Py

    robot.setJointPos(
        ["head_pan", "head_tilt"],
        [
            np.clip(np.radians(pan_angle), np.radians(-95), np.radians(95)),
            np.clip(np.radians(tilt_angle), np.radians(-95), np.radians(95))
        ]
    )

    return pan_angle, tilt_angle


def detect_middle(frame, Cx):
    height, width = frame.shape[:2]
    frame_middle_x = width // 2
    orientation = 2
    cv2.line(frame, (int(Cx), 0), (Cx, height), (255, 0, 0), 2)

    if (frame_middle_x + 4) >= Cx >= (frame_middle_x - 4):
        print("touching")
        detect_touch = True
    elif Cx > frame_middle_x + 4:
        orientation = 0
        detect_touch = False
    else:
        orientation = 1
        detect_touch = False

    return frame, detect_touch, orientation


def init():
    robot.setGeneralControlModule("action_module")
    robot.playMotion(1, wait_for_end=True)
    robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_tilt", "head_pan"], [-0.45, 0])
    rospy.sleep(1.0)


def prepare_walking():
    robot.walkStop()
    rospy.sleep(0.2)
    robot.setGeneralControlModule("walking_module")
    rospy.sleep(0.6)
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    rospy.sleep(0.1)


def prepare_leg_walking():
    robot.walkStop()
    rospy.sleep(0.2)
    robot.setJointsControlModule(LEG_JOINTS, ["walking_module"])
    rospy.sleep(0.6)
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    rospy.sleep(0.1)


def finish_and_exit():
    robot.walkStop()
    rospy.sleep(0.3)
    robot.setGeneralControlModule("none")
    rospy.sleep(0.5)
    robot.setGeneralControlModule("action_module")
    rospy.sleep(0.5)
    robot.playMotion(FINISH_MOTION_ID, wait_for_end=True)
    rospy.sleep(0.5)
    cap.release()
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Finished after motion {}".format(FINISH_MOTION_ID))


# Initialize node first
rospy.init_node("fira_sprint")

# Create robot
robot = Robot()

# IMU subscriber after node init
rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to open camera / read initial frame.")

hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Keep your original color setup
lower_hsv = np.array([109, 36, 42])
upper_hsv = np.array([136, 230, 255])

lower_g = np.array([68, 57, 48])
upper_g = np.array([96, 255, 255])

lower_w = np.array([0, 0, 190])
upper_w = np.array([180, 255, 255])

mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

rospy.sleep(3)
z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0

tickrate = 60
rate = rospy.Rate(tickrate)

STEP_LEVEL = 0
currState = States.INIT

while not rospy.is_shutdown():
    ret, frame = cap.read()
    result = None
    cX, cY, ang = None, None, None

    if ret:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result = detect_single_color(frame, lower_hsv, upper_hsv)

        if result is not None:
            cX, cY, ang = result

            if ang is not None:
                if ang > 0:
                    ang = 90.0 - ang
                else:
                    ang = (-90.0) - ang
                    print("slope:", ang)

            if cX is not None and cY is not None:
                pan_angle, tilt_angle = camera_theta(cX, cY, frame_width, frame_height)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    if robot.buttonCheck("mode"):
        STEP_LEVEL = 0
        walk_started = False
        currState = States.INIT

    if DEBUG_MODE:
        lower_hsv = np.array([100, 69, 128])
        upper_hsv = np.array([140, 255, 255])
        lower_g = np.array([68, 57, 48])
        upper_g = np.array([96, 255, 255])

        lower_r = np.array([0, 59, 137])
        upper_r = np.array([180, 255, 181])

        lower_w = np.array([0, 0, 231])
        upper_w = np.array([180, 255, 255])

        ret, dbg_frame = cap.read()
        if ret:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dbg_result = detect_single_color(dbg_frame, lower_hsv, upper_hsv)

            if dbg_result is not None:
                dbg_cX, dbg_cY, dbg_ang = dbg_result
                if dbg_ang is not None:
                    if dbg_ang > 0:
                        dbg_ang = 90 - dbg_ang
                    else:
                        dbg_ang = (-90) - dbg_ang

                if dbg_cX is not None and dbg_cY is not None:
                    touch_frame, tf, what = detect_middle(dbg_frame, dbg_cX)
                    cv2.imshow("god", touch_frame)

        cv2.waitKey(1)

    if currState == States.INIT:
        print("[INIT]")
        init()
        tick_count = 0
        direction = False
        walk_started = False
        currState = States.READY

    elif currState == States.READY:
        print("[READY]")
        if robot.buttonCheck("user"):
            rospy.sleep(5)
            z_gyro_offset_for_caculate = z_gyro_offset
            z_gyro = 0.0
            print(z_gyro)

            prepare_walking()
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.4])

            tick_count = 0
            walk_started = False
            currState = States.Adjust_Mode_V2

    elif currState == States.Adjust_Mode_V2:
        _count = _count + 1

        if not walk_started:
            prepare_walking()
            robot.walkStart()
            walk_started = True

        head_pan_deg = np.degrees(robot.joint_pos["head_pan"])

        y_track = np.clip(-0.35*head_pan_deg, -1.2, 1.2)
        y_cmd = np.clip(y_track +1.0, -0.5, 2.2)    

        robot.walkVelocities(
            x=4.58,
            y=-1.85,
            th=np.clip(z_gyro, 0, 2)*0.5 ,
            balance=True
        )

        show_tilt = 0
        if show_tilt == 0:
            print("tilt: ")
            print(abs(78 + np.degrees(robot.joint_pos["head_tilt"])))
            print("gyro: ")
            print(z_gyro)
        
        #8.3
        if abs(78 + np.degrees(robot.joint_pos["head_tilt"])) < 7.8:
            robot.walkStop()
            walk_started = False
            rospy.sleep(2)

            currState = States.PICK_BAR
            rospy.sleep(1)
            show_tilt = 1
            show_gyro = 0

            if show_gyro == 0:
                print("z_gyro: ")
                print(z_gyro)
                position_correct = False
                _count = 0
                if abs(z_gyro) < 15:
                    show_gyro = 1
                    currState = States.PICK_BAR
                    rospy.sleep(1)

    elif currState == States.PICK_BAR:
        rospy.loginfo("[PICK_BAR]")
        rospy.sleep(2)

        robot.walkStop()
        rospy.sleep(0.3)

        robot.setGeneralControlModule("none")
        rospy.sleep(1.0)
        robot.setGeneralControlModule("action_module")
        rospy.sleep(0.5)
        robot.playMotion(195, wait_for_end=True)
        rospy.sleep(1.0)

        z_gyro_offset_for_caculate = z_gyro_offset
        z_gyro = 0.0
        rospy.sleep(2.0)

        prepare_leg_walking()
        robot.walkStart()

        prev_time = time.time()
        while time.time() - prev_time < 2.0 and not rospy.is_shutdown():
            #robot.walkVelocities(x=5,y=0,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=-10)
            robot.walkVelocities(x=5,y=0,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.035,balance=True,z_offset=0,hip_pitch=2)
            print("pick imu:", z_gyro)

        robot.walkStop()
        walk_started = False
        rospy.sleep(1.0)
        currState = States.WALK_WITH_BAR

    elif currState == States.WALK_WITH_BAR:
        rospy.sleep(1)
        print("[WALK_WITH_BAR]")

        prepare_leg_walking()
        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        robot.setJointPos(["head_tilt", "head_pan"], [-1.5, 0])
        rospy.sleep(1)

        if not walk_started:
            robot.walkStart()
            walk_started = True

        prev_time99 = time.time()
        while time.time() - prev_time99 < PICK_TO_LIFT_WALK_TIME and not rospy.is_shutdown():
           # robot.walkVelocities(x=5,y=0,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=-10)
            robot.walkVelocities(x=5,y=-2,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=2)
            print("walk_with_bar imu:", z_gyro)

        robot.walkStop()
        walk_started = False
        currState = States.LIFT_BAR
        rospy.sleep(1)

    elif currState == States.LIFT_BAR:
        print("[LIFT_BAR]")

        robot.walkStop()
        rospy.sleep(0.3)

        robot.setGeneralControlModule("none")
        rospy.sleep(1.0)
        robot.setGeneralControlModule("action_module")
        rospy.sleep(0.5)
        robot.playMotion(186, wait_for_end=True)
        rospy.sleep(1.3)

        z_gyro_offset_for_caculate = z_gyro_offset
        z_gyro = 0.0
        rospy.sleep(0.8)

        prepare_leg_walking()
        rospy.sleep(0.5)
        walk_started = False
        currState = States.WALK_2_FINISH

    elif currState == States.WALK_2_FINISH:
        print("[WALK_2_FINISH]")

        for _ in range(8):
            robot.walkVelocities(x=0.0,y=0.0,th=0.0,z_move_amplitude=0.018,balance=True,z_offset=0.03,hip_pitch=0.0, feet_pitch=-4.3)
            rate.sleep()

        if not walk_started:
            robot.walkStart()
            walk_started = True


        t0 = time.time()
        while time.time() - t0 < 0.8 and not rospy.is_shutdown():
            #robot.walkVelocities(x=6.75,y=2,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=5.2)
            #robot.walkVelocities(x=6.0,y=2,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=5.0)
            robot.walkVelocities(x=0.0,y=0.0,th=0.0,z_move_amplitude=0.018,balance=True,z_offset=0.03,hip_pitch=0.0,feet_pitch=-4.0)
            print("finish imu:", z_gyro)
            rate.sleep()

        t1 = time.time()
        while time.time() - t1 < 1.0 and not rospy.is_shutdown():
            #robot.walkVelocities(x=6.75,y=2,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=5.2)
            #robot.walkVelocities(x=6.0,y=2,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=5.0)
            robot.walkVelocities(x=3.8,y=0.4,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.026,balance=True,z_offset=0.028,hip_pitch=0.0,feet_pitch=-4.2)
            print("finish imu:", z_gyro)
            rate.sleep()

        t2 = time.time()
        while time.time() - t2 < (LIFT_TO_FINISH_WALK_TIME-1.8) and not rospy.is_shutdown():
            #robot.walkVelocities(x=6.75,y=2,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=5.2)
            #robot.walkVelocities(x=6.0,y=2,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.045,balance=True,z_offset=0,hip_pitch=5.0)
            robot.walkVelocities(x=5.0,y=2,th=np.clip(z_gyro, -10, 10),z_move_amplitude=0.026,balance=True,z_offset=0.028,hip_pitch=0.0,feet_pitch=-4.2)
            print("finish imu:", z_gyro)
            rate.sleep()


        robot.walkStop()
        walk_started = False
        rospy.sleep(1.0)
        finish_and_exit()

    elif currState == States.END:
        finish_and_exit()

    rate.sleep()

cap.release()
cv2.destroyAllWindows()