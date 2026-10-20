#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
import time


from op3_ros_utils import getWalkingParams, Robot
from vision import *
from copy import copy
from sensor_msgs.msg import Imu
import sys

DEBUG_MODE = False
MIN_AREA = 300
CROSS_AREA = 1165
DEGREE2RADIAN = np.pi / 180

# ===== Post-pick / finish parameters =====
PICK_TO_LIFT_WALK_TIME = 14.0
LIFT_TO_FINISH_WALK_TIME = 15.0
FINISH_MOTION_ID = 185

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
    CHECK_PAN = 10
    CHECK_TOUCH = 12
    Buffer = 20
    END = 99

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0
def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    z_gyro = z_gyro - angular_velocity.z/2 + z_gyro_offset_for_caculate
    z_gyro_offset = angular_velocity.z/2

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

global last_Px, last_Py, time_for_Pd
last_Px = 0.0
last_Py = 0.0
time_for_Pd = time.time()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_hsv = np.array([22,  69, 128])
upper_hsv = np.array([ 38 ,255 ,255])

lower_g=np.array([68, 57, 48])
upper_g=np.array([ 96, 255, 255])

lower_w = np.array([ 0, 0, 190])
upper_w = np.array([180, 255, 255])

mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

go_back = 0
position_correct = []

# FIX:pan_angle/ilt_angle,avoid WALK_FORWAD
pan_angle = 0.0
tilt_angle = -22.9  #  head_tilt = -0.4 rad

rospy.init_node("fira_sprint")
robot = Robot()

rospy.sleep(3)
z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0


def detect_single_color(frame, lower_hsv, upper_hsv):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        selection = cv2.minAreaRect(largest_contour)
        (x, y), (width, height), slope = selection
        angle = compute_angle(width, height, slope)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            return cX, cY, angle
        else:
            return None, None, None


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
            angle = rect[2]
            if width < height:
                angle += 90
            cv2.imshow("frame", frame)
            return cX, cY, angle
        else:
            return None, None, None


def detect_line(frame, hsv_l, hsv_h):
    height, width = frame.shape[:2]
    point_x = width // 2
    point_y = height - 1 - 320
    color = frame[point_y, point_x]
    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)
    detected = cv2.inRange(hsv_color, hsv_l, hsv_h).any()
    circle_color = (0, 0, 0)
    cv2.circle(frame, (point_x, point_y), 5, circle_color, -1)
    return frame, detected


def compute_angle(width, height, angle):
    if angle < -90 or (width > height and angle < 0):
        return 90 + angle
    if width < height and angle > 0:
        return (90 - angle) * -1
    return angle


def camera_theta(cX, cY, frame_width, frame_height):
    global last_Px, last_Py, time_for_Pd

    Px = -1 * ((cX / float(frame_width)) - 0.5)
    Py = -1 * ((cY / float(frame_height)) - 0.5)

    pan_angle_old = np.degrees(robot.joint_pos["head_pan"])
    tilt_angle_old = np.degrees(robot.joint_pos["head_tilt"])

    Dx = (Px - last_Px) / (time.time() - time_for_Pd)
    Dy = (Py - last_Py) / (time.time() - time_for_Pd)
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
    print("Pan angle:", pan_angle)
    print("Tilt angle:", tilt_angle)
    return pan_angle, tilt_angle


# FIX: 加大容許誤差到 ±25px；修正 detect_touch 未定義問題
def detect_middle(frame, Cx):
    height, width = frame.shape[:2]
    frame_middle_x = width // 2
    TOLERANCE = 25  # 原本 4px 太嚴格，改為 25px
    orientation = 2
    cv2.line(frame, (Cx, 0), (Cx, height), (255, 0, 0), 2)

    if (frame_middle_x + TOLERANCE) >= Cx >= (frame_middle_x - TOLERANCE):
        print("touching")
        detect_touch = True
        orientation = 2
    elif Cx > frame_middle_x + TOLERANCE:   # 槓鈴在右邊，機器人向右移
        orientation = 0
        detect_touch = False
    else:                                   # 槓鈴在左邊，機器人向左移
        orientation = 1
        detect_touch = False

    return frame, detect_touch, orientation


def init():
    robot.setGeneralControlModule("action_module")
    robot.playMotion(1, wait_for_end=True)
    robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_tilt", "head_pan"], [-0.4, 0])
    rospy.sleep(1.0)


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


tickrate = 60
rate = rospy.Rate(tickrate)

STEP_LEVEL = 0
result = None
cX = None
cY = None
ang = None

currState = States.INIT
while not rospy.is_shutdown():
    ret, frame = cap.read()
    if ret:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result = detect_single_color(frame, lower_hsv, upper_hsv)
        if result is not None:
            cX, cY, ang = result
            print("computed:", ang)
            if ang is not None:
                if ang > 0:
                    ang = 90.0 - ang
                elif ang <= 0:
                    ang = (-90.0) - ang
                    print("slope:", ang)
            # FIX: 只有在偵測到物體且 cX/cY 有效時才更新 pan/tilt
            if cX is not None and cY is not None:
                if currState == States.WALK_FORWARD:
                    pan_angle, tilt_angle = camera_theta(cX, cY, frame_width, frame_height)
        cv2.imshow('MASK', mask)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    if robot.buttonCheck("mode"):
        STEP_LEVEL = 0
        currState = States.INIT

    if DEBUG_MODE:
        lower_hsv = np.array([22,  69, 128])
        upper_hsv = np.array([ 38, 255, 255])
        lower_g = np.array([68, 57, 48])
        upper_g = np.array([ 96, 255, 255])
        lower_r = np.array([0, 59, 137])
        upper_r = np.array([ 180, 255, 181])
        lower_w = np.array([ 0, 0, 231])
        upper_w = np.array([180, 255, 255])

        ret, frame = cap.read()
        print("ret,", ret)
        height, width = frame.shape[:2]
        frame_middle_x = width // 2
        print(frame_middle_x)
        print("width", width)

        if ret:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            result = detect_single_color(frame, lower_hsv, upper_hsv)
            if result is not None:
                cX, cY, ang = result
                if ang > 0:
                    ang = 90 - ang
                elif ang <= 0:
                    ang = (-90) - ang
                if cX is not None and cY is not None:
                    print("slope:", ang)
                    print("robot_pan_ang:", abs(np.degrees(robot.joint_pos["head_pan"])))
                    touch_frame, tf, what = detect_middle(frame, cX)
                    print("tf:", tf)
                    print("what:", what)
                    cv2.imshow("god", touch_frame)
        cv2.waitKey(1)

    # ==================== 主狀態機 ====================

    if currState == States.INIT:
        print("[INIT]")
        init()
        tick_count = 0
        direction = False
        currState = States.READY

    elif currState == States.READY:
        print("[READY]")
        if robot.buttonCheck("start"):
            rospy.sleep(10)
            z_gyro_offset_for_caculate = z_gyro_offset
            z_gyro = 0.0
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.4])
            tick_count = 0
            robot.walkStart()
            currState = States.WALK_FORWARD

    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD]")
        robot.walkStart()
        #robot.walkVelocities(x=3, y=-2, th=np.clip(pan_angle, -10, 10),z_move_amplitude=0.03, balance=True, z_offset=0)
        robot.walkVelocities(x=3, y=-1.20, th=np.clip(pan_angle, -2, 2)*0.56,z_move_amplitude=0.03, balance=True, z_offset=0)


        if result is not None and tilt_angle <= -50 and go_back == 0:
            robot.walkStop()
            currState = States.Adjust_Mode
            rospy.sleep(2)
        if result is not None and tilt_angle <= -62 and go_back == 1:
            robot.walkStop()
            position_correct = False
            currState = States.Adjust_Mode
            rospy.sleep(2)
        if result is not None and tilt_angle <= -62 and go_back == 2:
            robot.walkStop()
            position_correct = False
            currState = States.Adjust_Mode
            rospy.sleep(2)

    elif currState == States.Adjust_Mode:
        print('adjusting')
        if go_back == 0:
            h_tilt = -0.8
        elif go_back == 1:
            h_tilt = -1.3
        elif go_back == 2:
            h_tilt = -1.3

        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        robot.setJointPos(["head_tilt", "head_pan"], [h_tilt, 0])
        ret, frame = cap.read()

        if ret:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([22,  69, 128])
        upper_yellow = np.array([ 38, 255, 255])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        blurred = cv2.medianBlur(mask, 5)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=100,
                                 minLineLength=100, maxLineGap=10)

        slope = 0.0  # FIX: 預設值，避免 lines 為 None 時 slope 未定義
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                if float(x2 - x1) != 0:
                    slope = float(y2 - y1) / float(x2 - x1)

        cv2.imshow('Video', frame)
        cv2.waitKey(1)
        print("[ANGLE_CORRECTION]")
        print('slope:', slope)
        print('position', position_correct)

        if slope < -0.0215:
            print("turn left")
            robot.walkStart()
            #robot.walkVelocities(x=-4, th=4, z_move_amplitude=0.045, balance=True)
            robot.walkVelocities(x=-2, th=4, z_move_amplitude=0.045, balance=True)
            robot.walkStop()
        elif slope > 0.0215:
            print("turn right")
            robot.walkStart()
            #robot.walkVelocities(x=-4, th=-4, z_move_amplitude=0.045, balance=True)
            robot.walkVelocities(x=-2, th=-4, z_move_amplitude=0.045, balance=True)
            robot.walkStop()
        else:
            print("position correct")
            position_correct = True
            rospy.sleep(1)

        if position_correct == True and go_back == 0:
            robot.walkStop()
            currState = States.WALK_FORWARD
            go_back = 1
            rospy.sleep(2)
        elif position_correct == True and go_back == 1:
            robot.walkStop()
            currState = States.CHECK_TOUCH
            go_back = 2
            rospy.sleep(1)
        elif position_correct == True and go_back == 2:
            print("comp")
            currState = States.PICK_BAR
            rospy.sleep(2)
            robot.walkStop()

    # FIX: CHECK_TOUCH 縮排修正——與 Adjust_Mode 同層（主 elif）
    elif currState == States.CHECK_TOUCH:
        print("[CHECK_TOUCH]")
        robot.walkStop()
        rospy.sleep(0.5)

        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        robot.setJointPos(["head_tilt", "head_pan"], [-1.2, 0])
        rospy.sleep(0.25)

        ret2,frame2 = cap.read()
        
        if not ret2:
            print("CHECK_TOUCH camera failed")
            robot.walkStop()
            rospy.sleep(0.2)
        else:
            result2 = detect_single_color(frame2, lower_hsv, upper_hsv)

            if result2 is None :
                print("CHECK_TOUCH target lost")
                robot.walkStop()
                rospy.sleep(0.2)
            else:
                cX2, cY2, ang2 = result2
                detect_touch_frame, detected_touch, orien = detect_middle(frame2, cX2)
                print("CHECK cX2:", cX2, "cY2:", cY2, "orien:", orien)

                cv2.imshow("check_touch", detect_touch_frame)
                cv2.waitKey(1)

                PICK_CY= 310
                if cY2 is not None and cY2 >= 315:
                    print("[CHECK_TOUCH HARD_STOP] target too close, go final Adjust_Mode")
                    robot.walkStop()
                    position_correct = False
                    currState = States.PICK_BAR
                    rospy.sleep(1.0)


                elif detected_touch:
                    print("touching, go final Adjust_Mode")
                    robot.walkStop()
                    position_correct = False
                    currState = States.PICK_BAR
                    rospy.sleep(1.0)

                elif tilt_angle <= -74:
                    print("[CHECK_TOUCH EARLY_EXIT] close enough, go final Adjust_Mode")
                    robot.walkStop()
                    position_correct = False
                    currState = States.PICK_BAR
                    rospy.sleep(1.0)

                elif orien == 0:
                    print("GO LEFT once")
                    robot.walkStart()
                    t0 = time.time()
                    while time.time() - t0 < 0.18 and not rospy.is_shutdown():
                        robot.walkVelocities(x=-2.0,y=-4.5,th=0,z_move_amplitude=0.035,balance=True,z_offset=0)
                        rate.sleep()
                    robot.walkStop()
                    rospy.sleep(0.35)

                elif orien == 1:
                    print("GO RIGHT once")
                    robot.walkStart()
                    t0 = time.time()
                    while time.time() - t0 < 0.18 and not rospy.is_shutdown():
                        robot.walkVelocities(x=-2.0,y=4.5,th=0,z_move_amplitude=0.035,balance=True,z_offset=0)
                        rate.sleep()
                    robot.walkStop()
                    rospy.sleep(0.35)





    # FIX: 以下狀態全部回到主 elif 層級（不再縮排在 CHECK_TOUCH 內）
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
            robot.walkVelocities(x=5, y=0, th=np.clip(z_gyro, -10, 10),
                                 z_move_amplitude=0.035, balance=True, z_offset=0, hip_pitch=2)
            print("pick imu:", z_gyro)
            rate.sleep()

        robot.walkStop()
        rospy.sleep(1.0)

        currState = States.WALK_WITH_BAR

    elif currState == States.WALK_WITH_BAR:
        rospy.sleep(1)
        print("[WALK_WITH_BAR]")

        prepare_leg_walking()

        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        robot.setJointPos(["head_tilt", "head_pan"], [-1.5, 0])
        rospy.sleep(1)

        robot.walkStart()

        prev_time99 = time.time()
        while time.time() - prev_time99 < PICK_TO_LIFT_WALK_TIME and not rospy.is_shutdown():
            robot.walkVelocities(x=5, y=-2, th=np.clip(z_gyro, -10, 10),
                                 z_move_amplitude=0.045, balance=True, z_offset=0, hip_pitch=2)
            print("walk_with_bar imu:", z_gyro)
            rate.sleep()

        robot.walkStop()
        rospy.sleep(1)

        currState = States.LIFT_BAR

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

        currState = States.WALK_2_FINISH

    elif currState == States.WALK_2_FINISH:
        print("[WALK_2_FINISH]")

        for _ in range(8):
            robot.walkVelocities(x=0.0, y=0.0, th=0.0, z_move_amplitude=0.018,
                                 balance=True, z_offset=0.03, hip_pitch=0.0, feet_pitch=-4.3)
            rate.sleep()

        robot.walkStart()

        t0 = time.time()
        while time.time() - t0 < 0.8 and not rospy.is_shutdown():
            robot.walkVelocities(x=0.0, y=0.0, th=0.0, z_move_amplitude=0.018,
                                 balance=True, z_offset=0.03, hip_pitch=0.0, feet_pitch=-4.0)
            print("finish imu:", z_gyro)
            rate.sleep()

        t1 = time.time()
        while time.time() - t1 < 1.0 and not rospy.is_shutdown():
            robot.walkVelocities(x=3.8, y=0.4, th=np.clip(z_gyro, -10, 10),
                                 z_move_amplitude=0.026, balance=True, z_offset=0.028,
                                 hip_pitch=0.0, feet_pitch=-4.2)
            print("finish imu:", z_gyro)
            rate.sleep()

        t2 = time.time()
        while time.time() - t2 < (LIFT_TO_FINISH_WALK_TIME - 1.8) and not rospy.is_shutdown():
            robot.walkVelocities(x=5.0, y=2, th=np.clip(z_gyro, -10, 10),
                                 z_move_amplitude=0.026, balance=True, z_offset=0.028,
                                 hip_pitch=0.0, feet_pitch=-4.2)
            print("finish imu:", z_gyro)
            rate.sleep()

        robot.walkStop()
        rospy.sleep(1.0)

        finish_and_exit()

    elif currState == States.END:
        finish_and_exit()

    rate.sleep()
