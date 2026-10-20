#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
jeff_weightlifting_v2.py
改版重點：
  1. 停止判斷改用 cY（影像 Y 座標），廢掉不穩定的 tilt_angle 判距
  2. WALK_FORWARD 頭固定不追蹤，改用 IMU z_gyro 修正方向
  3. Adjust_Mode slope 改為 5 幀中位數，抗雜訊
  4. CHECK_TOUCH 改為 3 幀中位數，左右補償可獨立設定
  5. 所有關鍵參數集中在頂部「調整區」，賽前快速微調
"""

import rospy
import numpy as np
import cv2
import time

from op3_ros_utils import Robot
from vision import *
from sensor_msgs.msg import Imu
import sys

# ============================================================
#  ★★★  賽前調整區  ★★★
#  只需要改這裡，不用動下面的邏輯
# ============================================================

DEBUG_MODE = True

# ── 顏色偵測（槓鈴黃色）────────────────────────────────────
LOWER_HSV = np.array([22,  69, 128])
UPPER_HSV = np.array([38, 255, 255])

# ── 頭部固定角度（rad）─────────────────────────────────────
CY_STOP_FIRST   = 300   # 第一次停止 → Adjust_Mode (go_back==0)
CY_STOP_SECOND  = 330   # 第二次停止 → Adjust_Mode (go_back==1)
CY_STOP_FINAL   = 380   # 最終停止  → CHECK_TOUCH  (go_back==2)
CY_PICK_READY   = 400

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_callback, queue_size=1)

# ============================================================
#  狀態機
# ============================================================
class States:
    INIT         = -1
    READY        =  0
    WALK_FORWARD =  1
    PICK_BAR     =  5
    WALK_WITH_BAR=  6
    LIFT_BAR     =  7
    WALK_2_FINISH=  8
    Adjust_Mode  = 11
    CHECK_TOUCH  = 12
    END          = 99

LEG_JOINTS = [
    "r_hip_yaw",   "l_hip_yaw",
    "r_hip_roll",  "l_hip_roll",
    "r_hip_pitch", "l_hip_pitch",
    "r_knee",      "l_knee",
    "r_ank_pitch", "l_ank_pitch",
    "r_ank_roll",  "l_ank_roll"
]

# ============================================================
#  ROS / Robot 初始化
# ============================================================
rospy.init_node("fira_weightlifting_v2")
robot = Robot()
rospy.sleep(3)

z_gyro_offset_for_calc = z_gyro_offset
z_gyro = 0.0

cap = cv2.VideoCapture(0)

# ============================================================
#  共用視覺函式
# ============================================================
def detect_bar(frame):
    """偵測槓鈴（黃色），回傳 (cX, cY, angle) 或 (None, None, None)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    lc = max(contours, key=cv2.contourArea)
    M  = cv2.moments(lc)
    if M["m00"] == 0:
        return None, None, None
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    rect = cv2.minAreaRect(lc)
    (_, _), (w, h), ang = rect
    # 標準化角度
    if w > h:
        angle = ang if ang >= 0 else 90 + ang
    else:
        angle = -(90 - abs(ang))
    cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)
    cv2.drawContours(frame, [lc], -1, (0, 255, 0), 2)
    return cX, cY, angle


def detect_slope(frame):
    """用 HoughLinesP 取得槓鈴傾斜 slope，回傳 float"""
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    blur  = cv2.medianBlur(mask, 5)
    edges = cv2.Canny(blur, 50, 150)
    k     = np.ones((5, 5), np.uint8)
    dil   = cv2.dilate(edges, k, iterations=1)
    lines = cv2.HoughLinesP(dil, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    slope = 0.0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            dx = float(x2 - x1)
            if dx != 0:
                slope = float(y2 - y1) / dx
    return slope


def median_detect(n=3):
    """連拍 n 幀，取 cX、cY 中位數，回傳 (cX, cY) 或 (None, None)"""
    xs, ys = [], []
    for _ in range(n):
        ret, frm = cap.read()
        if not ret:
            rospy.sleep(0.05)
            continue
        x, y, _ = detect_bar(frm)
        if x is not None:
            xs.append(x)
            ys.append(y)
        rospy.sleep(0.05)
    if not xs:
        return None, None
    return int(np.median(xs)), int(np.median(ys))


def check_center(cX, frame_width, tol=CHECK_CENTER_TOL):
    """
    回傳 (detected_touch, orientation)
    orientation: 0=槓在右(往左移), 1=槓在左(往右移), 2=置中
    """
    mid = frame_width // 2
    if mid - tol <= cX <= mid + tol:
        return True, 2
    elif cX > mid + tol:
        return False, 0   # 槓在右
    else:
        return False, 1   # 槓在左


def set_head(tilt_rad, pan_rad=0.0, wait=0.25):
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_tilt", "head_pan"], [tilt_rad, pan_rad])
    rospy.sleep(wait)


# ============================================================
#  共用動作函式
# ============================================================
def prepare_leg_walking():
    robot.walkStop()
    rospy.sleep(0.2)
    robot.setJointsControlModule(LEG_JOINTS, ["walking_module"])
    rospy.sleep(0.6)
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    rospy.sleep(0.1)


def reset_imu():
    global z_gyro_offset_for_calc, z_gyro
    z_gyro_offset_for_calc = z_gyro_offset
    z_gyro = 0.0


def finish_and_exit():
    robot.walkStop()
    rospy.sleep(0.3)
    robot.setGeneralControlModule("none")
    rospy.sleep(0.5)
    robot.setGeneralControlModule("action_module")
    rospy.sleep(0.5)
    robot.playMotion(MOTION_FINISH, wait_for_end=True)
    rospy.sleep(0.5)
    cap.release()
    cv2.destroyAllWindows()
    rospy.signal_shutdown("Finished")


def walk_shift(x, y, th, duration, z_amp=0.035):
    """短時間走路修正，走完自動停"""
    robot.walkStart()
    t0 = time.time()
    while time.time() - t0 < duration and not rospy.is_shutdown():
        robot.walkVelocities(x=x, y=y, th=th,
                             z_move_amplitude=z_amp,
                             balance=True, z_offset=0)
        rate.sleep()
    robot.walkStop()
    rospy.sleep(0.8)


# ============================================================
#  主迴圈
# ============================================================
tickrate    = 60
rate        = rospy.Rate(tickrate)
currState   = States.INIT
go_back     = 0
position_correct = False
cX = cY = None

while not rospy.is_shutdown():

    # ── 每幀抓影像 ──────────────────────────────────────────
    ret, frame = cap.read()
    if ret:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cX, cY, _ = detect_bar(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        if DEBUG_MODE and cX is not None:
            print("[DEBUG] cX={} cY={} go_back={}".format(cX, cY, go_back))

    # ── 模式重置按鈕 ────────────────────────────────────────
    if robot.buttonCheck("mode"):
        go_back = 0
        currState = States.INIT

    # ============================================================
    #  INIT
    # ============================================================
    if currState == States.INIT:
        print("[INIT]")
        robot.setGeneralControlModule("action_module")
        robot.playMotion(MOTION_INIT, wait_for_end=True)
        robot.setGeneralControlModule("walking_module")
        set_head(HEAD_TILT_WALK, wait=1.0)
        go_back = 0
        position_correct = False
        currState = States.READY

    # ============================================================
    #  READY
    # ============================================================
    elif currState == States.READY:
        print("[READY] waiting start button...")
        if robot.buttonCheck("start"):
            rospy.sleep(10)
            reset_imu()
            set_head(HEAD_TILT_WALK)
            robot.walkStart()
            currState = States.WALK_FORWARD

    # ============================================================
    #  WALK_FORWARD
    #  ● 頭固定，不追蹤
    #  ● 用 cY 判斷距離，廢掉 tilt_angle 判距
    #  ● 用 IMU z_gyro 修正方向偏移
    # ============================================================
    elif currState == States.WALK_FORWARD:
        print("[WALK_FORWARD] go_back={} cY={}".format(go_back, cY))

        # 頭固定角度
        h_tilt = HEAD_TILT_WALK if go_back == 0 else HEAD_TILT_CLOSE
        set_head(h_tilt, wait=0.0)   # 不等待，邊走邊設

        # 走路
        th_corr = np.clip(z_gyro * WALK_FORWARD_TH_SCALE,
                          -WALK_FORWARD_CLAMP, WALK_FORWARD_CLAMP)
        robot.walkVelocities(
            x=WALK_FORWARD_X,
            y=WALK_FORWARD_Y,
            th=th_corr,
            z_move_amplitude=0.03,
            balance=True,
            z_offset=0
        )

        # 偵測不到目標 → 慢下來防止衝過去
        if cX is None or cY is None:
            print("[WALK_FORWARD] target lost, slowing")
            robot.walkVelocities(x=0.5, y=0, th=0,
                                 z_move_amplitude=0.03, balance=True, z_offset=0)
            rate.sleep()
            continue

        # ── 停止判斷（只用 cY）──────────────────────────────
        if go_back == 0 and cY >= CY_STOP_FIRST:
            print("[WALK_FORWARD] first stop, cY={}".format(cY))
            robot.walkStop()
            rospy.sleep(0.5)
            position_correct = False
            currState = States.Adjust_Mode

        elif go_back == 1 and cY >= CY_STOP_SECOND:
            print("[WALK_FORWARD] second stop, cY={}".format(cY))
            robot.walkStop()
            rospy.sleep(0.5)
            position_correct = False
            currState = States.Adjust_Mode

        elif go_back == 2 and cY >= CY_STOP_FINAL:
            print("[WALK_FORWARD] final stop, cY={}".format(cY))
            robot.walkStop()
            rospy.sleep(0.5)
            position_correct = False
            currState = States.CHECK_TOUCH

    # ============================================================
    #  Adjust_Mode
    #  ● 5 幀中位數 slope，抗雜訊
    #  ● 每次只轉一小步，再重新量測
    # ============================================================
    elif currState == States.Adjust_Mode:
        print("[Adjust_Mode] go_back={}".format(go_back))

        h_tilt = HEAD_TILT_ADJUST_0 if go_back == 0 else HEAD_TILT_ADJUST_1
        set_head(h_tilt, wait=0.3)

        # 連拍取中位數 slope
        slopes = []
        for _ in range(ADJUST_FRAMES):
            ret2, frm2 = cap.read()
            if ret2:
                slopes.append(detect_slope(frm2))
                cv2.imshow("adjust", frm2)
                cv2.waitKey(1)
            rospy.sleep(0.05)

        slope = float(np.median(slopes)) if slopes else 0.0
        print("[Adjust_Mode] median slope={:.4f}".format(slope))

        if slope < -SLOPE_THRESHOLD:
            print("  → turn left")
            walk_shift(ADJUST_TURN_X, 0, ADJUST_TURN_TH, ADJUST_TURN_TIME, z_amp=0.04)

        elif slope > SLOPE_THRESHOLD:
            print("  → turn right")
            walk_shift(ADJUST_TURN_X, 0, -ADJUST_TURN_TH, ADJUST_TURN_TIME, z_amp=0.04)

        else:
            print("  → aligned!")
            position_correct = True
            rospy.sleep(0.5)

        # 狀態轉移
        if position_correct:
            position_correct = False
            if go_back == 0:
                go_back = 1
                currState = States.WALK_FORWARD
                rospy.sleep(1.0)
            elif go_back == 1:
                go_back = 2
                currState = States.CHECK_TOUCH
                rospy.sleep(1.0)
            elif go_back == 2:
                currState = States.PICK_BAR
                rospy.sleep(1.0)

    # ============================================================
    #  CHECK_TOUCH
    #  ● 連拍 3 幀取中位數，左右補償獨立設定
    # ============================================================
    elif currState == States.CHECK_TOUCH:
        print("[CHECK_TOUCH]")
        robot.walkStop()
        rospy.sleep(0.5)
        set_head(HEAD_TILT_CHECK, wait=0.3)

        cX2, cY2 = median_detect(CHECK_DETECT_FRAMES)

        if cX2 is None:
            print("[CHECK_TOUCH] no detection, retry")
            rospy.sleep(0.5)
            # 留在 CHECK_TOUCH，下一輪再試

        else:
            touch, orien = check_center(cX2, fw)
            print("[CHECK_TOUCH] cX2={} cY2={} touch={} orien={}".format(
                  cX2, cY2, touch, orien))

            # 距離夠且置中 → 夾取
            if touch and cY2 >= CY_PICK_READY:
                print("  → PICK_BAR")
                robot.walkStop()
                currState = States.PICK_BAR
                rospy.sleep(1.0)

            # 置中但太遠 → 小步前進
            elif touch and cY2 < CY_PICK_READY:
                print("  → step forward once")
                walk_shift(CHECK_STEP_X, 0, 0, CHECK_STEP_TIME)

            # 槓在右 → 往左移
            elif orien == 0:
                print("  → shift left")
                walk_shift(0, CHECK_SHIFT_LEFT_Y, 0, CHECK_SHIFT_TIME)

            # 槓在左 → 往右移
            elif orien == 1:
                print("  → shift right")
                walk_shift(0, CHECK_SHIFT_RIGHT_Y, 0, CHECK_SHIFT_TIME)

    # ============================================================
    #  PICK_BAR
    # ============================================================
    elif currState == States.PICK_BAR:
        print("[PICK_BAR]")
        robot.walkStop()
        rospy.sleep(2.0)

        robot.setGeneralControlModule("none")
        rospy.sleep(1.0)
        robot.setGeneralControlModule("action_module")
        rospy.sleep(0.5)
        robot.playMotion(MOTION_PICK, wait_for_end=True)
        rospy.sleep(1.0)

        reset_imu()
        rospy.sleep(2.0)

        prepare_leg_walking()
        robot.walkStart()

        t0 = time.time()
        while time.time() - t0 < 2.0 and not rospy.is_shutdown():
            robot.walkVelocities(
                x=5, y=0,
                th=np.clip(z_gyro, -10, 10),
                z_move_amplitude=0.035,
                balance=True, z_offset=0, hip_pitch=2
            )
            print("[PICK] imu:", z_gyro)
            rate.sleep()

        robot.walkStop()
        rospy.sleep(1.0)
        currState = States.WALK_WITH_BAR

    # ============================================================
    #  WALK_WITH_BAR
    # ============================================================
    elif currState == States.WALK_WITH_BAR:
        print("[WALK_WITH_BAR]")
        rospy.sleep(1.0)

        prepare_leg_walking()
        set_head(-1.5, wait=1.0)
        robot.walkStart()

        t0 = time.time()
        while time.time() - t0 < WALK_WITH_BAR_TIME and not rospy.is_shutdown():
            robot.walkVelocities(
                x=WALK_WITH_BAR_X,
                y=WALK_WITH_BAR_Y,
                th=np.clip(z_gyro, -10, 10),
                z_move_amplitude=0.045,
                balance=True, z_offset=0, hip_pitch=2
            )
            print("[WALK_WITH_BAR] imu:", z_gyro)
            rate.sleep()

        robot.walkStop()
        rospy.sleep(1.0)
        currState = States.LIFT_BAR

    # ============================================================
    #  LIFT_BAR
    # ============================================================
    elif currState == States.LIFT_BAR:
        print("[LIFT_BAR]")
        robot.walkStop()
        rospy.sleep(0.3)

        robot.setGeneralControlModule("none")
        rospy.sleep(1.0)
        robot.setGeneralControlModule("action_module")
        rospy.sleep(0.5)
        robot.playMotion(MOTION_LIFT, wait_for_end=True)
        rospy.sleep(1.3)

        reset_imu()
        rospy.sleep(0.8)

        prepare_leg_walking()
        rospy.sleep(0.5)
        currState = States.WALK_2_FINISH

    # ============================================================
    #  WALK_2_FINISH
    # ============================================================
    elif currState == States.WALK_2_FINISH:
        print("[WALK_2_FINISH]")

        # 暖身步：原地踏步穩定重心
        for _ in range(8):
            robot.walkVelocities(
                x=0.0, y=0.0, th=0.0,
                z_move_amplitude=0.018,
                balance=True, z_offset=0.03,
                hip_pitch=0.0, feet_pitch=-4.3
            )
            rate.sleep()

        robot.walkStart()

        # 啟動段
        t0 = time.time()
        while time.time() - t0 < 0.8 and not rospy.is_shutdown():
            robot.walkVelocities(
                x=0.0, y=0.0, th=0.0,
                z_move_amplitude=0.018,
                balance=True, z_offset=0.03,
                hip_pitch=0.0, feet_pitch=-4.0
            )
            rate.sleep()

        # 加速段
        t1 = time.time()
        while time.time() - t1 < 1.0 and not rospy.is_shutdown():
            robot.walkVelocities(
                x=3.8, y=WALK_2_FINISH_Y,
                th=np.clip(z_gyro, -10, 10),
                z_move_amplitude=0.026,
                balance=True, z_offset=0.028,
                hip_pitch=0.0, feet_pitch=-4.2
            )
            print("[WALK_2_FINISH] imu:", z_gyro)
            rate.sleep()

        # 巡航段
        t2 = time.time()
        while time.time() - t2 < (LIFT_TO_FINISH_TIME - 1.8) and not rospy.is_shutdown():
            robot.walkVelocities(
                x=5.0, y=WALK_2_FINISH_Y,
                th=np.clip(z_gyro, -10, 10),
                z_move_amplitude=0.026,
                balance=True, z_offset=0.028,
                hip_pitch=0.0, feet_pitch=-4.2
            )
            print("[WALK_2_FINISH] imu:", z_gyro)
            rate.sleep()

        robot.walkStop()
        rospy.sleep(1.0)
        finish_and_exit()

    # ============================================================
    #  END
    # ============================================================
    elif currState == States.END:
        finish_and_exit()

    rate.sleep()
