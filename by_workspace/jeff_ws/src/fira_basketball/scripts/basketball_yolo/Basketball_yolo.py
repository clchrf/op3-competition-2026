#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import time
import cv2
import math
import numpy as np
from sensor_msgs.msg import Imu,JointState
from tf.transformations import euler_from_quaternion
from op3_ros_utils1 import getWalkingParams, Robot
from copy import copy
from std_msgs.msg import Int32, String, Float32, Float64, Bool, Float32MultiArray
from yolo_receiver import YoloReceiver   # 從 detector.py 收 YOLO 偵測


first=True
fixg=False
z_gyro = 0.0
z_list=[]
z_gyro_offset = 0.0
clock=0
loop_count=0
# ============================================================
# Right arm hold after PICK_MOTION
# ============================================================

RIGHT_ARM_JOINTS = [
    "r_sho_pitch",
    "r_sho_roll",
    "r_el",
    "r_gripper"
]

LEG_JOINTS = [
    "r_hip_yaw", "l_hip_yaw",
    "r_hip_roll", "l_hip_roll",
    "r_hip_pitch", "l_hip_pitch",
    "r_knee", "l_knee",
    "r_ank_pitch", "l_ank_pitch",
    "r_ank_roll", "l_ank_roll"
]

present_joint_pos = {}
present_joint_eff = {}

right_arm_hold_enabled = False
right_arm_hold_pos = None
right_arm_module_locked = False

# 夾爪保持角度微調。先維持 0.0，實測確定夾緊方向後再改成 +0.05 或 -0.05。
RIGHT_GRIPPER_HOLD_OFFSET = 0.0

# effort 先只列印，不自動判斷掉球。
GRIPPER_EFFORT_DEBUG = True
GRIPPER_EFFORT_PRINT_INTERVAL = 1.0
GRIPPER_DROP_CHECK_ENABLED = False
GRIPPER_DROP_EFFORT_MIN = 0.0
GRIPPER_DROP_FRAMES = 5

gripper_low_count = 0
last_gripper_effort_print = 0.0
gripper_effort_missing_printed = False


def read_joint_state(data):
    global present_joint_pos
    global present_joint_eff

    for i, name in enumerate(data.name):
        if i < len(data.position):
            present_joint_pos[name] = data.position[i]

        if i < len(data.effort):
            present_joint_eff[name] = data.effort[i]


def read_IMU(data):#imu_get_yaw_by_integral 
    global first
    global fixg
    global old_z
    global z_gyro
    global z_list
    global z_gyro_offset
    global clock
    global loop_count
    global argsIMU
    alpha = 0.7
    
    rospy.sleep(0.001)
    loop_count+=1 
    if clock==0:
        clock=time.time()
        ltime=0.0079
    else:
        
        loop=time.time()-clock
        ltime=loop/loop_count
    
    

    #print(ltime)
    
    angular_velocity = data.angular_velocity
    z=angular_velocity.z

    if first:     
        z=angular_velocity.z
    else:
        z=(alpha*old_z)+((1-alpha)*angular_velocity.z)
    old_z=z
    
    #print(z)
    #z+=0.003
    z=math.floor(z*100)/100
    

    if fixg:
        error=z-z_gyro_offset
        dg=(error)*ltime
        dg=dg*180/np.pi
        z_gyro = z_gyro - dg
        #print(z_gyro)
    else:
        z_list+=[z]
        if len(z_list)>800:
            del z_list[0]
        avg=sum(z_list)/(len(z_list))
        avg=math.floor(avg*100)/100
        if abs(avg-z_list[-1])<0.001 and len(z_list)>=500:
            print("ok")
            z_gyro_offset=sum(z_list)/(len(z_list))
            z_gyro_offset=math.floor(z_gyro_offset*100)/100
            fixg=True
    





rospy.init_node("action_test")

robot=Robot()
# 相機改由 detector.py (py3) 接管，控制節點不再自己開相機
# cap=cv2.VideoCapture(0)


robot.setGeneralControlModule("walking_module")
rospy.sleep(1)

robot.setGeneralControlModule("action_module")
robot.setJointsControlModule(["head_pan","head_tilt"],["none","none"])
robot.setJointsControlModule(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],["none","none","none","none"])
robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el","l_gripper"],[0,0.78,-0.78,0])
rospy.sleep(2)

fixg=False
z_gyro = 0.0
z_list=[]
print("init gyro")
rospy.Subscriber(
    "/robotis/open_cr/imu",
    Imu,
    read_IMU,
    queue_size=1
)

rospy.Subscriber(
    "/robotis/present_joint_states",
    JointState,
    read_joint_state,
    queue_size=1
)

rospy.sleep(1)
print("imu and joint state ok")

robot.setJointPos(["head_pan","head_tilt"],[0,-0.5])
rospy.sleep(1)


"""vfov 73.74 hfov 90"""
situation="wait"
gyro_offset=0
add=0.01
pan_angle=0
tilt_angle=-0.5
state="find"
mode="not move"
move_mode="h"
theta_list=[]


# ============================================================
# Basketball YOLO controller
# 對齊 weightlifting Adjust_Mode 的節奏:
#   每次決策前取樣 N 幀取中位數 -> 一次小脈衝 -> walkStop -> 停穩 1.0s -> 再取樣
#   (舉重: FINE_SAMPLE_FRAMES=7, 脈衝後 rospy.sleep(1.0))
# 另外移植舉重的兩個機制:
#   1. 丟失賭一把: 細調中丟球太久、但最後球寬已夠大 -> 球就在腳邊視野外, 直接抓
#      (舉重: "lost too many, force -> PICK_BAR")
#   2. iteration 上限: 細調修太多輪就強制放行 (舉重: FINE_MAX_ITER=24)
# 注意: 舉重速度值(x=2.5 等)是 op3_ros_utils 刻度且有 z_move_amplitude 等參數,
#       本程式是 op3_ros_utils1(前進為負 x、量級 0.0x), 速度值沿用籃球已實測量級。
#
# ball calibration:
#   head_tilt AE=1556, cx=352, cy=396, w=99, h=101.1 -> playMotion 100 可抓球
# hoop calibration:
#   head_tilt AE=1974, head_pan AE=991, cx=309, cy=315, w=277, h=325.5 -> playMotion 99 可投進
# ============================================================

def ae_to_rad(value):
    # ROBOTIS Action Editor / Dynamixel position value to radian.
    # 2048 is approximately 0 rad, 4096 values = 2*pi.
    return (float(value) - 2048.0) * 2.0 * np.pi / 4096.0

# ---------- Action pages ----------
PICK_MOTION = 100
THROW_MOTION = 99

# ---------- Head calibration ----------
BALL_HEAD_PAN = 0.0
BALL_HEAD_TILT = ae_to_rad(1556)       # about -0.755 rad
SCAN_HEAD_TILT = ae_to_rad(1800)   # 約 -0.38 rad

HOOP_HEAD_PAN = ae_to_rad(991)         # about -1.622 rad
HOOP_HEAD_TILT = ae_to_rad(1974)       # about -0.114 rad
THROW_HEAD_PAN = ae_to_rad(995)
THROW_HEAD_TILT = ae_to_rad(1554)
# 抓球後固定向右旋轉 8 秒
HOOP_RIGHT_TURN_TH = -2.0
HOOP_RIGHT_TURN_TIME = 7.5

# 籃框達到此尺寸後切換成最終投籃頭角
HOOP_FINAL_SWITCH_SIZE = 120.0

# 最終 CX 誤差大於此值時使用粗轉
HOOP_FINAL_COARSE_X_ERR = 60
# 前段籃框轉向較早完成，避免旋轉步態自然向前靠近
HOOP_TURN_DONE_PAN = 0.65
HOOP_TURN_DONE_FRAMES = 1

# 頭轉到最終右視角後，先往左旋轉 5 秒
HOOP_PRE_FINAL_LEFT_TH = 1.8
HOOP_PRE_FINAL_LEFT_TIME = 5.0

# ---------- Ball target in image ----------
BALL_APPROACH_CX_TARGET = 331
BALL_CX_TARGET = 352
BALL_CY_TARGET = 394
BALL_W_TARGET = 96.0
BALL_H_TARGET = 94.0

BALL_X_TOL = 16
BALL_Y_TOL = 12
BALL_W_TOL = 14
BALL_H_TOL = 16
BALL_STABLE_FRAMES = 2      # 連續幾「輪取樣」達標才抓(取樣制,一輪約 0.35s)

# close/fine walking values. x negative = toward ball in this codebase.
BALL_FWD_FAST = -0.020
BALL_FWD_MID  = -0.010
BALL_FWD_FINE = -0.004
BALL_BACK_FINE = 0.008

# ---------- Hoop target in image ----------
HOOP_CX_TARGET = 314
HOOP_CY_TARGET = 96
HOOP_W_TARGET = 281.0
HOOP_H_TARGET = 194.0

HOOP_X_TOL = 20
HOOP_Y_TOL = 14
HOOP_W_TOL = 28
HOOP_H_TOL = 28
HOOP_STABLE_FRAMES = 2


# ---------- Hoop staged search / center / walk / final ----------

# 前段只在正前方附近掃描，不要一看到框就把頭直接轉到最右側標定角。
HOOP_SCAN_HEAD_PANS = [
    0.0,
    -0.35,
    0.35,
    -0.70,
    0.70
]

HOOP_SCAN_INTERVAL = 0.80
HOOP_LOCK_FRAMES = 3
HOOP_SCAN_MIN_SIZE = 35.0
HOOP_SCAN_MIN_CONF = 0.65
HOOP_PAN_LIMIT = 1.75

# 第一次偵測到籃框後，先原地轉開放球底座，再重新偵測。
# 方向優先由第一次看到籃框的方向決定；方向太小時使用 DEFAULT_SIGN。
HOOP_INITIAL_TURN_TH = 2.0
HOOP_INITIAL_TURN_TIME = 7.0
HOOP_INITIAL_TURN_DEFAULT_SIGN = 1.0

# 中間視角縮小。
# log 顯示 -0.81 rad 時籃框立刻完全離開畫面，
# 但 head_pan=-0.35 rad 時仍能穩定看到籃框，因此改成 -0.35。
HOOP_MID_HEAD_PAN = -0.35

# 只依籃框尺寸進入中間視角，不再因為走完兩輪就強制切換。
HOOP_MID_ENTER_SIZE = 120.0

# 中間視角的對中條件與前段 HOOP_STAGE CX 校正一致。
HOOP_MID_CX_TARGET = 320
HOOP_MID_CX_TOL = 28
HOOP_MID_CENTER_FRAMES = 2

# 中間視角旋轉校正的增益、最小/最大力量、時間與停穩時間，
# 全部與前段 HOOP_STAGE_TURN 相同。
HOOP_MID_TH_SIGN = 1.0
HOOP_MID_TURN_KP = 0.020
HOOP_MID_TURN_MIN = 1.20
HOOP_MID_TURN_MAX = 2.00
HOOP_MID_TURN_TIME = 1.20
HOOP_MID_TURN_SETTLE = 0.55

# 切換中間視角後若暫時看不到籃框：
# 只用與前段相同的小轉參數，左右交替搜尋兩次；
# 找不到就回 scan_hoop，避免連續同方向轉八次。
HOOP_MID_SEARCH_TH = 1.20
HOOP_MID_SEARCH_TIME = 0.45
HOOP_MID_SEARCH_SETTLE = 0.55
HOOP_MID_SEARCH_MAX = 2

# 每一段都先把籃框對到「當時鏡頭中心」約 x=320。
HOOP_STAGE_CX_TARGET = 320
HOOP_STAGE_CX_TOL = 28
HOOP_STAGE_CENTER_FRAMES = 2

# 分段轉向要小力，不再一次 th=5、連轉 2.8 秒。
HOOP_STAGE_TURN_KP = 0.020
HOOP_STAGE_TURN_MIN = 1.20
HOOP_STAGE_TURN_MAX = 2.00
HOOP_STAGE_TURN_TIME = 0.80
HOOP_STAGE_TURN_SETTLE = 0.55

# 對中後前進一小段，再停下重新對中。
HOOP_STAGE_FORWARD_X = 0.065
HOOP_STAGE_FORWARD_TIME = 2.0
HOOP_STAGE_FORWARD_TH_KP = 0.010
HOOP_STAGE_FORWARD_TH_MAX = 0.65
HOOP_STAGE_LOST_MAX = 2
HOOP_STAGE_MAX_ROUNDS = 3

# 框接近這個尺寸後，才切換到最右側頭角做最後標定。
HOOP_FINAL_ENTER_SIZE = 180.0

# 最終頭轉右後，使用小脈衝尋找籃框。
# 依目前 log，正 th 能把籃框帶到右轉鏡頭中；若實機方向相反只改 SIGN。
HOOP_FINAL_SEARCH_SIGN = 1.0
HOOP_FINAL_SEARCH_TH = 2.0
HOOP_FINAL_SEARCH_TIME = 0.55
HOOP_FINAL_SEARCH_SETTLE = 0.65
HOOP_FINAL_SEARCH_MAX = 10

# 頭部掃完整輪仍找不到框時，身體轉一小段再重掃。
HOOP_SCAN_BODY_TURN_SIGN = 1.0
HOOP_SCAN_BODY_TURN_TH = 2.0
HOOP_SCAN_BODY_TURN_TIME = 0.80
HOOP_SCAN_BODY_TURN_SETTLE = 0.80

# 右側鏡頭最終校正的方向開關。
# 若 cx 修正後越來越偏，只翻 HOOP_FINAL_TH_SIGN。
# 若 forward 後框反而變小，只翻 HOOP_FINAL_FWD_SIGN。
HOOP_FINAL_TH_SIGN = 1.0
HOOP_FINAL_FWD_SIGN = 1.0
HOOP_FINAL_TURN_TIME = 1.20

# 帶球走路偏移補償，先設 0，實測偏移後再調。
HOOP_CARRY_Y_BIAS = 0.0
HOOP_CARRY_TH_BIAS = 0.0

# after grabbing ball, x positive follows the old find_board direction
HOOP_FWD_FAST = 0.080
HOOP_FWD_MID  = 0.050
HOOP_FWD_FINE = 0.025
HOOP_BACK_FINE = -0.020

# ---------- Adjust-mode like gains ----------
# error = target - current
BALL_TH_KP = 0.022
BALL_TH_CLAMP = 1.4

HOOP_TH_KP = 0.030
HOOP_TH_CLAMP = 3.0

# Because op3_ros_utils1.walkVelocities internally does "th = th - 1",
# add 1 here so th_actual is closer to what we ask for.
WALK_TH_ZERO_OFFSET = 1.0

STALE = 0.5

# ---------- 舉重 Adjust_Mode 節奏(核心移植) ----------
SAMPLE_FRAMES   = 7      # 每次決策前取樣幀數(舉重 FINE_SAMPLE_FRAMES=7)
SAMPLE_DT       = 0.05   # 取樣間隔(舉重 sample_bar dt=0.05)
ADJ_SETTLE_TIME = 1.0    # 每次脈衝後停穩時間(舉重脈衝後 rospy.sleep(1.0))

BALL_TURN_STEP_TIME = 0.20   # 球細調:只轉脈衝時間(舉重 FINE 前進段 0.20)
BALL_FWD_STEP_TIME  = 0.10   # 球細調:前進脈衝時間
HOOP_TURN_STEP_TIME = 0.12
HOOP_FWD_STEP_TIME  = 0.25
HOOP_SEARCH_STEP_TIME = 0.30 # 找不到框時的旋轉脈衝
HOOP_SEARCH_SETTLE    = 0.5  # 找框旋轉間隔可以短一點,加快掃視

BALL_ADJ_MAX_ITER = 999   # 球細調 iteration 上限(舉重 FINE_MAX_ITER=24),到了就直接抓
HOOP_ADJ_MAX_ITER = 60   # 框細調 iteration 上限,到了就直接投
BALL_LOST_MAX     = 3    # 細調中連續幾輪取樣皆空算丟失(一輪約 0.35s)
BALL_GAMBLE_W     = 999.0 # 丟失時最後球寬已 >= 這個 = 球在腳邊視野外 -> 賭一把直接抓
                         # (舉重同款: "lost too many, force -> PICK_BAR")

# ---------- Scan mode before ball adjust ----------
# 找不到球時不前進，只做「頭部掃描 + 原地小轉」鎖定球
SCAN_HEAD_PANS = [-0.75, -0.45, -0.15, 0.15, 0.45, 0.75, 0.45, 0.15, -0.15, -0.45]
SCAN_INTERVAL = 0.90
SCAN_TURN_TH = 0.0
SCAN_TURN_TIME = 0.0
BALL_LOCK_FRAMES = 4
SCAN_SETTLE_TIME = 0.25
BALL_SCAN_MIN_CONF = 0.75
BALL_SCAN_MIN_Y = 55
BALL_SCAN_MIN_SIZE = 30

# ---------- Turn-to-ball after scan lock ----------
# 掃描鎖定球之後,頭持續盯球、身體用脈衝轉向頭的方向,
# 頭 pan 幾乎歸零(球在正前方)才進 find_ball。節奏同舉重:脈衝->停穩->取樣。
TURN_PAN_KP      = 0.8    # 停穩後的中位數取樣可信,頭一次修掉大部分偏差

# 身體追球旋轉
TURN_BODY_KP = 3.5
TURN_TH_CLAMP = 5.0
TURN_TH_MIN = 1.8
# 大角度才連續轉 3 秒
TURN_COARSE_PAN = 0.30    
TURN_COARSE_TIME = 4.5
# 接近正面後改成短脈衝，避免轉過頭

TURN_FINE_TIME = 1.5
TURN_SETTLE_TIME = 1.0
# 旋轉姿態
TURN_Z_MOVE_AMPLITUDE = 0.014
TURN_Z_OFFSET = 0.034
TURN_HIP_PITCH = 5.0

TURN_DONE_PAN    = 0.22   # 頭 pan 小於這個(rad) = 球已在正前方
TURN_DONE_FRAMES = 2      # 連續幾輪達標才算轉完
# 籃框轉向只需一次達標，避免已經轉正後又多轉一輪
HOOP_TURN_DONE_FRAMES = 1

# 籃框接近正面後若仍需微調，只轉短時間
HOOP_TURN_FINE_TIME = 0.80
# 籃框定位專用柔和旋轉，不影響抓球參數
HOOP_GENTLE_TURN_KP = 2.0
HOOP_GENTLE_TH_MIN = 1.0
HOOP_GENTLE_TH_MAX = 1.3
HOOP_GENTLE_FINE_PAN = 0.25
HOOP_GENTLE_TIME_NORMAL = 0.45
HOOP_GENTLE_TIME_FINE = 0.25
TURN_LOST_TOL    = 3      # 取樣制,3 輪 ≈ 1 秒
PAN_LIMIT        = 1.35


# ---------- Ball continuous approach ----------
# 轉向完成後先連續接近，不要每 0.15 秒就停一次
BALL_APPROACH_HEAD_TILT = ae_to_rad(1690)  # 約 -0.55 rad

BALL_APPROACH_X = -0.022
BALL_APPROACH_X_START = -0.010

BALL_APPROACH_STOP_SIZE = 60.0
BALL_APPROACH_STOP_FRAMES = 3

BALL_APPROACH_TH_KP = 0.010
BALL_APPROACH_TH_CLAMP = 0.70
BALL_APPROACH_X_TOL = 35

BALL_APPROACH_TH_SIGN = 1.0

BALL_APPROACH_Z = 0.022
BALL_APPROACH_Z_OFFSET = 0.030
BALL_APPROACH_HIP_PITCH = 7.0

BALL_APPROACH_LOST_MAX = 8
BALL_APPROACH_TIMEOUT = 10.0

# ---------- Ball close Adjust_Mode ----------
# 球到這個範圍後，不再使用一般走路，改用微步校正
BALL_FINE_ENTER_Y = 220
BALL_FINE_ENTER_SIZE = 72.0

# 使用前面 turn_to_ball 的旋轉方向與姿態
BALL_FINE_TH_SIGN = 1.0
BALL_FINE_TH_KP = 0.025
BALL_FINE_TH_MIN = 1.5
BALL_FINE_TH_MAX = 2.2

# 每次旋轉只走一小段
BALL_FINE_TURN_TIME = 0.40

# 本程式負 x 為前進、正 x 為後退
BALL_FINE_FORWARD_X = -0.010
BALL_FINE_BACK_X = 0.015

BALL_FINE_FORWARD_TIME = 0.12
BALL_FINE_BACK_TIME = 1.0

# 旋轉時輕微後退，抵銷自然前移
BALL_FINE_TURN_X_COMP = 0.006

# 每次動作後停穩再取樣
BALL_FINE_SETTLE_TIME = 1.0

# 先鎖定 cx，再調整 cy；調 cy 時若 cx 漂掉就回 cx
BALL_CX_LOCK_FRAMES = 2

# 調 cy 時，cx 超過 24 px 就回去修 cx
BALL_CX_RETURN_TOL = 24

BALL_PHASE_SETTLE_TIME = 0.45

# 球過度靠近時直接停止測試
BALL_SAFE_Y = 405

BALL_FORCE_PICK_X_TOL = 35
BALL_FORCE_PICK_Y_MIN = 380
BALL_FORCE_PICK_Y_MAX = 425


BALL_CX_FINE_ZONE = 40  # cx 接近目標後使用較小的旋轉脈衝
BALL_CX_TURN_TIME_NORMAL = 0.8# 誤差較大時的旋轉時間
BALL_CX_TURN_TIME_FINE = 0.4# 接近目標時的旋轉時間
# CX 近距離專用旋轉限制
BALL_CX_TH_MIN = 1.2
BALL_CX_TH_MAX = 1.6

# 在 cx 階段若球已經太近，先後退再繼續修 cx
BALL_CX_SAFETY_Y_MARGIN = 30

# ---------- State variables ----------
last_ball_x = 320
last_ball_y = 240
last_ball_w = 0
last_ball_h = 0

last_hoop_x = 320
last_hoop_y = 240
last_hoop_w = 0
last_hoop_h = 0

ball_ok_count = 0
hoop_ok_count = 0
lost_count = 0
scan_idx = 0
scan_last_time = 0.0
scan_settle_until = 0.0
ball_lock_count = 0
scan_pan_now = SCAN_HEAD_PANS[0]   # 目前掃描頭角(鎖定時用來還原球的方向)
turn_pan = 0.0                     # turn_to_ball 狀態中頭盯球的 pan
turn_done_count = 0
ball_adj_iter = 0                  # 球細調 iteration 計數
ball_adjust_phase = "cx"  # 近距離調整子階段："cx" 或 "cy"
ball_cx_ok_count = 0 # cx 連續達標次數
hoop_adj_iter = 0                  # 框細調 iteration 計數
last_good_ball_w = 0.0             # 最後一次看到球的寬(丟失賭一把用)
# Hoop scan / turn state
hoop_scan_idx = 0
hoop_scan_last_time = 0.0
hoop_lock_count = 0

hoop_scan_pan_now = 0.0
hoop_stage_head_pan = 0.0
hoop_stage_center_count = 0
hoop_stage_round = 0
hoop_stage_lost_count = 0

# 第一次看到框後只做一次 5 秒原地旋轉。
hoop_initial_turn_done = False
hoop_initial_turn_sign = HOOP_INITIAL_TURN_DEFAULT_SIGN

# 中間頭角校正狀態。
hoop_mid_stage_done = False
hoop_mid_center_count = 0
hoop_mid_search_count = 0

hoop_final_search_count = 0
hoop_turn_pan = 0.0
hoop_turn_done_count = 0

track_ball_x = 320
track_ball_y = 240
track_ball_w = 0.0
track_ball_h = 0.0

track_hoop_x = 320
track_hoop_y = 240
track_hoop_w = 0.0
track_hoop_h = 0.0

yolo = YoloReceiver(port=5005)


def set_head_ae(pan_ae=None, tilt_ae=None):
    pan = 0.0 if pan_ae is None else ae_to_rad(pan_ae)
    tilt = 0.0 if tilt_ae is None else ae_to_rad(tilt_ae)
    robot.setJointPos(["head_pan", "head_tilt"], [pan, tilt])


def set_head_rad(pan, tilt):
    robot.setJointPos(["head_pan", "head_tilt"], [pan, tilt])

def capture_right_arm_pose():
    global right_arm_hold_enabled
    global right_arm_hold_pos

    missing = []

    for joint_name in RIGHT_ARM_JOINTS:
        if joint_name not in present_joint_pos:
            missing.append(joint_name)

    if len(missing) > 0:
        print(
            "[RIGHT_ARM] cannot capture, missing joints: %s" %
            str(missing)
        )
        return False

    right_arm_hold_pos = [
        present_joint_pos["r_sho_pitch"],
        present_joint_pos["r_sho_roll"],
        present_joint_pos["r_el"],
        present_joint_pos["r_gripper"] + RIGHT_GRIPPER_HOLD_OFFSET
    ]

    right_arm_hold_enabled = True

    print(
        "[RIGHT_ARM] captured "
        "pitch=%.3f roll=%.3f elbow=%.3f gripper=%.3f" %
        (
            right_arm_hold_pos[0],
            right_arm_hold_pos[1],
            right_arm_hold_pos[2],
            right_arm_hold_pos[3]
        )
    )

    return True


def lock_right_arm_module_once():
    """
    Assign the right arm to direct joint control only once.

    Do not call this while walking is active. Repeated service calls here
    can make walking_module switch between Enable/Disable and report
    "walking module is not ready".
    """
    global right_arm_module_locked

    if not right_arm_hold_enabled:
        return False

    if right_arm_hold_pos is None:
        return False

    if right_arm_module_locked:
        return True

    robot.setJointsControlModule(
        RIGHT_ARM_JOINTS,
        ["none", "none", "none", "none"]
    )

    rospy.sleep(0.25)
    right_arm_module_locked = True

    print("[RIGHT_ARM] module locked once")
    return True


def apply_right_arm_hold():
    """
    Refresh only the desired joint position.

    This function intentionally does not call setJointsControlModule().
    It is safe to call during walking loops.
    """
    if not right_arm_hold_enabled:
        return

    if right_arm_hold_pos is None:
        return

    robot.setJointPos(
        RIGHT_ARM_JOINTS,
        right_arm_hold_pos
    )


def monitor_right_gripper_effort(label):
    """
    回傳 True 代表判定可能掉球。

    預設 GRIPPER_DROP_CHECK_ENABLED=False，只每秒列印 effort，
    先收集「有球」與「沒球」的實測值後再開啟自動判斷。
    """
    global gripper_low_count
    global last_gripper_effort_print
    global gripper_effort_missing_printed

    effort = present_joint_eff.get("r_gripper")

    if effort is None:
        if GRIPPER_EFFORT_DEBUG and not gripper_effort_missing_printed:
            print(
                "[GRIPPER_EFFORT] no r_gripper effort data; "
                "drop detection disabled"
            )
            gripper_effort_missing_printed = True
        return False

    now = time.time()

    if (
        GRIPPER_EFFORT_DEBUG and
        now - last_gripper_effort_print >=
        GRIPPER_EFFORT_PRINT_INTERVAL
    ):
        print(
            "[GRIPPER_EFFORT] state=%s effort=%.4f abs=%.4f" %
            (
                label,
                effort,
                abs(effort)
            )
        )
        last_gripper_effort_print = now

    if not GRIPPER_DROP_CHECK_ENABLED:
        gripper_low_count = 0
        return False

    if abs(effort) < GRIPPER_DROP_EFFORT_MIN:
        gripper_low_count += 1
    else:
        gripper_low_count = 0

    if gripper_low_count >= GRIPPER_DROP_FRAMES:
        print(
            "[BALL_DROP] r_gripper effort low %.4f, "
            "count=%d/%d" %
            (
                effort,
                gripper_low_count,
                GRIPPER_DROP_FRAMES
            )
        )
        return True

    return False


def prepare_leg_walking_after_pick():
    """
    抓球後只把 12 個腿關節交給 walking_module。

    不呼叫 setGeneralControlModule("walking_module")，
    避免 walking_module 在切換瞬間接管右肩、右肘與夾爪。
    """
    global right_arm_module_locked

    robot.walkStop()
    rospy.sleep(0.20)

    # 先立刻鎖住右手，再啟用腿部 walking_module。
    right_arm_module_locked = False
    lock_right_arm_module_once()
    apply_right_arm_hold()

    robot.setJointsControlModule(
        LEG_JOINTS,
        ["walking_module"]
    )

    rospy.sleep(0.70)

    robot.setJointsControlModule(
        ["head_pan", "head_tilt"],
        ["none", "none"]
    )

    rospy.sleep(0.20)
    apply_right_arm_hold()

    print(
        "[WALKING] leg joints assigned to walking_module; "
        "right arm remains held"
    )


def walk_cmd(x=0.0, y=0.0, th_actual=0.0, hip_pitch=10, balance=True):
    # th_actual is the desired body turn in degrees-like command.
    robot.walkVelocities(
        x=x,
        y=y,
        th=th_actual + WALK_TH_ZERO_OFFSET,
        hip_pitch=hip_pitch,
        balance=balance
    )


def stop_walk():
    robot.walkStop()
    rospy.sleep(0.05)


def pulse_walk(x=0.0, y=0.0, th_actual=0.0, step_time=0.20, wait_time=ADJ_SETTLE_TIME, hip_pitch=10, balance=True):
    # 舉重 Adjust_Mode 節奏: 一次小脈衝 -> walkStop -> 停穩(預設 1.0s)再取樣。
    apply_right_arm_hold()

    walk_cmd(x=x, y=y, th_actual=th_actual, hip_pitch=hip_pitch, balance=balance)
    robot.walkStart()
    rospy.sleep(step_time)
    robot.walkStop()
    rospy.sleep(wait_time)

def turn_to_ball_walk(th_actual, duration):
    """
    追球專用原地旋轉：
    使用較高姿態連續旋轉 duration 秒，
    停止後再讓外層重新取樣。
    """
    apply_right_arm_hold()

    print(
        "[TURN_HIGH] th=%.2f duration=%.2f "
        "z_amp=%.3f z_offset=%.3f hip=%.1f" %
        (
            th_actual,
            duration,
            TURN_Z_MOVE_AMPLITUDE,
            TURN_Z_OFFSET,
            TURN_HIP_PITCH
        )
    )

    # 先設定較輕的起步參數，再啟動 walking
    robot.walkVelocities(
        x=0.0,
        y=0.0,
        th=th_actual * 0.45 + WALK_TH_ZERO_OFFSET,
        z_move_amplitude=TURN_Z_MOVE_AMPLITUDE,
        balance=True,
        z_offset=TURN_Z_OFFSET,
        hip_pitch=TURN_HIP_PITCH
    )

    robot.walkStart()
    rospy.sleep(0.35)

    t0 = time.time()

    while (
        time.time() - t0 < duration and
        not rospy.is_shutdown()
    ):
        elapsed = time.time() - t0

        # 前 0.5 秒逐漸增加旋轉力量，減少往後頓
        if elapsed < 0.5:
            gain = 0.45 + 0.55 * (elapsed / 0.5)
        else:
            gain = 1.0

        robot.walkVelocities(
            x=0.0,
            y=0.0,
            th=th_actual * gain + WALK_TH_ZERO_OFFSET,
            z_move_amplitude=TURN_Z_MOVE_AMPLITUDE,
            balance=True,
            z_offset=TURN_Z_OFFSET,
            hip_pitch=TURN_HIP_PITCH
        )

        rospy.sleep(0.05)

    robot.walkStop()
    rospy.sleep(TURN_SETTLE_TIME)


def ball_fine_move(x_cmd, th_actual, duration):

    apply_right_arm_hold()

    robot.walkVelocities(
        x=x_cmd,
        y=0.0,
        th=th_actual + WALK_TH_ZERO_OFFSET,
        z_move_amplitude=TURN_Z_MOVE_AMPLITUDE,
        balance=True,
        z_offset=TURN_Z_OFFSET,
        hip_pitch=TURN_HIP_PITCH
    )

    robot.walkStart()
    t0 = time.time()

    while (
        time.time() - t0 < duration and
        not rospy.is_shutdown()
    ):
        robot.walkVelocities(
            x=x_cmd,
            y=0.0,
            th=th_actual + WALK_TH_ZERO_OFFSET,
            z_move_amplitude=TURN_Z_MOVE_AMPLITUDE,
            balance=True,
            z_offset=TURN_Z_OFFSET,
            hip_pitch=TURN_HIP_PITCH
        )
        rospy.sleep(0.05)

    robot.walkStop()
    rospy.sleep(BALL_FINE_SETTLE_TIME)


def get_fresh_det():
    det = yolo.get_latest()
    if det is None:
        return None, None
    if det.get("age", 999) >= STALE:
        return None, None
    return det.get("ball"), det.get("hoop")


def sample_ball_median(n=SAMPLE_FRAMES, dt=SAMPLE_DT):
    # 舉重 sample_bar_stable 的做法: 多幀取樣取中位數,濾掉 YOLO 單幀跳動。
    xs, ys, ws, hs = [], [], [], []
    for _ in range(n):
        ball, _ = get_fresh_det()
        if ball is not None:
            xs.append(float(ball[0]))
            ys.append(float(ball[1]))
            ws.append(float(ball[2]))
            hs.append(float(ball[3]))
        rospy.sleep(dt)
    if not xs:
        return None, None, None, None
    return (int(np.median(xs)), int(np.median(ys)),
            float(np.median(ws)), float(np.median(hs)))


def sample_hoop_median(n=SAMPLE_FRAMES, dt=SAMPLE_DT):
    xs, ys, ws, hs = [], [], [], []
    for _ in range(n):
        _, hoop = get_fresh_det()
        if hoop is not None:
            xs.append(float(hoop[0]))
            ys.append(float(hoop[1]))
            ws.append(float(hoop[2]))
            hs.append(float(hoop[3]))
        rospy.sleep(dt)
    if not xs:
        return None, None, None, None
    return (int(np.median(xs)), int(np.median(ys)),
            float(np.median(ws)), float(np.median(hs)))


def valid_ball_for_scan(ball):
    if ball is None:
        return False
    x = int(ball[0])
    y = int(ball[1])
    w = float(ball[2])
    h = float(ball[3])
    conf = float(ball[4]) if len(ball) >= 5 else 1.0

    # 避免掃頭時誤把畫面邊緣、太上方或太小的球影像鎖定
    if conf < BALL_SCAN_MIN_CONF:
        return False
    if y < BALL_SCAN_MIN_Y:
        return False
    if max(w, h) < BALL_SCAN_MIN_SIZE:
        return False
    if x < 40 or x > 600:
        return False
    return True

def valid_hoop_for_scan(hoop):
    if hoop is None:
        return False

    x = int(hoop[0])
    y = int(hoop[1])
    w = float(hoop[2])
    h = float(hoop[3])

    conf = float(hoop[4]) if len(hoop) >= 5 else 1.0

    if conf < HOOP_SCAN_MIN_CONF:
        return False

    if max(w, h) < HOOP_SCAN_MIN_SIZE:
        return False

    if x < 30 or x > 610:
        return False

    return True


def smooth_box(prefix, box, alpha=0.6):
    # return smoothed x,y,w,h for ball/hoop (掃描鎖定用)
    global track_ball_x, track_ball_y, track_ball_w, track_ball_h
    global track_hoop_x, track_hoop_y, track_hoop_w, track_hoop_h

    x = int(box[0])
    y = int(box[1])
    w = float(box[2])
    h = float(box[3])

    if prefix == "ball":
        track_ball_x = int(alpha * track_ball_x + (1.0 - alpha) * x)
        track_ball_y = int(alpha * track_ball_y + (1.0 - alpha) * y)
        track_ball_w = alpha * track_ball_w + (1.0 - alpha) * w
        track_ball_h = alpha * track_ball_h + (1.0 - alpha) * h
        return track_ball_x, track_ball_y, track_ball_w, track_ball_h
    else:
        track_hoop_x = int(alpha * track_hoop_x + (1.0 - alpha) * x)
        track_hoop_y = int(alpha * track_hoop_y + (1.0 - alpha) * y)
        track_hoop_w = alpha * track_hoop_w + (1.0 - alpha) * w
        track_hoop_h = alpha * track_hoop_h + (1.0 - alpha) * h
        return track_hoop_x, track_hoop_y, track_hoop_w, track_hoop_h


def choose_ball_forward(w, h):
    # w/h smaller than target = ball still far, move forward.
    size = max(w, h)
    target = max(BALL_W_TARGET, BALL_H_TARGET)

    if size < target - 35:
        return BALL_FWD_FAST
    elif size < target - 18:
        return BALL_FWD_MID
    elif size < target - 6:
        return BALL_FWD_FINE
    elif size > target + 18:
        return BALL_BACK_FINE
    else:
        return 0.0


def choose_hoop_forward(w, h):
    # w/h smaller than target = hoop still far, move forward.
    size = max(w, h)
    target = max(HOOP_W_TARGET, HOOP_H_TARGET)

    if size < target - 100:
        return HOOP_FWD_FAST
    elif size < target - 55:
        return HOOP_FWD_MID
    elif size < target - 20:
        return HOOP_FWD_FINE
    elif size > target + 45:
        return HOOP_BACK_FINE
    else:
        return 0.0


def play_action_page(page_num, hold_right_after=False):
    global right_arm_module_locked

    robot.walkStop()
    rospy.sleep(0.30)

    robot.setGeneralControlModule("action_module")
    rospy.sleep(0.30)

    print("[ACTION] playMotion %d" % page_num)

    robot.playMotion(
        page_num,
        wait_for_end=True
    )

    if hold_right_after:
        # Action 100 結束後立刻記錄並鎖住右手。
        rospy.sleep(0.10)

        if capture_right_arm_pose():
            prepare_leg_walking_after_pick()
            rospy.sleep(0.30)
            return True

        # 擷取失敗時停在原地，不切換全身 walking_module，
        # 避免右手被走路模組拉走。
        print(
            "[RIGHT_ARM] capture failed; "
            "keep action pose and stop"
        )

        robot.walkStop()
        rospy.sleep(0.30)
        return False

    else:
        # Action 99 投球後不需要再保持球，恢復一般 walking_module。
        rospy.sleep(0.20)
        robot.setGeneralControlModule("walking_module")
        rospy.sleep(0.80)

        robot.setJointsControlModule(
            ["head_pan", "head_tilt"],
            ["none", "none"]
        )

        rospy.sleep(0.30)
        return True


print("[CALIB] BALL_HEAD_TILT rad=%.3f from AE=1556" % BALL_HEAD_TILT)
print("[CALIB] HOOP_HEAD_PAN rad=%.3f from AE=991, HOOP_HEAD_TILT rad=%.3f from AE=1974" % (
    HOOP_HEAD_PAN, HOOP_HEAD_TILT
))

# Start with ball grabbing head pose
set_head_rad(BALL_HEAD_PAN, BALL_HEAD_TILT)
rospy.sleep(0.5)

while not rospy.is_shutdown():

    # 抓球後定期顯示夾爪 effort。
    # 自動掉球判斷預設關閉，先收集實測值。
    if (
        right_arm_hold_enabled and
        situation in (
            "hoop_right_turn_8s",
            "scan_hoop",
            "turn_to_hoop",
            "pre_final_left_turn",
            "initial_hoop_turn",
            "center_hoop_stage",
            "advance_hoop_segment",
            "acquire_mid_hoop",
            "acquire_final_hoop",
            "find_board"
        )
    ):
        if monitor_right_gripper_effort(situation):
            robot.walkStop()
            situation = "stop"
            print("[STAGE] possible ball drop -> stop")
            continue

    if situation == "wait":
        print(z_gyro)
        if robot.buttonCheck("user"):
            gyro_offset = z_gyro
            situation = "scan_ball"
            ball_ok_count = 0
            lost_count = 0
            scan_idx = 0
            scan_last_time = 0.0
            scan_settle_until = 0.0
            ball_lock_count = 0
            ball_adj_iter = 0
            ball_adjust_phase = "cx"
            ball_cx_ok_count = 0
            hoop_adj_iter = 0
            print("[STAGE] wait -> scan_ball")
            robot.setGeneralControlModule("walking_module")
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            scan_pan_now = SCAN_HEAD_PANS[0]
            set_head_rad(scan_pan_now, SCAN_HEAD_TILT)
            stop_walk()
            rospy.sleep(0.3)

    # ============================================================
    # 0) Scan_Mode for ball: do not walk blindly.
    #    First lock the ball by head sweep + small in-place rotation.
    # ============================================================
    if situation == "scan_ball":
        ball, hoop = get_fresh_det()
        now = time.time()

        # 剛換頭角後先等一下，避免用上一個頭角的舊畫面直接鎖定
        if now < scan_settle_until:
            stop_walk()
            rospy.sleep(0.03)
            continue

        if valid_ball_for_scan(ball):
            x, y, w, h = smooth_box("ball", ball, alpha=0.30)
            ball_lock_count += 1
            print("[BALL_SCAN_LOCK] x=%d y=%d w=%.1f h=%.1f lock=%d/%d" % (
                x, y, w, h, ball_lock_count, BALL_LOCK_FRAMES
            ))
            stop_walk()

            if ball_lock_count >= BALL_LOCK_FRAMES:
                # 鎖定成功：不直接回正頭（會丟方向），改記下球的方向
                # = 目前掃描頭角 + 影像內偏差換算（hfov 90 度、640px）
                px_corr = (320.0 - float(x)) * (45.0 / 320.0) * np.pi / 180.0
                turn_pan = float(np.clip(scan_pan_now + px_corr, -PAN_LIMIT, PAN_LIMIT))
                set_head_rad(turn_pan, SCAN_HEAD_TILT)
                rospy.sleep(0.3)
                turn_done_count = 0
                lost_count = 0
                situation = "turn_to_ball"
                print("[STAGE] scan_ball -> turn_to_ball, ball direction pan=%.2f" % turn_pan)

        else:
            ball_lock_count = 0

            # 只掃頭與原地轉，不往前走，避免盲走撞球。
            if now - scan_last_time >= SCAN_INTERVAL:
                pan = SCAN_HEAD_PANS[scan_idx % len(SCAN_HEAD_PANS)]
                scan_pan_now = pan
                set_head_rad(pan, SCAN_HEAD_TILT)
                print("[BALL_SCAN] no valid ball, head_pan=%.2f head_tilt=%.2f" % (pan, SCAN_HEAD_TILT))
                scan_idx += 1
                scan_last_time = now
                scan_settle_until = now + SCAN_SETTLE_TIME

                # 每掃完一輪，原地小轉一下再掃下一輪。
                if scan_idx % len(SCAN_HEAD_PANS) == 0:
                    print("[BALL_SCAN] full sweep no valid ball -> stop and rescan")
                    stop_walk()
                    scan_settle_until = time.time() + 0.6
                else:
                    stop_walk()

    # ============================================================
    # 0.5) Turn_to_ball: 原地轉向球(舉重節奏: 取樣->脈衝->停穩1s)
    #      頭持續盯球，身體一次一小脈衝轉向頭的方向，
    #      頭 pan 幾乎歸零（球在正前方）才進 find_ball。
    # ============================================================
    if situation == "turn_to_ball":
        bx, by, bw, bh = sample_ball_median()

        if bx is not None:
            lost_count = 0
            last_good_ball_w = bw

            # 頭持續盯球：中位數取樣可信，一次修掉大部分偏差
            px_corr = (320.0 - float(bx)) * (45.0 / 320.0) * np.pi / 180.0
            turn_pan = float(np.clip(turn_pan + px_corr * TURN_PAN_KP, -PAN_LIMIT, PAN_LIMIT))
            set_head_rad(turn_pan, SCAN_HEAD_TILT)
            rospy.sleep(0.25)

            if abs(turn_pan) < TURN_DONE_PAN:
                turn_done_count += 1
            else:
                turn_done_count = 0

            print("[TURN2BALL] x=%d w=%.1f pan=%.2f done=%d/%d" % (
                bx, bw, turn_pan, turn_done_count, TURN_DONE_FRAMES
            ))

            if turn_done_count >= TURN_DONE_FRAMES:
                # 球已在正前方 → 頭回正，進入細調
                stop_walk()
                set_head_rad(BALL_HEAD_PAN, BALL_HEAD_TILT)
                rospy.sleep(0.35)
                ball_ok_count = 0
                lost_count = 0
                ball_adj_iter = 0

                set_head_rad(BALL_HEAD_PAN, BALL_APPROACH_HEAD_TILT)
                rospy.sleep(0.4)

                situation = "approach_ball"
                print("[STAGE] turn_to_ball -> approach_ball / continuous walk")
            
            else:
                # 一次小脈衝轉向頭的方向，停穩再取樣（舉重節奏）
                raw_th = turn_pan * TURN_BODY_KP

                # 只要還沒對正，就保證至少有足夠的旋轉力量
                if abs(raw_th) < TURN_TH_MIN:
                    if raw_th > 0:
                        raw_th = TURN_TH_MIN
                    else:
                        raw_th = -TURN_TH_MIN

                th_cmd = float(np.clip(
                    raw_th,
                    -TURN_TH_CLAMP,
                    TURN_TH_CLAMP
                ))

                # 大角度：連續旋轉 3 秒
                # 小角度：只轉 0.5 秒，防止超調
                pan_abs = abs(turn_pan)

                if pan_abs >= 0.55:
                    turn_time = 6.0
                    turn_mode = "LARGE"

                elif pan_abs >= 0.30:
                    turn_time = 5.0
                    turn_mode = "COARSE"

                else:
                    turn_time = 2.5
                    turn_mode = "FINE"

                print(
                    "[TURN2BALL_%s] pan=%.2f th=%.2f time=%.2f" %
                    (turn_mode, turn_pan, th_cmd, turn_time)
                )

                turn_to_ball_walk(
                    th_actual=th_cmd,
                    duration=turn_time
                )

                # 身體轉完後先把頭回到中間
                # 下一輪 sample_ball_median() 重新判斷球的位置
                turn_pan = 0.0
                set_head_rad(turn_pan, SCAN_HEAD_TILT)
                rospy.sleep(0.3)

        else:
            lost_count += 1
            stop_walk()
            print("[TURN2BALL] lost ball (%d/%d)" % (lost_count, TURN_LOST_TOL))
            if lost_count >= TURN_LOST_TOL:
                situation = "scan_ball"
                scan_idx = 0
                scan_last_time = 0.0
                ball_lock_count = 0
                scan_pan_now = SCAN_HEAD_PANS[0]
                set_head_rad(scan_pan_now, SCAN_HEAD_TILT)
                scan_settle_until = time.time() + SCAN_SETTLE_TIME
                print("[TURN2BALL] lost too long -> scan_ball")



    # ============================================================
    # 0.8) Approach_ball
    #      轉向完成後連續直走，球接近後才進微步 Adjust_Mode
    # ============================================================
    if situation == "approach_ball":
        robot.setGeneralControlModule("walking_module")
        set_head_rad(BALL_HEAD_PAN, BALL_APPROACH_HEAD_TILT)
        rospy.sleep(0.3)

        approach_lost = 0
        approach_close_count = 0
        approach_start = time.time()

        robot.walkStart()

        while (
            situation == "approach_ball" and
            not rospy.is_shutdown()
        ):
            elapsed = time.time() - approach_start

            if elapsed >= BALL_APPROACH_TIMEOUT:
                print("[BALL_APPROACH] timeout -> stop and enter fine mode")
                robot.walkStop()

                ball_ok_count = 0
                ball_cx_ok_count = 0
                ball_adjust_phase = "cx"

                lost_count = 0
                ball_adj_iter = 0

                situation = "find_ball"
                break

            ball, _ = get_fresh_det()

            if ball is None:
                approach_lost += 1

                # 暫時掉偵測先原地等，不繼續盲走
                robot.walkVelocities(
                    x=0.0,
                    y=0.0,
                    th=WALK_TH_ZERO_OFFSET,
                    z_move_amplitude=BALL_APPROACH_Z,
                    balance=True,
                    z_offset=BALL_APPROACH_Z_OFFSET,
                    hip_pitch=BALL_APPROACH_HIP_PITCH
                )

                print("[BALL_APPROACH] lost %d/%d" % (
                    approach_lost,
                    BALL_APPROACH_LOST_MAX
                ))

                if approach_lost >= BALL_APPROACH_LOST_MAX:
                    robot.walkStop()
                    situation = "scan_ball"
                    scan_idx = 0
                    scan_last_time = 0.0
                    ball_lock_count = 0
                    print("[BALL_APPROACH] lost too long -> scan_ball")
                    break

                rospy.sleep(0.05)
                continue

            approach_lost = 0

            bx = int(ball[0])
            by = int(ball[1])
            bw = float(ball[2])
            bh = float(ball[3])

            size = max(bw, bh)
            x_err = BALL_APPROACH_CX_TARGET - bx

            last_good_ball_w = bw
            last_ball_x = bx
            last_ball_y = by
            last_ball_w = int(bw)
            last_ball_h = int(bh)

            # 連續三次達到門檻才停止，避免單幀框跳大
            if size >= BALL_APPROACH_STOP_SIZE:
                approach_close_count += 1
            else:
                approach_close_count = 0

            print(
                "[BALL_APPROACH] x=%d y=%d w=%.1f h=%.1f "
                "size=%.1f close=%d/%d" %
                (
                    bx, by, bw, bh,
                    size,
                    approach_close_count,
                    BALL_APPROACH_STOP_FRAMES
                )
            )

            if approach_close_count >= BALL_APPROACH_STOP_FRAMES:
                robot.walkStop()
                rospy.sleep(0.5)

                # 接近後切換成抓球低頭角度
                set_head_rad(BALL_HEAD_PAN, BALL_HEAD_TILT)
                rospy.sleep(0.6)

                ball_ok_count = 0
                ball_cx_ok_count = 0
                ball_adjust_phase = "cx"
                lost_count = 0
                ball_adj_iter = 0
                situation = "find_ball"

                print("[STAGE] approach_ball -> find_ball / Fine Adjust_Mode")
                break

            # 水平偏差不要太大；小偏差時持續直走
            if abs(x_err) > BALL_APPROACH_X_TOL:
                th_actual = float(np.clip(
                    BALL_APPROACH_TH_SIGN *
                    BALL_APPROACH_TH_KP *
                    float(x_err),
                    -BALL_APPROACH_TH_CLAMP,
                    BALL_APPROACH_TH_CLAMP
                ))
            else:
                th_actual = 0.0

            # 起步前 0.8 秒較柔，之後才給正常速度
            if elapsed < 0.8:
                ratio = elapsed / 0.8
                x_cmd = (
                    BALL_APPROACH_X_START +
                    (BALL_APPROACH_X - BALL_APPROACH_X_START) * ratio
                )
            else:
                x_cmd = BALL_APPROACH_X

            # 偏得太多時減速，不要邊大轉邊高速接近
            if abs(x_err) > 70:
                x_cmd *= 0.45

            robot.walkVelocities(
                x=x_cmd,
                y=0.0,
                th=th_actual + WALK_TH_ZERO_OFFSET,
                z_move_amplitude=BALL_APPROACH_Z,
                balance=True,
                z_offset=BALL_APPROACH_Z_OFFSET,
                hip_pitch=BALL_APPROACH_HIP_PITCH
            )

            rospy.sleep(0.05)


    

    # ============================================================
    # 1) Adjust_Mode for ball(舉重節奏)
    #    每輪: 取樣中位數 -> 判斷 -> 一次脈衝 -> 停穩 1.0s
    # ============================================================
    if situation == "find_ball":
        ball_adj_iter += 1

        # ★ iteration 上限(舉重 FINE_MAX_ITER): 修太多輪就直接抓
        if ball_adj_iter > BALL_ADJ_MAX_ITER:
            print("[BALL_ADJ] max iter %d, force -> pick_ball" % BALL_ADJ_MAX_ITER)
            stop_walk()
            ball_adj_iter = 0
            situation = "pick_ball"
        else:
            set_head_rad(BALL_HEAD_PAN, BALL_HEAD_TILT)
            bx, by, bw, bh = sample_ball_median()

            if bx is not None:
                lost_count = 0
                last_good_ball_w = bw
                last_ball_x = bx
                last_ball_y = by
                last_ball_w = int(bw)
                last_ball_h = int(bh)

                x_err = BALL_CX_TARGET - bx
                y_err = BALL_CY_TARGET - by
                w_err = BALL_W_TARGET - bw
                h_err = BALL_H_TARGET - bh

                # 球是否已經進入近距離微調區
                fine_mode = (
                    by >= BALL_FINE_ENTER_Y or
                    max(bw, bh) >= BALL_FINE_ENTER_SIZE
                )

                cx_ok = abs(x_err) <= BALL_X_TOL
                cy_ok = abs(y_err) <= BALL_Y_TOL

                print(
                    "[BALL_ADJ] iter=%d phase=%s "
                    "x=%d y=%d w=%.1f h=%.1f "
                    "ex=%d ey=%d ew=%.1f eh=%.1f "
                    "cx_ok=%s cy_ok=%s final=%d/%d" %
                    (
                        ball_adj_iter,
                        ball_adjust_phase,
                        bx, by, bw, bh,
                        x_err, y_err, w_err, h_err,
                        str(cx_ok),
                        str(cy_ok),
                        ball_ok_count,
                        BALL_STABLE_FRAMES
                    )
                )

                # 連續兩輪真正達到標定範圍才抓球
                if fine_mode:

                    # =====================================================
                    # 安全停止：球已經過度靠近
                    # =====================================================
                    if by >= BALL_SAFE_Y:
                        stop_walk()

                        # 球已過近，但位置仍在可抓範圍
                        if (
                            abs(x_err) <= BALL_FORCE_PICK_X_TOL and
                            BALL_FORCE_PICK_Y_MIN <= by <= BALL_FORCE_PICK_Y_MAX and
                            bw >= 90.0 and
                            bh >= 90.0
                        ):
                            print(
                                "[BALL_SAFE_FORCE_PICK] "
                                "x=%d y=%d w=%.1f h=%.1f ex=%d "
                                "-> pick_ball" %
                                (
                                    bx,
                                    by,
                                    bw,
                                    bh,
                                    x_err
                                )
                            )

                            ball_ok_count = 0
                            ball_cx_ok_count = 0
                            situation = "pick_ball"

                        else:
                            print(
                                "[BALL_SAFE_FORCE_PICK_ANY] "
                                "x=%d y=%d w=%.1f h=%.1f ex=%d "
                                "-> pick_ball" %
                                (
                                    bx,
                                    by,
                                    bw,
                                    bh,
                                    x_err
                                )
                            )

                            ball_ok_count = 0
                            ball_cx_ok_count = 0
                            situation = "pick_ball"

                    # =====================================================
                    # Phase 1：先修正 CX
                    # =====================================================
                    elif ball_adjust_phase == "cx":

                        # CX 階段不能累積最後抓球確認次數
                        ball_ok_count = 0

                        # cx 還沒完成，但球已經太靠近：先後退
                        if by > BALL_CY_TARGET + BALL_CX_SAFETY_Y_MARGIN:
                            ball_cx_ok_count = 0

                            print(
                                "[BALL_CX_SAFE_BACK] "
                                "x=%d y=%d ex=%d ey=%d "
                                "back=%.3f time=%.2f" %
                                (
                                    bx,
                                    by,
                                    x_err,
                                    y_err,
                                    BALL_FINE_BACK_X,
                                    BALL_FINE_BACK_TIME
                                )
                            )

                            ball_fine_move(
                                x_cmd=BALL_FINE_BACK_X,
                                th_actual=0.0,
                                duration=BALL_FINE_BACK_TIME
                            )

                        # cx 已進入容許範圍，停著確認
                        elif cx_ok:
                            ball_cx_ok_count += 1
                            stop_walk()

                            print(
                                "[BALL_CX_HOLD] "
                                "x=%d target=%d ex=%d lock=%d/%d" %
                                (
                                    bx,
                                    BALL_CX_TARGET,
                                    x_err,
                                    ball_cx_ok_count,
                                    BALL_CX_LOCK_FRAMES
                                )
                            )

                            # cx 連續兩輪達標後切換到 cy
                            if ball_cx_ok_count >= BALL_CX_LOCK_FRAMES:
                                ball_adjust_phase = "cy"
                                ball_cx_ok_count = 0
                                ball_ok_count = 0

                                print("[BALL_PHASE] cx aligned -> cy")

                            rospy.sleep(BALL_PHASE_SETTLE_TIME)

                        # cx 尚未達標：使用和 turn_to_ball 相同的旋轉方式
                        else:
                            ball_cx_ok_count = 0

                            # 將影像 pixel 誤差轉換成頭部水平角度誤差
                            # 與前面 turn_to_ball 的 px_corr 算法相同
                            cx_pan_error = (
                                float(x_err) *
                                (45.0 / 320.0) *
                                np.pi / 180.0
                            )

                            # 使用 turn_to_ball 相同的旋轉增益
                            raw_th = cx_pan_error * TURN_BODY_KP

                            # 使用 turn_to_ball 相同的旋轉死區補償
                            if abs(raw_th) < BALL_CX_TH_MIN:
                                if raw_th > 0:
                                    raw_th = BALL_CX_TH_MIN
                                else:
                                    raw_th = -BALL_CX_TH_MIN

                            th_cmd = float(np.clip(
                                raw_th,
                                -BALL_CX_TH_MAX,
                                BALL_CX_TH_MAX
                            ))

                            # 接近目標時縮短旋轉時間，避免超調
                            if abs(x_err) <= BALL_CX_FINE_ZONE:
                                turn_time = BALL_CX_TURN_TIME_FINE
                                turn_label = "FINE"
                            else:
                                turn_time = BALL_CX_TURN_TIME_NORMAL
                                turn_label = "NORMAL"

                            print(
                                "[BALL_CX_TURN_%s] "
                                "x=%d target=%d ex=%d "
                                "pan_error=%.3f th=%.2f time=%.2f" %
                                (
                                    turn_label,
                                    bx,
                                    BALL_CX_TARGET,
                                    x_err,
                                    cx_pan_error,
                                    th_cmd,
                                    turn_time
                                )
                            )

                            # 使用與一開始方位旋轉相同的函式與姿態
                            turn_to_ball_walk(
                                th_actual=th_cmd,
                                duration=turn_time
                            )

                    # =====================================================
                    # Phase 2：再修正 CY
                    # =====================================================
                    elif ball_adjust_phase == "cy":

                        # 調 cy 時 cx 漂掉，立即回 cx
                        if abs(x_err) > BALL_CX_RETURN_TOL:
                            stop_walk()

                            ball_adjust_phase = "cx"
                            ball_cx_ok_count = 0
                            ball_ok_count = 0

                            print(
                                "[BALL_PHASE] cx drifted "
                                "x=%d target=%d ex=%d -> cx" %
                                (
                                    bx,
                                    BALL_CX_TARGET,
                                    x_err
                                )
                            )

                            rospy.sleep(BALL_PHASE_SETTLE_TIME)

                        # cy 太小，往前
                        elif by < BALL_CY_TARGET - BALL_Y_TOL:
                            ball_ok_count = 0

                            print(
                                "[BALL_CY_FORWARD] "
                                "x=%d y=%d target_y=%d "
                                "ex=%d ey=%d fwd=%.3f time=%.2f" %
                                (
                                    bx,
                                    by,
                                    BALL_CY_TARGET,
                                    x_err,
                                    y_err,
                                    BALL_FINE_FORWARD_X,
                                    BALL_FINE_FORWARD_TIME
                                )
                            )

                            ball_fine_move(
                                x_cmd=BALL_FINE_FORWARD_X,
                                th_actual=0.0,
                                duration=BALL_FINE_FORWARD_TIME
                            )

                        # cy 太大，往後
                        # 球比抓球標定位置稍微更近
                        # 目前後退步態尚未校正，不再執行錯誤的 BACK
                        elif by > BALL_CY_TARGET + BALL_Y_TOL:
                            ball_ok_count = 0
                            stop_walk()

                            # 稍微超過 CY，但水平位置和球大小仍合理
                            # 直接嘗試抓球
                            if (
                                by <= 412 and
                                abs(x_err) <= 25 and
                                bw >= 90.0 and
                                bh >= 90.0
                            ):
                                print(
                                    "[BALL_CLOSE_FORCE_PICK] "
                                    "x=%d y=%d w=%.1f h=%.1f ex=%d "
                                    "-> pick_ball" %
                                    (
                                        bx,
                                        by,
                                        bw,
                                        bh,
                                        x_err
                                    )
                                )

                                ball_ok_count = 0
                                ball_cx_ok_count = 0
                                situation = "pick_ball"

                            else:
                                print(
                                    "[BALL_TOO_CLOSE_FORCE_PICK] "
                                    "x=%d y=%d w=%.1f h=%.1f ex=%d "
                                    "-> pick_ball" %
                                    (
                                        bx,
                                        by,
                                        bw,
                                        bh,
                                        x_err
                                    )
                                )

                                ball_ok_count = 0
                                ball_cx_ok_count = 0
                                situation = "pick_ball"

                        # cy 已達標，最後確認 cx 是否仍達標
                        else:
                            if cx_ok and cy_ok:
                                ball_ok_count += 1
                                stop_walk()

                                print(
                                    "[BALL_FINAL_HOLD] "
                                    "x=%d y=%d ok=%d/%d" %
                                    (
                                        bx,
                                        by,
                                        ball_ok_count,
                                        BALL_STABLE_FRAMES
                                    )
                                )

                                if ball_ok_count >= BALL_STABLE_FRAMES:
                                    situation = "pick_ball"

                                    print(
                                        "[STAGE] cx and cy aligned -> pick_ball"
                                    )

                                else:
                                    rospy.sleep(BALL_FINE_SETTLE_TIME)

                            else:
                                # cy 到位，但 cx 不在嚴格範圍，回去修 cx
                                stop_walk()

                                ball_adjust_phase = "cx"
                                ball_cx_ok_count = 0
                                ball_ok_count = 0

                                print(
                                    "[BALL_PHASE] cy ready but cx not ready "
                                    "x=%d ex=%d -> cx" %
                                    (
                                        bx,
                                        x_err
                                    )
                                )

                                rospy.sleep(BALL_PHASE_SETTLE_TIME)

                else:
                    # =====================================================
                    # 尚未進入近距離模式，保留原本遠距離接近
                    # =====================================================
                    ball_adjust_phase = "cx"
                    ball_cx_ok_count = 0
                    ball_ok_count = 0

                    if abs(x_err) > 40:
                        raw_th = (
                            BALL_FINE_TH_SIGN *
                            BALL_FINE_TH_KP *
                            float(x_err)
                        )

                        if abs(raw_th) < BALL_FINE_TH_MIN:
                            if raw_th > 0:
                                raw_th = BALL_FINE_TH_MIN
                            else:
                                raw_th = -BALL_FINE_TH_MIN

                        th_cmd = float(np.clip(
                            raw_th,
                            -BALL_FINE_TH_MAX,
                            BALL_FINE_TH_MAX
                        ))

                        print(
                            "[BALL_APPROACH_TURN] ex=%d th=%.2f" %
                            (
                                x_err,
                                th_cmd
                            )
                        )

                        ball_fine_move(
                            x_cmd=BALL_FINE_TURN_X_COMP,
                            th_actual=th_cmd,
                            duration=0.55
                        )

                    else:
                        print("[BALL_APPROACH_FORWARD] x=-0.012 time=0.15")

                        ball_fine_move(
                            x_cmd=-0.012,
                            th_actual=0.0,
                            duration=0.15
                        )

            else:
                lost_count += 1
                ball_ok_count = 0
                ball_cx_ok_count = 0
                stop_walk()
                print("[BALL_ADJ] no detection (%d/%d), last_w=%.1f" % (
                    lost_count, BALL_LOST_MAX, last_good_ball_w))

                if lost_count >= BALL_LOST_MAX:
                    if last_good_ball_w >= BALL_FINE_ENTER_SIZE:
                        print("[BALL_ADJ] close ball lost -> small back and retry")

                        ball_fine_move(
                            x_cmd=BALL_FINE_BACK_X,
                            th_actual=0.0,
                            duration=0.15
                        )

                        lost_count = 0
                        ball_ok_count = 0
                        ball_cx_ok_count = 0
                        ball_adjust_phase = "cx"

                    else:
                        print("[BALL_ADJ] lost while far -> scan_ball")

                        lost_count = 0
                        ball_adj_iter = 0

                        ball_ok_count = 0
                        ball_cx_ok_count = 0
                        ball_adjust_phase = "cx"

                        situation = "scan_ball"
                        scan_idx = 0
                        scan_last_time = 0.0
                        scan_settle_until = 0.0
                        ball_lock_count = 0
                        scan_pan_now = 0.0

    # ============================================================
    # 2) Grab ball with Action Editor page 100
    # ============================================================
    if situation == "pick_ball":
        stop_walk()
        set_head_rad(BALL_HEAD_PAN, BALL_HEAD_TILT)
        rospy.sleep(0.5)

        print("[PICK] target was x=%d y=%d w=%d h=%d, play page %d" % (
            last_ball_x, last_ball_y, last_ball_w, last_ball_h, PICK_MOTION
        ))

        pick_hold_ok = play_action_page(
            PICK_MOTION,
            hold_right_after=True
        )

        if not pick_hold_ok:
            situation = "stop"
            print(
                "[STAGE] pick action finished but "
                "right arm hold setup failed -> stop"
            )
            continue

        # 抓球後先在正前方附近找框，再採用：
        # 對中 -> 前進一段 -> 再對中 -> 再前進。
        hoop_ok_count = 0
        lost_count = 0
        hoop_adj_iter = 0

        hoop_scan_idx = 0
        hoop_scan_last_time = 0.0
        hoop_lock_count = 0

        hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[0]
        hoop_stage_head_pan = hoop_scan_pan_now
        hoop_stage_center_count = 0
        hoop_stage_round = 0
        hoop_stage_lost_count = 0

        hoop_initial_turn_done = False
        hoop_initial_turn_sign = HOOP_INITIAL_TURN_DEFAULT_SIGN

        hoop_mid_stage_done = False
        hoop_mid_center_count = 0
        hoop_mid_search_count = 0

        hoop_final_search_count = 0
        hoop_turn_pan = 0.0
        hoop_turn_done_count = 0

        set_head_rad(
            0.0,
            HOOP_HEAD_TILT
        )

        apply_right_arm_hold()
        rospy.sleep(0.5)

        situation = "hoop_right_turn_8s"

        print(
            "[STAGE] pick_ball -> hoop_right_turn_8s"
        )

    # ============================================================
    # 2.2) Grabbed ball: fixed right rotation for 8 seconds
    # ============================================================
    if situation == "hoop_right_turn_8s":
        robot.walkStop()
        apply_right_arm_hold()

        set_head_rad(
            0.0,
            HOOP_HEAD_TILT
        )

        rospy.sleep(0.30)

        print(
            "[HOOP_RIGHT_TURN] th=%.2f time=%.2f" %
            (
                HOOP_RIGHT_TURN_TH,
                HOOP_RIGHT_TURN_TIME
            )
        )

        turn_to_ball_walk(
            th_actual=HOOP_RIGHT_TURN_TH,
            duration=HOOP_RIGHT_TURN_TIME
        )

        hoop_scan_idx = 0
        hoop_scan_last_time = 0.0
        hoop_lock_count = 0
        hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[0]

        set_head_rad(
            hoop_scan_pan_now,
            HOOP_HEAD_TILT
        )

        rospy.sleep(0.50)
        situation = "scan_hoop"

        print(
            "[STAGE] hoop_right_turn_8s -> scan_hoop"
        )

    # ============================================================
    # 2.3) Scan hoop near the forward-facing direction
    # ============================================================
    if situation == "scan_hoop":
        _, hoop = get_fresh_det()
        now = time.time()

        apply_right_arm_hold()

        if valid_hoop_for_scan(hoop):
            hx, hy, hw, hh = smooth_box(
                "hoop",
                hoop,
                alpha=0.30
            )

            hoop_lock_count += 1

            print(
                "[HOOP_SCAN_LOCK] "
                "pan=%.2f x=%d y=%d w=%.1f h=%.1f "
                "lock=%d/%d" %
                (
                    hoop_scan_pan_now,
                    hx,
                    hy,
                    hw,
                    hh,
                    hoop_lock_count,
                    HOOP_LOCK_FRAMES
                )
            )

            if hoop_lock_count >= HOOP_LOCK_FRAMES:
                robot.walkStop()
                apply_right_arm_hold()

                # 目前掃描頭角，加上籃框在畫面中的水平偏差。
                px_corr = (
                    (320.0 - float(hx)) *
                    (45.0 / 320.0) *
                    np.pi / 180.0
                )

                hoop_turn_pan = float(
                    np.clip(
                        hoop_scan_pan_now + px_corr,
                        -PAN_LIMIT,
                        PAN_LIMIT
                    )
                )

                set_head_rad(
                    hoop_turn_pan,
                    HOOP_HEAD_TILT
                )

                rospy.sleep(0.30)

                hoop_turn_done_count = 0
                lost_count = 0
                hoop_lock_count = 0
                situation = "turn_to_hoop"

                print(
                    "[STAGE] scan_hoop -> turn_to_hoop "
                    "direction_pan=%.2f" %
                    hoop_turn_pan
                )

        else:
            hoop_lock_count = 0

            if now - hoop_scan_last_time >= HOOP_SCAN_INTERVAL:
                hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[
                    hoop_scan_idx % len(HOOP_SCAN_HEAD_PANS)
                ]

                set_head_rad(
                    hoop_scan_pan_now,
                    HOOP_HEAD_TILT
                )

                print(
                    "[HOOP_SCAN] no valid hoop, "
                    "head_pan=%.2f head_tilt=%.2f" %
                    (
                        hoop_scan_pan_now,
                        HOOP_HEAD_TILT
                    )
                )

                hoop_scan_idx += 1
                hoop_scan_last_time = now

                # 頭掃完整輪仍無籃框：身體小轉一次，再從正前方重掃。
                if (
                    hoop_scan_idx > 0 and
                    hoop_scan_idx % len(HOOP_SCAN_HEAD_PANS) == 0
                ):
                    print(
                        "[HOOP_SCAN] full sweep no hoop "
                        "-> body turn and rescan"
                    )

                    set_head_rad(
                        0.0,
                        HOOP_HEAD_TILT
                    )

                    pulse_walk(
                        x=0.0,
                        y=HOOP_CARRY_Y_BIAS,
                        th_actual=(
                            HOOP_SCAN_BODY_TURN_SIGN *
                            HOOP_SCAN_BODY_TURN_TH +
                            HOOP_CARRY_TH_BIAS
                        ),
                        step_time=HOOP_SCAN_BODY_TURN_TIME,
                        wait_time=HOOP_SCAN_BODY_TURN_SETTLE,
                        hip_pitch=7,
                        balance=True
                    )

                    hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[0]
                    hoop_scan_last_time = time.time()

                    set_head_rad(
                        hoop_scan_pan_now,
                        HOOP_HEAD_TILT
                    )

        rospy.sleep(0.03)

    # ============================================================
    # 2.4) Turn toward hoop, same strategy as turn_to_ball
    # ============================================================
    if situation == "turn_to_hoop":
        hx, hy, hw, hh = sample_hoop_median()

        if hx is not None:
            lost_count = 0

            last_hoop_x = hx
            last_hoop_y = hy
            last_hoop_w = int(hw)
            last_hoop_h = int(hh)

            size = max(hw, hh)

            px_corr = (
                (320.0 - float(hx)) *
                (45.0 / 320.0) *
                np.pi / 180.0
            )

            hoop_turn_pan = float(
                np.clip(
                    hoop_turn_pan + px_corr * TURN_PAN_KP,
                    -PAN_LIMIT,
                    PAN_LIMIT
                )
            )

            set_head_rad(
                hoop_turn_pan,
                HOOP_HEAD_TILT
            )

            rospy.sleep(0.25)

            if abs(hoop_turn_pan) < HOOP_TURN_DONE_PAN:
                hoop_turn_done_count += 1
            else:
                hoop_turn_done_count = 0

            print(
                "[TURN2HOOP] "
                "x=%d y=%d w=%.1f h=%.1f "
                "size=%.1f pan=%.2f done=%d/%d" %
                (
                    hx,
                    hy,
                    hw,
                    hh,
                    size,
                    hoop_turn_pan,
                    hoop_turn_done_count,
                    HOOP_TURN_DONE_FRAMES
                )
            )

            # 已轉正，而且籃框尺寸達到門檻：
            # 切換到最終投籃頭角。
            if (
                hoop_turn_done_count >= HOOP_TURN_DONE_FRAMES and
                size >= HOOP_FINAL_SWITCH_SIZE
            ):
                robot.walkStop()
                apply_right_arm_hold()

                set_head_rad(
                    THROW_HEAD_PAN,
                    THROW_HEAD_TILT
                )

                rospy.sleep(0.50)

                hoop_final_search_count = 0
                hoop_ok_count = 0
                hoop_adj_iter = 0
                lost_count = 0
                situation = "pre_final_left_turn"

                print(
                    "[STAGE] turn_to_hoop "
                    "-> pre_final_left_turn size=%.1f" %
                    size
                )

            # 已經大致正對籃框，但尺寸尚未達到門檻：
            # 使用帶球階段既有的小前進參數，前進一小步後重新辨識。
            elif hoop_turn_done_count >= HOOP_TURN_DONE_FRAMES:
                robot.walkStop()
                apply_right_arm_hold()

                print(
                    "[TURN2HOOP_FORWARD_DISTANCE] "
                    "size=%.1f need=%.1f "
                    "x=%.3f time=%.2f" %
                    (
                        size,
                        HOOP_FINAL_SWITCH_SIZE,
                        HOOP_FWD_FINE,
                        HOOP_FWD_STEP_TIME
                    )
                )

                ball_fine_move(
                    x_cmd=HOOP_FWD_FINE,
                    th_actual=HOOP_CARRY_TH_BIAS,
                    duration=HOOP_FWD_STEP_TIME
                )

                # 前進後重新確認一次籃框方向與尺寸。
                hoop_turn_done_count = 0

            # 尚未轉正：使用籃框專用的小力短旋轉。
            else:
                raw_th = (
                    hoop_turn_pan *
                    HOOP_GENTLE_TURN_KP
                )

                if abs(raw_th) < HOOP_GENTLE_TH_MIN:
                    if raw_th > 0:
                        raw_th = HOOP_GENTLE_TH_MIN
                    else:
                        raw_th = -HOOP_GENTLE_TH_MIN

                th_cmd = float(
                    np.clip(
                        raw_th,
                        -HOOP_GENTLE_TH_MAX,
                        HOOP_GENTLE_TH_MAX
                    )
                )

                if abs(hoop_turn_pan) <= HOOP_GENTLE_FINE_PAN:
                    turn_time = HOOP_GENTLE_TIME_FINE
                    turn_mode = "FINE"
                else:
                    turn_time = HOOP_GENTLE_TIME_NORMAL
                    turn_mode = "NORMAL"

                print(
                    "[TURN2HOOP_%s] "
                    "pan=%.2f th=%.2f time=%.2f" %
                    (
                        turn_mode,
                        hoop_turn_pan,
                        th_cmd,
                        turn_time
                    )
                )

                ball_fine_move(
                    x_cmd=0.0,
                    th_actual=th_cmd,
                    duration=turn_time
                )

                # 身體轉完後頭回到正面，重新偵測籃框。
                hoop_turn_pan = 0.0

                set_head_rad(
                    hoop_turn_pan,
                    HOOP_HEAD_TILT
                )

                rospy.sleep(0.30)

        else:
            lost_count += 1
            robot.walkStop()
            apply_right_arm_hold()

            print(
                "[TURN2HOOP] lost hoop %d/%d" %
                (
                    lost_count,
                    TURN_LOST_TOL
                )
            )

            if lost_count >= TURN_LOST_TOL:
                hoop_scan_idx = 0
                hoop_scan_last_time = 0.0
                hoop_lock_count = 0
                hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[0]

                set_head_rad(
                    hoop_scan_pan_now,
                    HOOP_HEAD_TILT
                )

                lost_count = 0
                situation = "scan_hoop"

                print(
                    "[TURN2HOOP] lost too long -> scan_hoop"
                )


    # ============================================================
    # 2.4) First hoop detection: rotate in place for 5 seconds
    #      This happens only once after Action 100.
    # ============================================================
    if situation == "initial_hoop_turn":
        robot.walkStop()
        apply_right_arm_hold()

        # 轉動時頭先回正，避免頭已經在側邊又繼續增加視角落差。
        set_head_rad(
            0.0,
            HOOP_HEAD_TILT
        )

        rospy.sleep(0.30)

        print(
            "[HOOP_INITIAL_TURN] "
            "th=%.2f time=%.2f" %
            (
                hoop_initial_turn_sign *
                HOOP_INITIAL_TURN_TH,
                HOOP_INITIAL_TURN_TIME
            )
        )

        turn_to_ball_walk(
            th_actual=(
                hoop_initial_turn_sign *
                HOOP_INITIAL_TURN_TH
            ),
            duration=HOOP_INITIAL_TURN_TIME
        )

        hoop_initial_turn_done = True

        # 旋轉完成後一定重新掃描，不能直接盲目前進。
        hoop_scan_idx = 0
        hoop_scan_last_time = 0.0
        hoop_lock_count = 0
        hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[0]

        set_head_rad(
            hoop_scan_pan_now,
            HOOP_HEAD_TILT
        )

        rospy.sleep(0.50)
        situation = "scan_hoop"

        print(
            "[STAGE] initial_hoop_turn finished "
            "-> scan_hoop"
        )

    # ============================================================
    # 2.5) Center hoop in the CURRENT camera angle
    # ============================================================
    if situation == "center_hoop_stage":
        set_head_rad(
            hoop_stage_head_pan,
            HOOP_HEAD_TILT
        )

        hx, hy, hw, hh = sample_hoop_median()

        if hx is None:
            hoop_stage_lost_count += 1
            robot.walkStop()
            apply_right_arm_hold()

            print(
                "[HOOP_STAGE_CENTER] lost %d/%d" %
                (
                    hoop_stage_lost_count,
                    HOOP_STAGE_LOST_MAX
                )
            )

            if hoop_stage_lost_count >= HOOP_STAGE_LOST_MAX:
                hoop_scan_idx = 0
                hoop_scan_last_time = 0.0
                hoop_lock_count = 0
                hoop_stage_lost_count = 0
                hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[0]

                set_head_rad(
                    hoop_scan_pan_now,
                    HOOP_HEAD_TILT
                )

                situation = "scan_hoop"

                print(
                    "[HOOP_STAGE_CENTER] lost too long -> scan_hoop"
                )

        else:
            hoop_stage_lost_count = 0

            last_hoop_x = hx
            last_hoop_y = hy
            last_hoop_w = int(hw)
            last_hoop_h = int(hh)

            size = max(hw, hh)
            x_err = HOOP_STAGE_CX_TARGET - hx

            if abs(x_err) <= HOOP_STAGE_CX_TOL:
                hoop_stage_center_count += 1
            else:
                hoop_stage_center_count = 0

            print(
                "[HOOP_STAGE_CENTER] "
                "round=%d pan=%.2f x=%d target=%d ex=%d "
                "size=%.1f hold=%d/%d" %
                (
                    hoop_stage_round,
                    hoop_stage_head_pan,
                    hx,
                    HOOP_STAGE_CX_TARGET,
                    x_err,
                    size,
                    hoop_stage_center_count,
                    HOOP_STAGE_CENTER_FRAMES
                )
            )

            if abs(x_err) <= HOOP_STAGE_CX_TOL:
                robot.walkStop()
                apply_right_arm_hold()

                if hoop_stage_center_count >= HOOP_STAGE_CENTER_FRAMES:
                    # 先經過縮小後的中間頭角，不直接跳到 -1.62 rad。
                    if (
                        not hoop_mid_stage_done and
                        size >= HOOP_MID_ENTER_SIZE
                    ):
                        hoop_mid_center_count = 0
                        hoop_mid_search_count = 0

                        set_head_rad(
                            HOOP_MID_HEAD_PAN,
                            HOOP_HEAD_TILT
                        )

                        rospy.sleep(0.70)
                        situation = "acquire_mid_hoop"

                        print(
                            "[STAGE] center_hoop_stage "
                            "-> acquire_mid_hoop "
                            "size=%.1f round=%d mid_pan=%.2f" %
                            (
                                size,
                                hoop_stage_round,
                                HOOP_MID_HEAD_PAN
                            )
                        )

                    elif (
                        hoop_mid_stage_done and
                        size >= HOOP_FINAL_ENTER_SIZE
                    ):
                        hoop_final_search_count = 0

                        set_head_rad(
                            HOOP_HEAD_PAN,
                            HOOP_HEAD_TILT
                        )

                        rospy.sleep(0.80)
                        situation = "acquire_final_hoop"

                        print(
                            "[STAGE] center_hoop_stage "
                            "-> acquire_final_hoop "
                            "size=%.1f round=%d" %
                            (
                                size,
                                hoop_stage_round
                            )
                        )

                    else:
                        situation = "advance_hoop_segment"

                        print(
                            "[STAGE] center_hoop_stage "
                            "-> advance_hoop_segment"
                        )

                else:
                    print(
                        "[HOOP_STAGE_HOLD] "
                        "x=%d ex=%d hold=%d/%d" %
                        (
                            hx,
                            x_err,
                            hoop_stage_center_count,
                            HOOP_STAGE_CENTER_FRAMES
                        )
                    )

                    rospy.sleep(HOOP_STAGE_TURN_SETTLE)

            else:
                raw_th = (
                    HOOP_STAGE_TURN_KP *
                    float(x_err)
                )

                if abs(raw_th) < HOOP_STAGE_TURN_MIN:
                    if raw_th > 0:
                        raw_th = HOOP_STAGE_TURN_MIN
                    else:
                        raw_th = -HOOP_STAGE_TURN_MIN

                th_cmd = float(
                    np.clip(
                        raw_th,
                        -HOOP_STAGE_TURN_MAX,
                        HOOP_STAGE_TURN_MAX
                    )
                )

                hoop_pan_error = (
                    float(x_err) *
                    (45.0 / 320.0) *
                    np.pi / 180.0
                )

                pan_abs = abs(hoop_pan_error)

                if pan_abs >= 0.55:
                    turn_time = 5.5
                    turn_mode = "LARGE"

                elif pan_abs >= 0.30:
                    turn_time = 4.5
                    turn_mode = "COARSE"

                else:
                    turn_time = 2.0
                    turn_mode = "FINE"

                print(
                    "[HOOP_STAGE_TURN_%s] "
                    "x=%d ex=%d pan_error=%.2f "
                    "th=%.2f time=%.2f" %
                    (
                        turn_mode,
                        hx,
                        x_err,
                        hoop_pan_error,
                        th_cmd,
                        turn_time
                    )
                )

                turn_to_ball_walk(
                    th_actual=th_cmd + HOOP_CARRY_TH_BIAS,
                    duration=turn_time
                )

    # ============================================================
    # 2.7) Walk one segment, then stop and center again
    # ============================================================
    if situation == "advance_hoop_segment":
        apply_right_arm_hold()

        set_head_rad(
            hoop_stage_head_pan,
            HOOP_HEAD_TILT
        )

        rospy.sleep(0.25)

        segment_lost = 0
        segment_start = time.time()

        robot.walkStart()

        while (
            situation == "advance_hoop_segment" and
            not rospy.is_shutdown()
        ):
            elapsed = time.time() - segment_start

            _, hoop = get_fresh_det()

            if hoop is None:
                segment_lost += 1

                robot.walkVelocities(
                    x=0.0,
                    y=0.0,
                    th=WALK_TH_ZERO_OFFSET,
                    z_move_amplitude=BALL_APPROACH_Z,
                    balance=True,
                    z_offset=BALL_APPROACH_Z_OFFSET,
                    hip_pitch=7.0
                )

                print(
                    "[HOOP_STAGE_FORWARD] lost %d/%d" %
                    (
                        segment_lost,
                        HOOP_STAGE_LOST_MAX
                    )
                )

                if segment_lost >= HOOP_STAGE_LOST_MAX:
                    robot.walkStop()
                    apply_right_arm_hold()

                    # 不再盲目前進，也不立刻把頭切回完全正面。
                    # 保留目前頭角，回到該視角重新抓框。
                    hoop_stage_center_count = 0
                    hoop_stage_lost_count = 0
                    situation = "center_hoop_stage"

                    print(
                        "[HOOP_STAGE_FORWARD] lost target "
                        "-> stop and center again at pan=%.2f" %
                        hoop_stage_head_pan
                    )

                    break

                rospy.sleep(0.05)
                continue

            segment_lost = 0

            hx = int(hoop[0])
            hy = int(hoop[1])
            hw = float(hoop[2])
            hh = float(hoop[3])

            size = max(hw, hh)
            x_err = HOOP_STAGE_CX_TARGET - hx

            last_hoop_x = hx
            last_hoop_y = hy
            last_hoop_w = int(hw)
            last_hoop_h = int(hh)

            th_actual = float(
                np.clip(
                    HOOP_STAGE_FORWARD_TH_KP *
                    float(x_err),
                    -HOOP_STAGE_FORWARD_TH_MAX,
                    HOOP_STAGE_FORWARD_TH_MAX
                )
            )

            print(
                "[HOOP_STAGE_FORWARD] "
                "round=%d t=%.2f/%.2f "
                "x=%d ex=%d size=%.1f "
                "walk_x=%.3f th=%.2f" %
                (
                    hoop_stage_round,
                    elapsed,
                    HOOP_STAGE_FORWARD_TIME,
                    hx,
                    x_err,
                    size,
                    HOOP_STAGE_FORWARD_X,
                    th_actual
                )
            )

            # 走到中段時先切到中間角度重新偵測與對中。
            if (
                not hoop_mid_stage_done and
                size >= HOOP_MID_ENTER_SIZE
            ):
                robot.walkStop()
                apply_right_arm_hold()

                hoop_mid_center_count = 0
                hoop_mid_search_count = 0

                set_head_rad(
                    HOOP_MID_HEAD_PAN,
                    HOOP_HEAD_TILT
                )

                rospy.sleep(0.70)
                situation = "acquire_mid_hoop"

                print(
                    "[STAGE] advance_hoop_segment "
                    "-> acquire_mid_hoop size=%.1f "
                    "mid_pan=%.2f" %
                    (
                        size,
                        HOOP_MID_HEAD_PAN
                    )
                )

                break

            # 中間角已校正完成，而且框夠大，才切到最終右視角。
            if (
                hoop_mid_stage_done and
                size >= HOOP_FINAL_ENTER_SIZE
            ):
                robot.walkStop()
                apply_right_arm_hold()

                hoop_final_search_count = 0

                set_head_rad(
                    HOOP_HEAD_PAN,
                    HOOP_HEAD_TILT
                )

                rospy.sleep(0.80)
                situation = "acquire_final_hoop"

                print(
                    "[STAGE] advance_hoop_segment "
                    "-> acquire_final_hoop size=%.1f" %
                    size
                )

                break

            if elapsed >= HOOP_STAGE_FORWARD_TIME:
                robot.walkStop()
                apply_right_arm_hold()

                hoop_stage_round += 1
                hoop_stage_center_count = 0
                situation = "center_hoop_stage"

                print(
                    "[STAGE] advance segment finished "
                    "-> center_hoop_stage round=%d" %
                    hoop_stage_round
                )

                break

            # Soft start while carrying the ball.
            if elapsed < 0.80:
                gain = 0.45 + 0.55 * (elapsed / 0.80)
            else:
                gain = 1.0

            robot.walkVelocities(
                x=HOOP_STAGE_FORWARD_X * gain,
                y=HOOP_CARRY_Y_BIAS,
                th=(
                    th_actual +
                    HOOP_CARRY_TH_BIAS +
                    WALK_TH_ZERO_OFFSET
                ),
                z_move_amplitude=0.020,
                balance=True,
                z_offset=0.028,
                hip_pitch=7.0
            )

            rospy.sleep(0.05)

    # ============================================================
    # 2.8) Intermediate view correction
    #      Head pan is halfway between front view and final right view.
    #      No forward motion is allowed until the hoop is seen and centered.
    # ============================================================
    if situation == "acquire_mid_hoop":
        robot.walkStop()
        apply_right_arm_hold()

        set_head_rad(
            HOOP_MID_HEAD_PAN,
            HOOP_HEAD_TILT
        )

        hx, hy, hw, hh = sample_hoop_median()

        if hx is None:
            hoop_mid_center_count = 0
            hoop_mid_search_count += 1

            # 第一次往正方向找，第二次往反方向找，
            # 避免像原 log 一樣連續八次往同方向轉走。
            if hoop_mid_search_count % 2 == 1:
                mid_search_sign = 1.0
            else:
                mid_search_sign = -1.0

            print(
                "[HOOP_MID_SEARCH] no hoop %d/%d "
                "pan=%.2f th=%.2f time=%.2f" %
                (
                    hoop_mid_search_count,
                    HOOP_MID_SEARCH_MAX,
                    HOOP_MID_HEAD_PAN,
                    mid_search_sign *
                    HOOP_MID_SEARCH_TH,
                    HOOP_MID_SEARCH_TIME
                )
            )

            # 看不到框時只原地小轉，不准前進。
            # 使用和前段 CX 校正相同的力量、時間與停穩時間。
            pulse_walk(
                x=0.0,
                y=HOOP_CARRY_Y_BIAS,
                th_actual=(
                    mid_search_sign *
                    HOOP_MID_SEARCH_TH +
                    HOOP_CARRY_TH_BIAS
                ),
                step_time=HOOP_MID_SEARCH_TIME,
                wait_time=HOOP_MID_SEARCH_SETTLE,
                hip_pitch=7,
                balance=True
            )

            set_head_rad(
                HOOP_MID_HEAD_PAN,
                HOOP_HEAD_TILT
            )

            if hoop_mid_search_count >= HOOP_MID_SEARCH_MAX:
                robot.walkStop()
                apply_right_arm_hold()

                hoop_scan_idx = 0
                hoop_scan_last_time = 0.0
                hoop_lock_count = 0
                hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[0]

                set_head_rad(
                    hoop_scan_pan_now,
                    HOOP_HEAD_TILT
                )

                situation = "scan_hoop"

                print(
                    "[HOOP_MID_SEARCH] max search "
                    "-> scan_hoop"
                )

        else:
            hoop_mid_search_count = 0

            last_hoop_x = hx
            last_hoop_y = hy
            last_hoop_w = int(hw)
            last_hoop_h = int(hh)

            size = max(hw, hh)
            x_err = HOOP_MID_CX_TARGET - hx

            if abs(x_err) <= HOOP_MID_CX_TOL:
                hoop_mid_center_count += 1
            else:
                hoop_mid_center_count = 0

            print(
                "[HOOP_MID_CENTER] "
                "x=%d target=%d ex=%d size=%.1f "
                "hold=%d/%d pan=%.2f" %
                (
                    hx,
                    HOOP_MID_CX_TARGET,
                    x_err,
                    size,
                    hoop_mid_center_count,
                    HOOP_MID_CENTER_FRAMES,
                    HOOP_MID_HEAD_PAN
                )
            )

            if abs(x_err) <= HOOP_MID_CX_TOL:
                robot.walkStop()
                apply_right_arm_hold()

                if hoop_mid_center_count >= HOOP_MID_CENTER_FRAMES:
                    hoop_mid_stage_done = True
                    hoop_stage_head_pan = HOOP_MID_HEAD_PAN
                    hoop_stage_center_count = 0
                    hoop_stage_lost_count = 0

                    if size >= HOOP_FINAL_ENTER_SIZE:
                        hoop_final_search_count = 0

                        set_head_rad(
                            HOOP_HEAD_PAN,
                            HOOP_HEAD_TILT
                        )

                        rospy.sleep(0.80)
                        situation = "acquire_final_hoop"

                        print(
                            "[STAGE] mid angle aligned "
                            "-> acquire_final_hoop size=%.1f" %
                            size
                        )

                    else:
                        situation = "advance_hoop_segment"

                        print(
                            "[STAGE] mid angle aligned "
                            "-> advance_hoop_segment"
                        )

                else:
                    print(
                        "[HOOP_MID_HOLD] "
                        "x=%d ex=%d hold=%d/%d" %
                        (
                            hx,
                            x_err,
                            hoop_mid_center_count,
                            HOOP_MID_CENTER_FRAMES
                        )
                    )

                    rospy.sleep(HOOP_MID_TURN_SETTLE)

            else:
                raw_th = (
                    HOOP_MID_TH_SIGN *
                    HOOP_MID_TURN_KP *
                    float(x_err)
                )

                if abs(raw_th) < HOOP_MID_TURN_MIN:
                    if raw_th > 0:
                        raw_th = HOOP_MID_TURN_MIN
                    else:
                        raw_th = -HOOP_MID_TURN_MIN

                th_cmd = float(
                    np.clip(
                        raw_th,
                        -HOOP_MID_TURN_MAX,
                        HOOP_MID_TURN_MAX
                    )
                )

                print(
                    "[HOOP_MID_TURN] "
                    "x=%d ex=%d th=%.2f time=%.2f" %
                    (
                        hx,
                        x_err,
                        th_cmd,
                        HOOP_MID_TURN_TIME
                    )
                )

                turn_to_ball_walk(
                    th_actual=th_cmd + HOOP_CARRY_TH_BIAS,
                    duration=HOOP_MID_TURN_TIME
                )

    # ============================================================
    # 2.85) Final head angle, then rotate left for 5 seconds
    # ============================================================
    if situation == "pre_final_left_turn":
        robot.walkStop()
        apply_right_arm_hold()

        # 維持最終投籃頭角
        set_head_rad(
            THROW_HEAD_PAN,
            THROW_HEAD_TILT
        )

        rospy.sleep(0.30)

        print(
            "[PRE_FINAL_LEFT_TURN] "
            "th=%.2f time=%.2f" %
            (
                HOOP_PRE_FINAL_LEFT_TH,
                HOOP_PRE_FINAL_LEFT_TIME
            )
        )

        # 使用與一開始找球相同的旋轉函式與姿態
        turn_to_ball_walk(
            th_actual=HOOP_PRE_FINAL_LEFT_TH,
            duration=HOOP_PRE_FINAL_LEFT_TIME
        )

        robot.walkStop()
        apply_right_arm_hold()

        set_head_rad(
            THROW_HEAD_PAN,
            THROW_HEAD_TILT
        )

        rospy.sleep(0.50)

        hoop_final_search_count = 0
        hoop_ok_count = 0
        hoop_adj_iter = 0
        lost_count = 0
        situation = "acquire_final_hoop"

        print(
            "[STAGE] pre_final_left_turn "
            "-> acquire_final_hoop"
        )


    # ============================================================
    # 2.9) Final acquire with the calibrated right-facing head angle
    # ============================================================
    if situation == "acquire_final_hoop":
        set_head_rad(
            THROW_HEAD_PAN,
            THROW_HEAD_TILT
        )

        hx, hy, hw, hh = sample_hoop_median()

        if hx is not None:
            robot.walkStop()
            apply_right_arm_hold()

            last_hoop_x = hx
            last_hoop_y = hy
            last_hoop_w = int(hw)
            last_hoop_h = int(hh)

            hoop_ok_count = 0
            hoop_adj_iter = 0
            lost_count = 0
            situation = "find_board"

            print(
                "[STAGE] acquire_final_hoop -> find_board "
                "x=%d y=%d w=%.1f h=%.1f" %
                (
                    hx,
                    hy,
                    hw,
                    hh
                )
            )

        else:
            hoop_final_search_count += 1

            print(
                "[HOOP_FINAL_SEARCH] no hoop %d/%d "
                "th=%.2f time=%.2f" %
                (
                    hoop_final_search_count,
                    HOOP_FINAL_SEARCH_MAX,
                    HOOP_FINAL_SEARCH_SIGN *
                    HOOP_FINAL_SEARCH_TH,
                    HOOP_FINAL_SEARCH_TIME
                )
            )

            pulse_walk(
                x=0.0,
                y=HOOP_CARRY_Y_BIAS,
                th_actual=(
                    HOOP_FINAL_SEARCH_SIGN *
                    HOOP_FINAL_SEARCH_TH +
                    HOOP_CARRY_TH_BIAS
                ),
                step_time=HOOP_FINAL_SEARCH_TIME,
                wait_time=HOOP_FINAL_SEARCH_SETTLE,
                hip_pitch=7,
                balance=True
            )

            set_head_rad(
                THROW_HEAD_PAN,
                THROW_HEAD_TILT
            )

            if hoop_final_search_count >= HOOP_FINAL_SEARCH_MAX:
                robot.walkStop()
                apply_right_arm_hold()

                hoop_scan_idx = 0
                hoop_scan_last_time = 0.0
                hoop_lock_count = 0
                hoop_scan_pan_now = HOOP_SCAN_HEAD_PANS[0]
                hoop_stage_round = 0

                # 最終視角完全找不到時，重新走中間角流程。
                hoop_mid_stage_done = False
                hoop_mid_center_count = 0
                hoop_mid_search_count = 0

                set_head_rad(
                    hoop_scan_pan_now,
                    HOOP_HEAD_TILT
                )

                situation = "scan_hoop"

                print(
                    "[HOOP_FINAL_SEARCH] max search -> scan_hoop"
                )

    # ============================================================
    # 3) Final hoop adjustment at the calibrated right head angle
    # ============================================================
    if situation == "find_board":
        hoop_adj_iter += 1

        apply_right_arm_hold()

        set_head_rad(
            THROW_HEAD_PAN,
            THROW_HEAD_TILT
        )

        if hoop_adj_iter > HOOP_ADJ_MAX_ITER:
            print(
                "[HOOP_ADJ] max iter %d "
                "-> acquire_final_hoop" %
                HOOP_ADJ_MAX_ITER
            )

            robot.walkStop()
            apply_right_arm_hold()

            hoop_adj_iter = 0
            hoop_ok_count = 0
            lost_count = 0
            hoop_final_search_count = 0
            situation = "acquire_final_hoop"

        else:
            hx, hy, hw, hh = sample_hoop_median()

            if hx is not None:
                lost_count = 0

                last_hoop_x = hx
                last_hoop_y = hy
                last_hoop_w = int(hw)
                last_hoop_h = int(hh)

                x_err = HOOP_CX_TARGET - hx
                y_err = HOOP_CY_TARGET - hy
                w_err = HOOP_W_TARGET - hw
                h_err = HOOP_H_TARGET - hh


                centered = (
                    abs(x_err) <= HOOP_X_TOL and
                    abs(y_err) <= HOOP_Y_TOL
                )

                if centered:
                    hoop_ok_count += 1
                else:
                    hoop_ok_count = 0

                print(
                    "[HOOP_FINAL_ADJ] "
                    "iter=%d x=%d y=%d w=%.1f h=%.1f "
                    "ex=%d ey=%d ew=%.1f eh=%.1f "
                    "ok=%d/%d" %
                    (
                        hoop_adj_iter,
                        hx,
                        hy,
                        hw,
                        hh,
                        x_err,
                        y_err,
                        w_err,
                        h_err,
                        hoop_ok_count,
                        HOOP_STABLE_FRAMES
                    )
                )

                if hoop_ok_count >= HOOP_STABLE_FRAMES:
                    robot.walkStop()
                    apply_right_arm_hold()

                    situation = "throw"

                    print(
                        "[STAGE] final hoop calibration aligned "
                        "-> throw"
                    )

                elif abs(x_err) > HOOP_X_TOL:
                    cx_pan_error = (
                        float(x_err) *
                        (45.0 / 320.0) *
                        np.pi / 180.0
                    )

                    # 使用與抓球 CX 細調相同的旋轉增益。
                    raw_th = (
                        HOOP_FINAL_TH_SIGN *
                        cx_pan_error *
                        TURN_BODY_KP
                    )

                    # 使用與抓球 CX 細調相同的最小旋轉力量。
                    if abs(raw_th) < HOOP_GENTLE_TH_MIN:
                        if raw_th > 0:
                            raw_th = HOOP_GENTLE_TH_MIN
                        else:
                            raw_th = -HOOP_GENTLE_TH_MIN

                    th_cmd = float(
                        np.clip(
                            raw_th,
                            -HOOP_GENTLE_TH_MAX,
                            HOOP_GENTLE_TH_MAX
                        )
                    )

                    # 誤差小於等於 40 px 轉 0.4 秒；
                    # 誤差大於 40 px 轉 0.8 秒。
                    if abs(x_err) <= BALL_CX_FINE_ZONE:
                        turn_time = HOOP_GENTLE_TIME_FINE
                        turn_mode = "FINE"
                    else:
                        turn_time = HOOP_GENTLE_TIME_NORMAL
                        turn_mode = "NORMAL"

                    print(
                        "[HOOP_FINAL_TURN_%s] "
                        "x=%d target=%d ex=%d "
                        "pan_error=%.3f th=%.2f time=%.2f" %
                        (
                            turn_mode,
                            hx,
                            HOOP_CX_TARGET,
                            x_err,
                            cx_pan_error,
                            th_cmd,
                            turn_time
                        )
                    )

                    ball_fine_move(
                        x_cmd=0.0,
                        th_actual=(
                            th_cmd +
                            HOOP_CARRY_TH_BIAS
                        ),
                        duration=turn_time
                    )

                else:
                    robot.walkStop()
                    apply_right_arm_hold()

                    print(
                        "[HOOP_FINAL_NO_FORWARD] "
                        "x=%d y=%d target_x=%d target_y=%d "
                        "ex=%d ey=%d, only rotation allowed" %
                        (
                            hx,
                            hy,
                            HOOP_CX_TARGET,
                            HOOP_CY_TARGET,
                            x_err,
                            y_err
                        )
                    )

                    rospy.sleep(0.40)

            else:
                lost_count += 1
                hoop_ok_count = 0

                print(
                    "[HOOP_FINAL_ADJ] lost %d/%d" %
                    (
                        lost_count,
                        TURN_LOST_TOL
                    )
                )

                if lost_count >= TURN_LOST_TOL:
                    hoop_final_search_count = 0
                    lost_count = 0
                    situation = "acquire_final_hoop"

                    print(
                        "[HOOP_FINAL_ADJ] lost too long "
                        "-> acquire_final_hoop"
                    )

    # ============================================================
    # 4) Throw with Action Editor page 99
    # ============================================================
    if situation == "throw":
        stop_walk()

        # Let Action 99 take control of the right arm.
        right_arm_hold_enabled = False
        right_arm_module_locked = False
        gripper_low_count = 0

        set_head_rad(THROW_HEAD_PAN, THROW_HEAD_TILT)
        rospy.sleep(0.5)

        print("[THROW] target was x=%d y=%d w=%d h=%d, play page %d" % (
            last_hoop_x, last_hoop_y, last_hoop_w, last_hoop_h, THROW_MOTION
        ))

        play_action_page(THROW_MOTION)
        situation = "stop"
        print("[STAGE] throw -> stop")

    if situation == "stop":
        stop_walk()

    rospy.sleep(0.03)

    if robot.buttonCheck("start") and situation != "wait":
        robot.walkStop()
        break