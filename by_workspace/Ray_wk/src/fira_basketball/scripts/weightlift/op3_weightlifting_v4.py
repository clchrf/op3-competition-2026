#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import time
import socket
import json
import threading
import sys

from op3_ros_utils import Robot

import op3_ros_utils
print("[USING_UTILS]", op3_ros_utils.__file__)
from sensor_msgs.msg import Imu

# ============================================================
#  ★★★  賽前調整區  ★★★
# ============================================================

DEBUG_MODE = False

# ── socket 來源 ────────────────────────────────────────────
DET_BIND_HOST = "0.0.0.0"
DET_BIND_PORT = 5005
DET_STALE_SEC = 0.3 #0.6

# ── 頭部角度(rad)─────────────────────────────────────────
HEAD_TILT_WALK      = -0.645   # WALK_FORWARD 第一段(找槓鈴遠) 0.55
HEAD_TILT_CLOSE     = -0.65   # WALK_FORWARD 第二段(找槓鈴近)
HEAD_TILT_ADJUST_0  = -0.75
HEAD_TILT_ADJUST_1  = -0.75
HEAD_TILT_CHECK     = -0.85
HEAD_TILT_PICK_WALK = -1.2   # PICK 後短暫直走的低頭
HEAD_TILT_LIFTWALK  = -1.0   # WALK_WITH_BAR 抬槓走小段時低頭
HEAD_TILT_LINE_NEAR = -1.0   # ★ STOP_AT_LIFT_LINE / WALK_2_FINISH 看地面找線
HEAD_TILT_AFTER_LINE_LOW = -1.2   # 第一次看到線停下後，頭再低一點重新找線
HEAD_TILT_FINISH_LOW_CHECK = -0.95
HEAD_TILT_FINISH_PREWALK = -0.65
HEAD_TILT_FINISH_LINE = -0.90   # 舉過頭後，往終點線走時專用頭角度

HEAD_TILT_SEARCH    = -0.55   # 找不到槓鈴或槓鈴跑到畫面上方時，抬頭搜尋

LIFT_AFTER_LINE_X = 1.5
FINAL_AFTER_LINE_X = 1.0

# ── cY 停止閾值(實測填) ────────────────────────────────────
CY_STOP_FIRST   = 460   # 接近槓鈴第一停 → Adjust_Mode
CY_STOP_SECOND  = 410   # 接近槓鈴第二停 → Adjust_Mode
CY_STOP_FINAL   = 420   # 接近槓鈴最終停 → CHECK_TOUCH
CY_PICK_READY   = 425   # CHECK_TOUCH 判斷可夾取

# ★ 線停止閾值(實測填) ────────────────────────────────────
#   做法:把機器人擺在「準備停的位置」上,印出 line cY,填進來。
#   注意:walkStop 後會慣性多滑一點,所以這個值要設得比實際略小,抓提前量。
LIFT_LINE_STOP_CY  = 380    # STOP_AT_LIFT_LINE:該停下舉過頭的位置(cY)
FINAL_LINE_STOP_CY = 360    # WALK_2_FINISH:第一段看到終點線先停，邏輯比照 LIFT_LINE

LIFT_LINE_SECOND_STOP_CY = 420
FINAL_LINE_SECOND_STOP_CY = 420

# 偵測到停止線後，先停下穩定，再往前補走一段
LINE_STABLE_WAIT = 1.0
LIFT_AFTER_LINE_WALK_TIME  = 1.8   # 偵測到舉重線後，停穩，再往前走 2 秒
FINAL_AFTER_LINE_WALK_TIME = 2.5   # 終點線二次確認後，停穩，再往前走 2 秒
AFTER_LINE_X = 2.5
AFTER_LINE_Y = 0.0



FINISH_PRE_WALK_TIME = 4.0
FINISH_PRE_WALK_X = 1.35
FINISH_IGNORE_LINE_TIME = 0.0
FINISH_ARM_FAR_CY = 360

FINISH_LINE_CRUISE_X = 1.60
FINISH_WALK_Z = 0.026
FINISH_Z_OFFSET = 0.0
FINISH_HIP_PITCH = 7.2
FINISH_FEET_PITCH = -1.2

HEAD_TILT_FINISH_PREWALK = -0.65
HEAD_TILT_FINISH_LINE = -0.90


LINE_CX_KP   = 0.006         # 對準線中心的 y 補正增益
LINE_Y_CLAMP = 0.6          # 對線時 y 補正最大值

# ── 走路參數 ───────────────────────────────────────────────
WALK_LOST_MAX = 6       # 連續看不到 6 次才真的 head up search
WALK_START_WAIT_SEC = 0.5

WALK_FORWARD_X          = 1.5
WALK_FORWARD_Y          = 0.0
WALK_FORWARD_TH_SCALE   = 0.05
WALK_FORWARD_CLAMP      = 0.5

#WALK_BIAS_Y             = 10.0
#WALK_BIAS_TH            = 3.0
WALK_BIAS_Y             = -1.2#1.0
WALK_BIAS_TH            = 1.5

CALIBRATE_MODE          = False
CALIBRATE_TIME          = 10.0

CX_KP                   = 0.01
CX_Y_CLAMP              = 1.0

# WALK_WITH_BAR:抬槓後走「跨過撿起槓鈴附近那條白線」的短段
#   這段用固定時間就好(看線不可靠,負重頭壓低,起點變動小)
WALK_WITH_BAR_X         = 2.8
WALK_WITH_BAR_Y         = -1.25#-1.0
WALK_WITH_BAR_TIME      = 8.0   
                    
WALK_WITH_BAR_Z          = 0.040
WALK_WITH_BAR_Z_OFFSET   = 0.0
WALK_WITH_BAR_HIP_PITCH  = 2.0
WALK_WITH_BAR_FEET_PITCH = 0.0

# STOP_AT_LIFT_LINE / WALK_2_FINISH 的巡航速度
LINE_CRUISE_X   = 4.0
LINE_CRUISE_Y   = 0.0
LINE_CRUISE_Z   = 0.026

# 安全 timeout(看不到線時最多走這秒數就強制停)
LIFT_LINE_TIMEOUT  = 15.0     # 50cm 距離,設寬鬆一點
FINAL_LINE_TIMEOUT = 18

# ── Adjust_Mode ────────────────────────────────────────────
ADJUST_FRAMES       = 5

CENTER_TOL_ADJUST   = 35
ADJUST_SHIFT_Y      = 0.5
ADJUST_SHIFT_X      = -0.2
ADJUST_SHIFT_STEPS  = 3
ADJUST_MAX_SHIFTS   = 8


# ── Adjust_Mode ─────────────────────────────
HEAD_TILT_FINE      = -0.9      #  細修時頭再低一點（比 -0.75 更低）

FINE_CY_TARGET      = 450       # cY 目標值：讓 Adjust_Mode 走到原本 CHECK_TOUCH 的距離
FINE_CX_TOL         = 15        # cX 容許誤差（< 15 就算對中，即 305±15）
FINE_SAMPLE_FRAMES  = 7         # 每次取樣幀數（穩定用）

FINE_X_STEP         = 0.04       # 前進速度（比 WALK_FORWARD_X 更小）
FINE_Y_KP           = 0.006     # cX 反饋強度
FINE_Y_CLAMP        = 0.18   # 側移上限

FINE_TH_KP          = 0.10      # angle 反饋強度（很小！避免修過頭）
FINE_TH_CLAMP       = 3.0       # 旋轉上限（很小！）
FINE_TH_SIGN        = -1.0
FINE_TH_BIAS        = 1.0       # Adjust_Mode 前進時的防偏轉補償，先用 0.8

FINE_TURN_X      = -1.0
FINE_TURN_Y      = 0.8
FINE_TURN_TH     = 3.5
FINE_TURN_TIME   = 3.0
FINE_ACCEPT_CY   = 445
FINE_MIN_PICK_CY = 442
FINE_ACCEPT_ANG  = 4.5

FINE_STEP_TIME      = 0.6       # 每輪走的時間

FINE_LOST_MAX   = 2      # 連續看不到 N 次就強制結束
FINE_MAX_ITER   = 24     # 總 iteration 上限：直接用 Adjust_Mode 走到可夾距離

# ── angle 穩定判斷參數 ─────────────────────────────
ANGLE_THRESHOLD         = 2.0     # angle 小於這個角度就當作平行
ANGLE_VALID_MAX         = 35.0    # 超過這角度通常是 YOLO OBB 跳角，不拿來修
ANGLE_MIN_VALID_FRAMES  = 4       # 7 幀裡至少 4 幀有效才相信 angle
ANGLE_SIGN_RATIO        = 0.6     # 多數角度方向一致才算穩定

# ── CHECK_TOUCH ─────────────────────────────────────────────
CHECK_DETECT_FRAMES = 3
CHECK_CENTER_TOL    = 45

CHECK_STEP_X        = 0.4
CHECK_STEP_TIME     = 0.3

CHECK_SHIFT_LEFT_Y  = -1.5
CHECK_SHIFT_RIGHT_Y =  2.0
CHECK_SHIFT_TIME    = 0.3

HEAD_TILT_CHECK_LOW = -0.9      # CHECK 時再低,更接近腳邊看槓鈴
CHECK_SAMPLE_FRAMES = 7         # 取樣幀數
CHECK_ANGLE_TOL     = 5       # 角度容許 ±3 度
CHECK_TURN_TH       = 0.8       # 每次原地轉的 th 大小
CHECK_TURN_X_COMP   = -0.3      # 抵消 walkStart 慣性前進的負 x
CHECK_TURN_TIME     = 1.0       # 每次轉動時間
CHECK_MAX_TURNS     = 6         # 最多轉幾次就放行
CHECK_LOST_MAX      = 3         # 連續看不到

CHECK_ANGLE_FIX_X     = -1.0
CHECK_ANGLE_FIX_Y     = -1.1
CHECK_ANGLE_FIX_TH    = 3.5
CHECK_ANGLE_FIX_TIME  = 0.45

CHECK_CENTER_FIX_KP    = 0.002
CHECK_CENTER_FIX_CLAMP = 0.10

CHECK_CLOSE_CY         = 420
CHECK_CLOSE_ANGLE_TOL  = 8
CHECK_CLOSE_OFFSET_TOL = 140

# ── 動作 ID ────────────────────────────────────────────────
MOTION_INIT         = 1
MOTION_PICK         = 195 #30 cd
#MOTION_PICK         = 196 #60 cd
MOTION_LIFT         = 186 #30 cd
MOTION_FINISH       = 185 #30 cd



# ============================================================
#  偵測 socket 客戶端
# ============================================================
class DetClient(object):
    def __init__(self, host=DET_BIND_HOST, port=DET_BIND_PORT, stale=DET_STALE_SEC):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((host, port))
        self.s.settimeout(0.2)
        self.stale  = stale
        self.latest = None
        t = threading.Thread(target=self._loop)
        t.daemon = True
        t.start()

    def _loop(self):
        while True:
            try:
                data, _ = self.s.recvfrom(8192)
                self.latest = json.loads(data.decode())
            except socket.timeout:
                continue
            except Exception:
                continue

    def _fresh(self):
        m = self.latest
        if m is None:
            return None
        if time.time() - m["t"] > self.stale:
            return None
        return m

    def get_bar(self):
        m = self._fresh()
        if not m or not m["bar"].get("found"):
            return None, None, None
        b = m["bar"]
        return int(b["cx"]), int(b["cy"]), float(b["angle"])

    def get_lines(self):
        """回傳所有白線 list(detector 已按 cY 由大到小排,近→遠)。"""
        m = self._fresh()
        if not m:
            return []
        return m.get("lines", [])

    def get_nearest_line(self):
        """最近的白線(cY 最大)。回傳 (cX, cY, angle) 或 (None, None, None)。"""
        lines = self.get_lines()
        if not lines:
            return None, None, None
        l = lines[0]
        return int(l["cx"]), int(l["cy"]), float(l["angle"])

    def get_fw(self):
        m = self.latest
        return int(m["fw"]) if m else 640

# ============================================================
#  IMU
# ============================================================
z_gyro                    = 0.0
z_gyro_offset             = 0.0
z_gyro_offset_for_calc    = 0.0

def imu_callback(data):
    global z_gyro, z_gyro_offset
    av = data.angular_velocity
    z_gyro = z_gyro - av.z / 2.0 + z_gyro_offset_for_calc
    z_gyro_offset = av.z / 2.0

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_callback, queue_size=1)

# ============================================================
#  狀態機
# ============================================================
class States:
    INIT              = -1
    READY             =  0
    WALK_FORWARD      =  1
    PICK_BAR          =  5
    WALK_WITH_BAR     =  6      # 抬槓走短段跨過第 0 條線
    STOP_AT_LIFT_LINE =  7      # ★ 新:看到線就停,準備舉過頭
    LIFT_BAR          =  8      # 舉過頭動作
    WALK_2_FINISH     =  9      # 看到終點線就停放下
    Adjust_Mode       = 11
    CHECK_TOUCH       = 12
    END               = 99

LEG_JOINTS = [
    "r_hip_yaw",   "l_hip_yaw",
    "r_hip_roll",  "l_hip_roll",
    "r_hip_pitch", "l_hip_pitch",
    "r_knee",      "l_knee",
    "r_ank_pitch", "l_ank_pitch",
    "r_ank_roll",  "l_ank_roll"
]

# ============================================================
#  初始化
# ============================================================
rospy.init_node("fira_weightlifting_v4")
robot = Robot()
rospy.sleep(0.2)

z_gyro_offset_for_calc = z_gyro_offset
z_gyro = 0.0

det = DetClient()

# ============================================================
#  共用函式
# ============================================================
def sample_bar(n=5, dt=0.05):
    xs, ys, angs = [], [], []
    for _ in range(n):
        x, y, a = det.get_bar()
        if x is not None:
            xs.append(x); ys.append(y); angs.append(a)
        rospy.sleep(dt)
    if not xs:
        return None, None, None
    return int(np.median(xs)), int(np.median(ys)), float(np.median(angs))


def normalize_bar_angle(a):
    """
    YOLO OBB angle 有時會跳到接近 90 度。
    槓鈴理想上接近水平，所以先折回 -45~45。
    """
    if a is None:
        return None

    while a > 45.0:
        a -= 90.0
    while a < -45.0:
        a += 90.0

    return a


def sample_bar_stable(n=9, dt=0.05):
    """
    多幀取樣，回傳比較穩定的 cX, cY, angle。
    angle_stable=False 代表這次角度不可信，不要拿來旋轉。
    """
    xs, ys, angs = [], [], []

    for _ in range(n):
        x, y, a = det.get_bar()
        if x is not None:
            xs.append(x)
            ys.append(y)

            a = normalize_bar_angle(a)
            if a is not None and abs(a) <= ANGLE_VALID_MAX:
                angs.append(a)

        rospy.sleep(dt)

    if not xs:
        return None, None, None, False

    cx_med = int(np.median(xs))
    cy_med = int(np.median(ys))

    if len(angs) < ANGLE_MIN_VALID_FRAMES:
        return cx_med, cy_med, None, False

    angle_med = float(np.median(angs))

    if abs(angle_med) <= ANGLE_THRESHOLD:
        return cx_med, cy_med, angle_med, True

    same_sign = 0
    for a in angs:
        if angle_med > 0 and a > 0:
            same_sign += 1
        elif angle_med < 0 and a < 0:
            same_sign += 1

    ratio = float(same_sign) / float(len(angs))
    angle_stable = ratio >= ANGLE_SIGN_RATIO

    return cx_med, cy_med, angle_med, angle_stable



def check_center(cX, frame_width, tol=CHECK_CENTER_TOL):
    mid = frame_width // 2
    if mid - tol <= cX <= mid + tol:
        return True, 2
    elif cX > mid + tol:
        return False, 0
    else:
        return False, 1


def set_head(tilt_rad, pan_rad=0.0, wait=0.25):
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_tilt", "head_pan"], [tilt_rad, pan_rad])
    rospy.sleep(wait)


def prepare_leg_walking():
    robot.walkStop()
    rospy.sleep(0.2)
    robot.setJointsControlModule(LEG_JOINTS, ["walking_module"])
    rospy.sleep(0.2)
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
    rospy.sleep(0.3)
    robot.setGeneralControlModule("action_module")
    rospy.sleep(0.3)
    robot.playMotion(MOTION_FINISH, wait_for_end=True)
    rospy.sleep(0.3)
    rospy.signal_shutdown("Finished")


def walk_shift(x, y, th, duration, z_amp=0.035):
    robot.walkStart()
    t0 = time.time()
    while time.time() - t0 < duration and not rospy.is_shutdown():
        robot.walkVelocities(x=x, y=y, th=th,
                             z_move_amplitude=z_amp,
                             balance=True, z_offset=0)
        rate.sleep()
    robot.walkStop()
    rospy.sleep(0.3)


def pulse_move(x=0, y=0, th=0, steps=2, z_amp=0.03):
    robot.walkStart()
    for _ in range(steps):
        if rospy.is_shutdown():
            break
        robot.walkVelocities(x=x, y=y, th=th,
                             z_move_amplitude=z_amp,
                             balance=True, z_offset=0)
        rate.sleep()
    robot.walkStop()
    rospy.sleep(0.3)


def cruise_until_line(stop_cy, timeout_sec, head_tilt, label, cruise_x=None, cruise_z=None,
                      z_offset=None, hip_pitch=None, feet_pitch=None,
                      ignore_line_sec=0.0, require_far_first=False, far_cy=360,
                      force_straight=False):
    """
    共用:邊走邊看,等最近的線 cY >= stop_cy 就停。
    用於 STOP_AT_LIFT_LINE 和 WALK_2_FINISH 的巡航段。
    回傳:('line' 看到線停 / 'timeout' 超時強制停)
    """
    set_head(head_tilt, wait=0.3)
    robot.walkStart()
    t0 = time.time()
    fw = det.get_fw()
    seen_far_line = not require_far_first

    if cruise_x is None:
        cruise_x = LINE_CRUISE_X
    if cruise_z is None:
        cruise_z = LINE_CRUISE_Z
    if z_offset is None:
        z_offset = WALK_WITH_BAR_Z_OFFSET
    if hip_pitch is None:
        hip_pitch = WALK_WITH_BAR_HIP_PITCH
    if feet_pitch is None:
        feet_pitch = WALK_WITH_BAR_FEET_PITCH

    while not rospy.is_shutdown():
        # 安全 timeout
        if time.time() - t0 > timeout_sec:
            print("[{}] timeout {:.1f}s, force stop".format(label, timeout_sec))
            robot.walkStop()
            return "timeout"

        lx, ly, la = det.get_nearest_line()
        th_line = 0.0 if force_straight else np.clip(z_gyro, -10, 10)

        if lx is None:
            # 看不到線 → 直走找線(配 IMU 修向)
            robot.walkVelocities(
                x=cruise_x, y=LINE_CRUISE_Y,
                th=th_line,
                z_move_amplitude=cruise_z,
                balance=True,
                z_offset=z_offset,
                hip_pitch=hip_pitch,
                feet_pitch=feet_pitch
            )
        else:
            elapsed = time.time() - t0

            # 終點線模式：
            # 一開始如果看到很近的線，通常是 lifting line，不要停。
            # 必須先看到一次比較遠的線 cY <= far_cy，才允許後面停線。
            if require_far_first and (not seen_far_line):
                if ly <= far_cy:
                    seen_far_line = True
                    print("[{}] armed final line detection, first far line cY={} <= {}".format(
                          label, ly, far_cy))
                else:
                    print("[{}] ignore near/lift line before armed cY={} > {}".format(
                          label, ly, far_cy))

                    robot.walkVelocities(
                        x=cruise_x,
                        y=LINE_CRUISE_Y,
                        th=th_line,
                        z_move_amplitude=cruise_z,
                        balance=True,
                        z_offset=z_offset,
                        hip_pitch=hip_pitch,
                        feet_pitch=feet_pitch
                    )
                    rate.sleep()
                    continue

            if ly >= stop_cy and elapsed < ignore_line_sec:
                print("[{}] ignore early line cY={} elapsed={:.1f}/{:.1f}".format(
                    label, ly, elapsed, ignore_line_sec))

                robot.walkVelocities(
                    x=cruise_x,
                    y=LINE_CRUISE_Y,
                    th=th_line,
                    z_move_amplitude=cruise_z,
                    balance=True,
                    z_offset=z_offset,
                    hip_pitch=hip_pitch,
                    feet_pitch=feet_pitch
                )
                rate.sleep()
                continue

            if ly >= stop_cy:
                print("[{}] line reached cY={} >= {} -> stop".format(label, ly, stop_cy))
                robot.walkStop()
                return "line"
            
            # 邊走邊用 cX 對準線中心
            cx_off = lx - (fw // 2)
            y_corr = np.clip(-cx_off * LINE_CX_KP, -LINE_Y_CLAMP, LINE_Y_CLAMP)
            print("[{}] line cY={} cX_off={} y={:.2f}".format(label, ly, cx_off, y_corr))
            robot.walkVelocities(
                x=cruise_x, y=LINE_CRUISE_Y + y_corr,
                th=th_line,
                z_move_amplitude=cruise_z,
                balance=True, 
                z_offset=z_offset,
                hip_pitch=hip_pitch,
                feet_pitch=feet_pitch
            )
        rate.sleep()

    return "shutdown"


def second_check_then_forward_after_line(label, stop_cy, timeout_sec, second_head_tilt, forward_time, forward_x):
    """
    第一次 cruise_until_line 已經用目前頭角度找到線並停下。
    接著：
    1. 頭再低一點
    2. 穩定 2 秒
    3. 再次偵測停止線，cY >= stop_cy 後停下
    4. 再穩定 2 秒
    5. 不再看線，自由往前走 forward_time 秒
    6. 停下，交給 playMotion

    注意：
    STOP_AT_LIFT_LINE 還是用原本較強的 WALK_WITH_BAR 參數。
    WALK_2_FINISH 因為槓鈴已經舉過頭，重心高，所以改用比較保守的 FINISH 參數。
    """

    print("[{}] first line detected -> stop and lower head".format(label))
    robot.walkStop()
    rospy.sleep(0.2)


    print("[{}] lower head to {:.2f}".format(label, second_head_tilt))
    set_head(second_head_tilt, wait=0.4)

    print("[{}] stable before second line check {:.1f}s".format(label, LINE_STABLE_WAIT))
    rospy.sleep(LINE_STABLE_WAIT)

    # ============================================================
    # 依照不同階段選擇二次確認線時的走路參數
    # ============================================================
    if label == "WALK_2_FINISH":
        check_cruise_x = FINISH_LINE_CRUISE_X
        check_cruise_z = FINISH_WALK_Z
        check_z_offset = FINISH_Z_OFFSET
        check_hip_pitch = FINISH_HIP_PITCH
        check_feet_pitch = FINISH_FEET_PITCH
    else:
        check_cruise_x = None
        check_cruise_z = None
        check_z_offset = None
        check_hip_pitch = None
        check_feet_pitch = None

    print("[{}] second line check with lower head".format(label))
    result2 = cruise_until_line(
        stop_cy=stop_cy,
        timeout_sec=timeout_sec,
        head_tilt=second_head_tilt,
        label=label + "_LOW_CHECK",
        cruise_x=check_cruise_x,
        cruise_z=check_cruise_z,
        z_offset=check_z_offset,
        hip_pitch=check_hip_pitch,
        feet_pitch=check_feet_pitch,
        force_straight=(label == "STOP_AT_LIFT_LINE")
    )

    print("[{}] second check result = {}".format(label, result2))
    robot.walkStop()

    print("[{}] stable after second stop {:.1f}s".format(label, LINE_STABLE_WAIT))
    rospy.sleep(LINE_STABLE_WAIT)

    # ============================================================
    # 依照不同階段選擇最後補走參數
    # ============================================================
    if label == "WALK_2_FINISH":
        free_z = FINISH_WALK_Z
        free_z_offset = FINISH_Z_OFFSET
        free_hip_pitch = FINISH_HIP_PITCH
        free_feet_pitch = FINISH_FEET_PITCH
        free_th_limit = 6
    else:
        free_z = WALK_WITH_BAR_Z
        free_z_offset = WALK_WITH_BAR_Z_OFFSET
        free_hip_pitch = WALK_WITH_BAR_HIP_PITCH
        free_feet_pitch = WALK_WITH_BAR_FEET_PITCH
        free_th_limit = 10

    print("[{}] free forward after second stop for {:.1f}s".format(label, forward_time))
    robot.walkStart()

    t0 = time.time()
    while time.time() - t0 < forward_time and not rospy.is_shutdown():
        elapsed = time.time() - t0

        if label == "WALK_2_FINISH":
            if elapsed < 0.8:
                x_now = forward_x * 0.5
            else:
                x_now = forward_x
        else:
            x_now = forward_x

        robot.walkVelocities(
            x=x_now,
            y=AFTER_LINE_Y,
            th=0.0 if label == "STOP_AT_LIFT_LINE" else np.clip(z_gyro, -free_th_limit, free_th_limit),
            z_move_amplitude=free_z,
            balance=True,
            z_offset=free_z_offset,
            hip_pitch=free_hip_pitch,
            feet_pitch=free_feet_pitch
        )
        rate.sleep()

    robot.walkStop()
    print("[{}] free forward finished -> stop".format(label))
    rospy.sleep(0.5)


def forward_after_first_line_only(label, forward_time, forward_x):
    """
    終點線用：
    第一次偵測到線後，不再二次低頭偵測，
    直接自由往前走 forward_time 秒，再停下 playMotion。
    """
    print("[{}] first line detected -> free forward {:.1f}s".format(label, forward_time))

    robot.walkStop()
    rospy.sleep(0.3)

    robot.walkStart()
    t0 = time.time()

    while time.time() - t0 < forward_time and not rospy.is_shutdown():
        robot.walkVelocities(
            x=forward_x,
            y=AFTER_LINE_Y,
            th=np.clip(z_gyro, -6, 6),
            z_move_amplitude=FINISH_WALK_Z,
            balance=True,
            z_offset=FINISH_Z_OFFSET,
            hip_pitch=FINISH_HIP_PITCH,
            feet_pitch=FINISH_FEET_PITCH
        )
        rate.sleep()

    robot.walkStop()
    print("[{}] free forward finished -> playMotion".format(label))
    rospy.sleep(0.3)


# ============================================================
#  主迴圈
# ============================================================
tickrate    = 60
rate        = rospy.Rate(tickrate)
currState   = States.INIT
go_back     = 0
position_correct = False
cX = cY = None

adjust_phase     = "shift"
adjust_try_count = 0
verify_retry     = 0
adjust_lost_count = 0
adjust_iter_count = 0
walk_lost_count = 0
walk_forward_start_time = None

while not rospy.is_shutdown():

    cX, cY, _ = det.get_bar()
    fw = det.get_fw()
    if DEBUG_MODE and cX is not None:
        print("[DEBUG] cX={} cY={} go_back={}".format(cX, cY, go_back))

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
            rospy.sleep(0.5)
            reset_imu()
            set_head(HEAD_TILT_WALK)

            if CALIBRATE_MODE:
                print("=" * 50)
                print("[CALIBRATE] {:.0f}s straight walk".format(CALIBRATE_TIME))
                print("[CALIBRATE] BIAS_Y={} BIAS_TH={}".format(WALK_BIAS_Y, WALK_BIAS_TH))
                print("=" * 50)
                robot.walkStart()
                t0 = time.time()
                last_print = 0.0
                while time.time() - t0 < CALIBRATE_TIME and not rospy.is_shutdown():
                    robot.walkVelocities(
                        x=WALK_FORWARD_X, y=WALK_BIAS_Y, th=WALK_BIAS_TH,
                        z_move_amplitude=0.03, balance=True, z_offset=0)
                    elapsed = time.time() - t0
                    if elapsed - last_print >= 0.5:
                        print("[CALIBRATE] t={:.1f}s imu={:.3f}".format(elapsed, z_gyro))
                        last_print = elapsed
                    rate.sleep()
                robot.walkStop()
                print("=" * 50)
                print("[CALIBRATE] FINISHED")
                print("[CALIBRATE] Final imu = {:.3f}".format(z_gyro))
                print("[CALIBRATE] BIAS_Y={} BIAS_TH={}".format(WALK_BIAS_Y, WALK_BIAS_TH))
                if abs(z_gyro) < 0.3:
                    print("[CALIBRATE] GOOD! Very straight")
                elif z_gyro > 0:
                    print("[CALIBRATE] drift RIGHT (positive) -> DECREASE WALK_BIAS_TH")
                else:
                    print("[CALIBRATE] drift LEFT (negative) -> INCREASE WALK_BIAS_TH")
                print("=" * 50)
                currState = States.READY


            else:
                robot.walkStart()
                walk_forward_start_time = time.time()
                currState = States.WALK_FORWARD

    # ============================================================
    #  WALK_FORWARD(找槓鈴)
    # ============================================================
    elif currState == States.WALK_FORWARD:

        if go_back == 0:
            h_tilt = HEAD_TILT_WALK
        
        else:
            h_tilt = HEAD_TILT_CLOSE

        set_head(h_tilt, wait=0.0)

        th_corr = np.clip(z_gyro * WALK_FORWARD_TH_SCALE,
                          -WALK_FORWARD_CLAMP, WALK_FORWARD_CLAMP)

        if cX is not None:
            walk_lost_count = 0
            cx_offset = cX - (fw // 2)
            y_corr = np.clip(-cx_offset * CX_KP, -CX_Y_CLAMP, CX_Y_CLAMP)
        else:
            cx_offset = 0
            y_corr = 0.0
            walk_lost_count += 1
            print("[WALK_FORWARD] temporary lost ({}/{}), wait detector".format(
                  walk_lost_count, WALK_LOST_MAX))

            if walk_lost_count < WALK_LOST_MAX:
                robot.walkVelocities(
                    x=0.0,
                    y=0.0,
                    th=0.0,
                    z_move_amplitude=0.03,
                    balance=True,
                    z_offset=0
                )
                rate.sleep()
                continue
            
            print("[WALK_FORWARD] target lost -> stop and head up search")
            walk_lost_count = 0
            robot.walkStop()
            set_head(HEAD_TILT_SEARCH, wait=0.4)
            rate.sleep()
            continue

        if walk_forward_start_time is None:
            walk_forward_start_time = time.time()

        start_elapsed = time.time() - walk_forward_start_time

        # ============================================================
        # WALK_FORWARD soft start
        # 0.0 ~ 0.6s：只讓 walking module 啟動，不給偏移
        # 0.6 ~ 2.0s：只慢慢加 x
        # 2.0 ~ 3.5s：再慢慢加入 y / th bias
        # ============================================================

        if start_elapsed < 0.8:
            x_cmd = 0.10
            y_cmd = 0.0
            th_cmd = 0.0
            z_cmd = 0.016

        elif start_elapsed < 2.0:
            r = (start_elapsed - 0.6) / 1.4
            r = max(0.0, min(1.0, r))
            smooth = r * r * (3.0 - 2.0 * r)

            x_cmd = WALK_FORWARD_X * (0.25 + 0.75 * smooth)
            y_cmd = 0.0
            th_cmd = 0.0
            z_cmd = 0.022 + 0.008 * smooth

        elif start_elapsed < 3.5:
            r = (start_elapsed - 2.0) / 1.5
            r = max(0.0, min(1.0, r))
            smooth = r * r * (3.0 - 2.0 * r)

            x_cmd = WALK_FORWARD_X
            y_cmd = (y_corr + WALK_BIAS_Y) * smooth
            th_cmd = (th_corr + WALK_BIAS_TH) * smooth
            z_cmd = 0.030

        else:
            x_cmd = WALK_FORWARD_X
            y_cmd = y_corr + WALK_BIAS_Y
            th_cmd = th_corr + WALK_BIAS_TH
            z_cmd = 0.030

        robot.walkVelocities(
            x=x_cmd,
            y=y_cmd,
            th=th_cmd,
            z_move_amplitude=z_cmd,
            balance=True,
            z_offset=0
        )

        if go_back == 0 and cY >= CY_STOP_FIRST:
            robot.walkStop()
            rospy.sleep(0.3)
            position_correct = False
            adjust_lost_count = 0
            adjust_iter_count = 0
            currState = States.Adjust_Mode

        elif go_back == 1 and cY >= CY_STOP_SECOND:
            robot.walkStop()
            rospy.sleep(0.3)
            position_correct = False
            adjust_lost_count = 0
            adjust_iter_count = 0
            currState = States.Adjust_Mode

        elif go_back == 2 and cY >= CY_STOP_FINAL:
            robot.walkStop(); rospy.sleep(0.3)
            position_correct = False
            currState = States.CHECK_TOUCH


    # ============================================================
    #  Adjust_Mode
    # ============================================================
    elif currState == States.Adjust_Mode:
        print("[Adjust_Mode] fine-tune to bar")
        adjust_iter_count += 1

        # 頭部低一點看近處
        set_head(HEAD_TILT_FINE, wait=0.8)

        # ★ 總 iteration 上限
        if adjust_iter_count >= FINE_MAX_ITER:
            print("  max iter {}, force -> PICK_BAR".format(FINE_MAX_ITER))
            adjust_lost_count = 0
            adjust_iter_count = 0
            go_back = 1
            currState = States.PICK_BAR
            rospy.sleep(0.3)
            continue

        # 取樣
        cx_now, cy_now, ang_now, angle_stable = sample_bar_stable(FINE_SAMPLE_FRAMES)

        if cx_now is None:
            adjust_lost_count += 1
            print("  no detection ({}/{}), head up to search".format(
                  adjust_lost_count, FINE_LOST_MAX))
    
            # ★ 連續看不到太多次 → 直接進 CHECK_TOUCH 賭一把
            if adjust_lost_count >= FINE_LOST_MAX:
                print("  lost too many, force -> PICK_BAR")
                adjust_lost_count = 0
                adjust_iter_count = 0
                go_back = 1
                currState = States.PICK_BAR
                rospy.sleep(0.3)
                continue
    
            set_head(HEAD_TILT_ADJUST_0, wait=0.4)
            cx_retry, cy_retry, _, _ = sample_bar_stable(FINE_SAMPLE_FRAMES)
            if cx_retry is not None:
                print("  recovered! cX={} cY={} -> PICK_BAR".format(cx_retry, cy_retry))
                adjust_lost_count = 0
                adjust_iter_count = 0
                go_back = 1
                currState = States.PICK_BAR
                rospy.sleep(0.3)
                continue
            else:
                print("  still lost, small step back")
                set_head(HEAD_TILT_FINE, wait=0.3)
                pulse_move(x=-0.3, y=0, th=0, steps=2)
                continue
        # 有偵測 → 重置 counter
        adjust_lost_count = 0

        frame_cx = fw // 2
        offset = cx_now - frame_cx
        angle = ang_now if ang_now is not None else 0.0

        cy_ok = cy_now >= FINE_CY_TARGET
        cx_ok = abs(offset) <= FINE_CX_TOL
        angle_ok = (not angle_stable) or (abs(angle) <= ANGLE_THRESHOLD)
        cy_close_enough = cy_now >= (FINE_CY_TARGET - 20)   # cY >= 390 就算很近

        print("  cX={} cY={} angle={:.1f} stable={} | cX_ok={} cY_ok={} ang_ok={}".format(
              cx_now, cy_now, angle, angle_stable, cx_ok, cy_ok, angle_ok))
        
        # ★ angle 太歪時，不先側移，避免越修越斜
        angle_need_fix = angle_stable and (abs(angle) > 12.0) and (abs(offset) <= 45)

        # ★ CENTER_FIRST：cX 偏離中心太多時，先側移回中心，不前進、不旋轉
        if False and abs(offset) > 75 and not angle_need_fix:
            x_cmd = 0.0
            y_cmd = np.clip(offset * FINE_Y_KP,
                            -FINE_Y_CLAMP,
                            FINE_Y_CLAMP)
            th_cmd = 0.0
            step_time = 0.25

            print("  [CENTER_FIRST] offset={} y={:.2f} step_time={:.2f}".format(
                  offset, y_cmd, step_time))

            robot.walkStart()
            t0 = time.time()
            while time.time() - t0 < step_time and not rospy.is_shutdown():
                robot.walkVelocities(
                    x=x_cmd,
                    y=y_cmd,
                    th=th_cmd,
                    z_move_amplitude=0.03,
                    balance=True,
                    z_offset=0
                )
                rate.sleep()

            robot.walkStop()
            rospy.sleep(1.0)
            continue

        # 主要條件：三個都達標
        main_ok = cx_ok and cy_ok and angle_stable and abs(angle) <= FINE_ACCEPT_ANG


        # ★ 目前實測：cY 約 400 以上且 angle 已經平行，就進 CHECK_TOUCH
        if not main_ok and cy_now >= FINE_ACCEPT_CY and angle_stable and abs(angle) <= FINE_ACCEPT_ANG:
            print("  angle aligned and close enough -> PICK_BAR")
            main_ok = True

        cy_close_enough = cy_now >= FINE_MIN_PICK_CY
        # 備用條件：cY 已經接近原本 CHECK_TOUCH 距離，且 cX 偏移不太大就接受
        if not main_ok and cy_close_enough and abs(offset) <= 20 and angle_stable and abs(angle) <= FINE_ACCEPT_ANG :
            print("  cY close enough, accept with cX offset={}".format(offset))
            main_ok = True

        if not main_ok and cy_now >= (FINE_CY_TARGET + 10) and angle_ok and abs(offset) <= 45:
            print("  cY too close ({}), force accept to avoid overshoot".format(cy_now))
            main_ok = True

        if main_ok:
            print("  ALL OK -> PICK_BAR (skip CHECK_TOUCH)")
            go_back = 1
            currState = States.PICK_BAR
            rospy.sleep(0.2)
            continue

        # 沒達標 → 計算三軸修正量
        # x：越接近目標越慢
        # 沒達標 → 計算修正量
        # 角度優先：只要 angle 還不對，就先只轉，不前進、不側移


        # =====================================================
        #  TH TEST MODE：只測旋轉方向
        #  測完記得還原
        # =====================================================
        if angle_stable and abs(angle) > FINE_ACCEPT_ANG:

            if angle < 0:
                x_cmd = FINE_TURN_X
                y_cmd = FINE_TURN_Y
                th_cmd = FINE_TURN_TH
            else:
                x_cmd = FINE_TURN_X
                y_cmd = -FINE_TURN_Y
                th_cmd = -FINE_TURN_TH

            step_time = FINE_TURN_TIME
            z_amp = 0.02

            print("  [ANGLE_PULSE] x={:.2f} y={:.2f} th={:.2f} angle={:.1f} step_time={:.2f}".format(
                   x_cmd, y_cmd, th_cmd, angle, step_time))

        else:
            # 角度已經差不多，不再修角度，只微微靠近
            if not cy_ok:
                x_cmd = 0.015
            else:
                x_cmd = 0.0

            y_cmd = 0.0
            th_cmd = 0.0
            step_time = 0.20
            z_amp = 0.02

            print("  [FINE_FORWARD] x={:.2f} y={:.2f} th={:.2f} angle={:.1f} step_time={:.2f}".format(
                  x_cmd, y_cmd, th_cmd, angle, step_time))

        print("  [TH_TEST] x={:.2f} y={:.2f} th={:.2f} step_time={:.2f}".format(
              x_cmd, y_cmd, th_cmd, step_time))

############################################
       # angle_only = angle_need_fix

        #if angle_only:
           # x_cmd = 0.0
           # y_cmd = 0.0
           # th_cmd = np.clip(FINE_TH_SIGN * angle * FINE_TH_KP,
                             #-FINE_TH_CLAMP,
                             #FINE_TH_CLAMP)
           # step_time = 0.35

            #print("  [ANGLE_ONLY] angle={:.1f} th={:.2f} sign={} step_time={:.2f}".format(
            #      angle, th_cmd, FINE_TH_SIGN, step_time))

       # else:
            # x：越接近目標越慢
           # if abs(offset) > 60:
               # x_cmd = 0.04 if not cy_ok else 0.0
               # step_time = 0.25
           # else:
                # x：越接近目標越慢
                #if not cy_ok:
                    #cy_gap = FINE_CY_TARGET - cy_now

                    #if not cy_ok:
                        #x_cmd = FINE_X_STEP
                       # step_time = 0.20
                   # else:
                       # x_cmd = 0.0
                        #step_time = 0.20

               # else:
                   # x_cmd = 0.0
                    #step_time = FINE_STEP_TIME * 0.3
   
            
           # y_cmd = 0.0

            # angle_ok 才會進到這裡，所以 th 不修
            #if angle_stable:
                #th_cmd = np.clip(-angle * FINE_TH_KP,
                    # -FINE_TH_CLAMP,
                    # FINE_TH_CLAMP)
            #else:
               # th_cmd = 0.0

            #print("  [FINE_MOVE] x={:.2f} y={:.2f} th={:.2f} angle={:.1f} step_time={:.2f}".format(
                  #x_cmd, y_cmd, th_cmd, angle, step_time))
####################################################################
        # 執行
        robot.walkStart()
        t0 = time.time()
        while time.time() - t0 < step_time and not rospy.is_shutdown():
            robot.walkVelocities(
                x=x_cmd,
                y=y_cmd ,
                th=th_cmd ,
                z_move_amplitude=0.018,
                balance=True, z_offset=0.0
            )
            rate.sleep()
        robot.walkStop()
        rospy.sleep(0.3)

        

    # ============================================================
    #  CHECK_TOUCH
    # ============================================================
    elif currState == States.CHECK_TOUCH:
        print("[CHECK_TOUCH] head down, align angle to +-{}".format(CHECK_ANGLE_TOL))
        robot.walkStop(); rospy.sleep(0.3)
    
        set_head(HEAD_TILT_CHECK_LOW, wait=0.5)

        check_turn_count = 0
        check_lost_count = 0

        while not rospy.is_shutdown():
            # 用 sample_bar 抓多幀取中位數
            cx_now, cy_now, ang_now, angle_stable = sample_bar_stable(CHECK_SAMPLE_FRAMES)

            if cx_now is None:
                check_lost_count += 1
                print("  [CHECK] no detection ({}/{})".format(
                      check_lost_count, CHECK_LOST_MAX))
                if check_lost_count >= CHECK_LOST_MAX:
                    print("  [CHECK] lost too many, force PICK")
                    break
                rospy.sleep(0.3)
                continue

            check_lost_count = 0
            angle = ang_now if ang_now is not None else 0.0
            offset = cx_now - (fw // 2)

            angle_ok = (not angle_stable) or (abs(angle) <= CHECK_ANGLE_TOL)       # ★ 不再放 angle_stable 邏輯
            center_ok = abs(offset) <= CHECK_CENTER_TOL

            if cy_now >= CHECK_CLOSE_CY and abs(angle) <= CHECK_CLOSE_ANGLE_TOL and abs(offset) <= CHECK_CLOSE_OFFSET_TOL:
                print("  [CHECK] close enough, accept cY={} offset={} angle={:.1f} -> PICK_BAR".format(
                    cy_now, offset, angle))
                break

            print("  [CHECK] cX={} cY={} angle={:.2f} | ang_ok={} center_ok={} turns={}/{}".format(
                  cx_now, cy_now, angle, angle_ok, center_ok, check_turn_count, CHECK_MAX_TURNS))

            if angle_ok and center_ok and cy_now >= CY_PICK_READY:
                print("  [CHECK] aligned and close enough! cY={} angle={:.2f} offset={} -> PICK_BAR".format(
                    cy_now, angle, offset))
                break

            if angle_ok and center_ok and cy_now < CY_PICK_READY:
                print("  [CHECK] aligned but not close enough, small forward cY={} < {}".format(
                    cy_now, CY_PICK_READY))

                x_cmd = 0.05
                y_cmd = 0.0
                th_cmd = 0.5
                step_time = 0.45

                robot.walkStart()
                t0 = time.time()
                while time.time() - t0 < step_time and not rospy.is_shutdown():
                    robot.walkVelocities(
                        x=x_cmd,
                        y=y_cmd,
                        th=th_cmd,
                        z_move_amplitude=0.018,
                        balance=True,
                        z_offset=0
                    )
                    rate.sleep()

                robot.walkStop()
                rospy.sleep(0.5)
                check_turn_count += 1
                continue

            if check_turn_count >= CHECK_MAX_TURNS:
                print("  [CHECK] max turns reached, force PICK")
                break

            # =====================================================
            # CHECK_TOUCH 修正策略：
            # 1. 已經夠近就直接 PICK，不再硬修到完美
            # 2. angle 太大時，優先用 Adjust_Mode 類似方式修角度
            # 3. angle 可以時，才修 cX
            # =====================================================

            if cy_now >= CHECK_CLOSE_CY and abs(angle) <= CHECK_CLOSE_ANGLE_TOL and abs(offset) <= CHECK_CLOSE_OFFSET_TOL:
                print("  [CHECK] close enough, accept cY={} offset={} angle={:.1f} -> PICK_BAR".format(
                    cy_now, offset, angle))
                break

            if not angle_ok and abs(offset) <= CHECK_CLOSE_OFFSET_TOL:
                print("  [CHECK] angle fix by Adjust_Mode style")

                if angle < 0:
                    x_cmd = CHECK_ANGLE_FIX_X
                    y_cmd = CHECK_ANGLE_FIX_Y
                    th_cmd = CHECK_ANGLE_FIX_TH
                else:
                    x_cmd = CHECK_ANGLE_FIX_X
                    y_cmd = -CHECK_ANGLE_FIX_Y
                    th_cmd = -CHECK_ANGLE_FIX_TH

                step_time = CHECK_ANGLE_FIX_TIME
                print("  [CHECK_ANGLE_FIX] x={:.2f} y={:.2f} th={:.2f} angle={:.1f} time={:.2f}".format(
                    x_cmd, y_cmd, th_cmd, angle, step_time))

            elif abs(offset) > 60:
                # 從你的 log 看，原本 y=+0.15 會讓 cX 往更左跑
                # 所以這裡改成 offset * KP，不要用 -offset
                x_cmd = 0.0
                y_cmd = np.clip(offset * CHECK_CENTER_FIX_KP,
                                -CHECK_CENTER_FIX_CLAMP,
                                CHECK_CENTER_FIX_CLAMP)
                th_cmd = 0.0
                step_time = 0.35

                print("  [CHECK_CENTER_FIX] y={:.2f} offset={} time={:.2f}".format(
                    y_cmd, offset, step_time))

            else:
                # 角度和中心都差不多，慢慢靠近
                x_cmd = 0.02
                y_cmd = 0.0
                th_cmd = 0.5
                step_time = 0.45

                print("  [CHECK_FORWARD] x={:.2f} angle={:.1f} offset={} time={:.2f}".format(
                    x_cmd, angle, offset, step_time))



            # 執行
            robot.walkStart()
            t0 = time.time()
            while time.time() - t0 < step_time and not rospy.is_shutdown():
                robot.walkVelocities(
                    x=x_cmd, y=y_cmd, th=th_cmd,
                    z_move_amplitude=0.018,
                    balance=True, z_offset=0
                )
                rate.sleep()
            robot.walkStop()
            rospy.sleep(0.5)

            check_turn_count += 1

        rospy.sleep(0.3)
        currState = States.PICK_BAR
        rospy.sleep(0.3)

    # ============================================================
    #  PICK_BAR
    # ============================================================
    elif currState == States.PICK_BAR:
        print("[PICK_BAR]")
        robot.walkStop(); rospy.sleep(0.5)

        robot.setGeneralControlModule("none"); rospy.sleep(0.3)
        robot.setGeneralControlModule("action_module"); rospy.sleep(0.3)
        robot.playMotion(MOTION_PICK, wait_for_end=True); rospy.sleep(0.3)

        reset_imu(); rospy.sleep(0.5)

        prepare_leg_walking()
        robot.walkStart()

        t0 = time.time()
        while time.time() - t0 < 2.0 and not rospy.is_shutdown():
            robot.walkVelocities(
                x=5, y=0,
                th=np.clip(z_gyro, -2.0, 2.0),
                z_move_amplitude=0.035,
                balance=True, z_offset=0, hip_pitch=2
            )
            rate.sleep()

        robot.walkStop(); rospy.sleep(0.3)
        currState = States.WALK_WITH_BAR

    # ============================================================
    #  WALK_WITH_BAR(抬槓走短段、跨過撿起槓鈴附近的線)
    # ============================================================
    elif currState == States.WALK_WITH_BAR:
        print("[WALK_WITH_BAR] cross over the bar-side line for {:.1f}s".format(
              WALK_WITH_BAR_TIME))
        rospy.sleep(0.3)

        prepare_leg_walking()
        set_head(HEAD_TILT_LIFTWALK, wait=1.0)
        robot.walkStart()

        t0 = time.time()
        while time.time() - t0 < WALK_WITH_BAR_TIME and not rospy.is_shutdown():
            elapsed = time.time() - t0

            # 前 1 秒慢慢啟動，最後 1 秒慢慢收，避免突然站起或煞停
            x_now = WALK_WITH_BAR_X
            if elapsed < 1.0:
                x_now = WALK_WITH_BAR_X * 0.6
            elif WALK_WITH_BAR_TIME - elapsed < 1.0:
                x_now = WALK_WITH_BAR_X * 0.6

            robot.walkVelocities(
                x=x_now,
                y=WALK_WITH_BAR_Y,
                th=np.clip(z_gyro, -2.0, 2.0),
                z_move_amplitude=WALK_WITH_BAR_Z,
                balance=True,
                z_offset=WALK_WITH_BAR_Z_OFFSET,
                hip_pitch=WALK_WITH_BAR_HIP_PITCH,
                feet_pitch=WALK_WITH_BAR_FEET_PITCH
            )
            rate.sleep()

        robot.walkStop(); rospy.sleep(0.3)
        currState = States.STOP_AT_LIFT_LINE

    # ============================================================
    #  STOP_AT_LIFT_LINE(走到「該舉過頭的線」停下)
    #  邊走邊看,看到最近的線 cY >= LIFT_LINE_STOP_CY 就停。
    # ============================================================
    elif currState == States.STOP_AT_LIFT_LINE:
        print("[STOP_AT_LIFT_LINE] cruise to lift line")
        result = cruise_until_line(
            stop_cy=LIFT_LINE_STOP_CY,
            timeout_sec=LIFT_LINE_TIMEOUT,
            head_tilt=HEAD_TILT_LINE_NEAR,
            label="STOP_AT_LIFT_LINE",
            force_straight=True
        )
        if result == "line":
            second_check_then_forward_after_line(
                label="STOP_AT_LIFT_LINE",
                stop_cy=LIFT_LINE_SECOND_STOP_CY,
                timeout_sec=4.0,
                second_head_tilt=HEAD_TILT_AFTER_LINE_LOW,
                forward_time=LIFT_AFTER_LINE_WALK_TIME,
                forward_x=LIFT_AFTER_LINE_X
            )
        else:
            print("[STOP_AT_LIFT_LINE] first line not detected, skip second check")
        
        rospy.sleep(0.3)
        currState = States.LIFT_BAR

    # ============================================================
    #  LIFT_BAR(舉過頭)
    # ============================================================
    elif currState == States.LIFT_BAR:
        print("[LIFT_BAR]")
        robot.walkStop(); rospy.sleep(0.1)

        robot.setGeneralControlModule("none"); rospy.sleep(0.1)
        robot.setGeneralControlModule("action_module"); rospy.sleep(0.1)
        robot.playMotion(MOTION_LIFT, wait_for_end=True); rospy.sleep(0.1)

        reset_imu(); rospy.sleep(0.2)

        prepare_leg_walking(); rospy.sleep(0.1)
        currState = States.WALK_2_FINISH

    # ============================================================
    #  WALK_2_FINISH(走到終點線停下放槓鈴)
    #  ● 暖身 + 啟動 + 加速段照舊,讓姿態穩
    #  ● 巡航段:看到最近的線 cY >= FINAL_LINE_STOP_CY 就停
    # ============================================================
    elif currState == States.WALK_2_FINISH:
        print("[WALK_2_FINISH] pre-walk then cruise to final line")

        prepare_leg_walking()

        set_head(HEAD_TILT_FINISH_PREWALK, wait=0.5)

        robot.walkStart()

        # 舉過頭後先讓身體進入前傾姿態，不要馬上往前衝
        t_pre = time.time()
        while time.time() - t_pre < 0.4 and not rospy.is_shutdown():
            robot.walkVelocities(
                x=0.0,
                y=0.0,
                th=0.0,
                z_move_amplitude=FINISH_WALK_Z,
                balance=True,
                z_offset=FINISH_Z_OFFSET,
                hip_pitch=FINISH_HIP_PITCH,
                feet_pitch=FINISH_FEET_PITCH
            )
            rate.sleep()

        t0 = time.time()
        while time.time() - t0 < FINISH_PRE_WALK_TIME and not rospy.is_shutdown():
            elapsed = time.time() - t0

            # 舉過頭後重心高，不能一開始就給滿速度，避免後倒
            if elapsed < 0.8:
                x_now = FINISH_PRE_WALK_X * 0.65
            elif elapsed < 1.6:
                x_now = FINISH_PRE_WALK_X * 0.85
            else:
                x_now = FINISH_PRE_WALK_X

            robot.walkVelocities(
                x=x_now,
                y=0.0,
                th=np.clip(z_gyro, -8, 8),
                z_move_amplitude=FINISH_WALK_Z,
                balance=True,
                z_offset=FINISH_Z_OFFSET,
                hip_pitch=FINISH_HIP_PITCH,
                feet_pitch=FINISH_FEET_PITCH
            )
            rate.sleep()

        # 這裡先不要 stop，避免舉槓後重新啟動步態造成後仰
        # robot.walkStop()
        # rospy.sleep(0.3)

        result = cruise_until_line(
            stop_cy=FINAL_LINE_STOP_CY,
            timeout_sec=FINAL_LINE_TIMEOUT,
            head_tilt=HEAD_TILT_FINISH_LINE,
            label="WALK_2_FINISH",
            cruise_x=FINISH_LINE_CRUISE_X,
            cruise_z=FINISH_WALK_Z,
            z_offset=FINISH_Z_OFFSET,
            hip_pitch=FINISH_HIP_PITCH,
            feet_pitch=FINISH_FEET_PITCH,
            ignore_line_sec=FINISH_IGNORE_LINE_TIME
        )

        if result == "line":
            forward_after_first_line_only(
                label="WALK_2_FINISH",
                forward_time=FINAL_AFTER_LINE_WALK_TIME,
                forward_x=FINAL_AFTER_LINE_X
            )
        else:
            print("[WALK_2_FINISH] first line not detected, force finish")

        finish_and_exit()

    # ============================================================
    #  END
    # ============================================================
    elif currState == States.END:
        finish_and_exit()

    rate.sleep()
