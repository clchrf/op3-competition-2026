#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
op3_weightlifting_v4.py  ── OP3 端控制節點(Python 2 / ROS1)

v4 改動(配合多階段停線):
  ● WALK_WITH_BAR 維持「固定時間走一小段、跨過撿起槓鈴附近那條白線」
    → 跨完後低頭、進 STOP_AT_LIFT_LINE
  ● STOP_AT_LIFT_LINE(新):邊走邊看,看到最近的線 cY >= LIFT_LINE_STOP_CY 就停
    → 進 LIFT_BAR(舉過頭動作)
  ● LIFT_BAR 完後維持低頭,進 WALK_2_FINISH
  ● WALK_2_FINISH:邊走邊看,看到最近的線 cY >= FINAL_LINE_STOP_CY 就停放下
  ● DetClient 改用 get_lines / get_nearest_line(配合 detector.py 送 list)

座標慣例:
  cX/cY = 相機原生像素;fw/fh 由封包帶進來;
  angle 範圍 -45~+45,水平=0。

啟動順序:先跑 detector.py,再跑這支。
"""

import rospy
import numpy as np
import time
import socket
import json
import threading
import sys

from op3_ros_utils import Robot
from sensor_msgs.msg import Imu

# ============================================================
#  ★★★  賽前調整區  ★★★
# ============================================================

DEBUG_MODE = False

# ── socket 來源 ────────────────────────────────────────────
DET_BIND_HOST = "0.0.0.0"
DET_BIND_PORT = 5005
DET_STALE_SEC = 0.3

# ── 頭部角度(rad)─────────────────────────────────────────
HEAD_TILT_WALK      = -0.4   # WALK_FORWARD 第一段(找槓鈴遠)
HEAD_TILT_CLOSE     = -0.9   # WALK_FORWARD 第二段(找槓鈴近)
HEAD_TILT_ADJUST_0  = -0.8
HEAD_TILT_ADJUST_1  = -1.1
HEAD_TILT_CHECK     = -1.2
HEAD_TILT_PICK_WALK = -1.5   # PICK 後短暫直走的低頭
HEAD_TILT_LIFTWALK  = -1.5   # WALK_WITH_BAR 抬槓走小段時低頭
HEAD_TILT_LINE_NEAR = -1.5   # ★ STOP_AT_LIFT_LINE / WALK_2_FINISH 看地面找線

# ── cY 停止閾值(實測填) ────────────────────────────────────
CY_STOP_FIRST   = 200   # 接近槓鈴第一停 → Adjust_Mode
CY_STOP_SECOND  = 260   # 接近槓鈴第二停 → Adjust_Mode
CY_STOP_FINAL   = 305   # 接近槓鈴最終停 → CHECK_TOUCH
CY_PICK_READY   = 305   # CHECK_TOUCH 判斷可夾取

# ★ 線停止閾值(實測填) ────────────────────────────────────
#   做法:把機器人擺在「準備停的位置」上,印出 line cY,填進來。
#   注意:walkStop 後會慣性多滑一點,所以這個值要設得比實際略小,抓提前量。
LIFT_LINE_STOP_CY  = 360    # STOP_AT_LIFT_LINE:該停下舉過頭的位置(cY)
FINAL_LINE_STOP_CY = 360    # WALK_2_FINISH:終點線停下放槓鈴的位置(cY)

LINE_CX_KP   = 0.02         # 對準線中心的 y 補正增益
LINE_Y_CLAMP = 2.0          # 對線時 y 補正最大值

# ── 走路參數 ───────────────────────────────────────────────
WALK_FORWARD_X          = 3.0
WALK_FORWARD_Y          = 0.0
WALK_FORWARD_TH_SCALE   = 0.5
WALK_FORWARD_CLAMP      = 3.0

WALK_BIAS_Y             = 0.0
WALK_BIAS_TH            = 0.0

CALIBRATE_MODE          = False
CALIBRATE_TIME          = 5.0

CX_KP                   = 0.02
CX_Y_CLAMP              = 2.5

# WALK_WITH_BAR:抬槓後走「跨過撿起槓鈴附近那條白線」的短段
#   這段用固定時間就好(看線不可靠,負重頭壓低,起點變動小)
WALK_WITH_BAR_X         = 5.0
WALK_WITH_BAR_Y         = 0.0
WALK_WITH_BAR_TIME      = 3.0   # ★ 縮短:只走「跨過第 0 條線」需要的時間
                                #   實測:在槓鈴位置 walkStart → cY 看到第 0 條線
                                #   消失需要幾秒,填那個秒數 + 0.5s 緩衝

# STOP_AT_LIFT_LINE / WALK_2_FINISH 的巡航速度
LINE_CRUISE_X   = 4.0
LINE_CRUISE_Y   = 0.0
LINE_CRUISE_Z   = 0.026

# 安全 timeout(看不到線時最多走這秒數就強制停)
LIFT_LINE_TIMEOUT  = 12.0     # 50cm 距離,設寬鬆一點
FINAL_LINE_TIMEOUT = 12.0

# ── Adjust_Mode ────────────────────────────────────────────
ADJUST_FRAMES       = 5

CENTER_TOL_ADJUST   = 35
ADJUST_SHIFT_Y      = 2.0
ADJUST_SHIFT_X      = -1.0
ADJUST_SHIFT_STEPS  = 2
ADJUST_MAX_SHIFTS   = 8

ANGLE_THRESHOLD     = 5.0
ADJUST_TURN_TH      = 1.0
ADJUST_TURN_X       = -1.5
ADJUST_TURN_STEPS   = 2
ADJUST_MAX_TURNS    = 8

VERIFY_FRAMES       = 7
VERIFY_MAX_RETRY    = 3

# ── CHECK_TOUCH ─────────────────────────────────────────────
CHECK_DETECT_FRAMES = 3
CHECK_CENTER_TOL    = 25

CHECK_STEP_X        = 1.2
CHECK_STEP_TIME     = 0.15

CHECK_SHIFT_LEFT_Y  = -4.0
CHECK_SHIFT_RIGHT_Y =  5.0
CHECK_SHIFT_TIME    = 0.18

# ── 動作 ID ────────────────────────────────────────────────
MOTION_INIT         = 1
MOTION_PICK         = 195
MOTION_LIFT         = 186
MOTION_FINISH       = 185

# ============================================================
#  偵測 socket 客戶端
# ============================================================
class DetClient(object):
    def __init__(self, host=DET_BIND_HOST, port=DET_BIND_PORT, stale=DET_STALE_SEC):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((host, port))
        self.s.settimeout(0.5)
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
rospy.sleep(3)

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
    rospy.sleep(0.8)


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
    rospy.sleep(0.6)


def cruise_until_line(stop_cy, timeout_sec, head_tilt, label):
    """
    共用:邊走邊看,等最近的線 cY >= stop_cy 就停。
    用於 STOP_AT_LIFT_LINE 和 WALK_2_FINISH 的巡航段。
    回傳:('line' 看到線停 / 'timeout' 超時強制停)
    """
    set_head(head_tilt, wait=0.3)
    robot.walkStart()
    t0 = time.time()
    fw = det.get_fw()

    while not rospy.is_shutdown():
        # 安全 timeout
        if time.time() - t0 > timeout_sec:
            print("[{}] timeout {:.1f}s, force stop".format(label, timeout_sec))
            robot.walkStop()
            return "timeout"

        lx, ly, la = det.get_nearest_line()

        if lx is None:
            # 看不到線 → 直走找線(配 IMU 修向)
            robot.walkVelocities(
                x=LINE_CRUISE_X, y=LINE_CRUISE_Y,
                th=np.clip(z_gyro, -10, 10),
                z_move_amplitude=LINE_CRUISE_Z,
                balance=True, z_offset=0.028,
                hip_pitch=0.0, feet_pitch=-4.2
            )
        else:
            if ly >= stop_cy:
                print("[{}] line reached cY={} >= {} -> stop".format(label, ly, stop_cy))
                robot.walkStop()
                return "line"
            # 邊走邊用 cX 對準線中心
            cx_off = lx - (fw // 2)
            y_corr = np.clip(-cx_off * LINE_CX_KP, -LINE_Y_CLAMP, LINE_Y_CLAMP)
            print("[{}] line cY={} cX_off={} y={:.2f}".format(label, ly, cx_off, y_corr))
            robot.walkVelocities(
                x=LINE_CRUISE_X, y=LINE_CRUISE_Y + y_corr,
                th=np.clip(z_gyro, -10, 10),
                z_move_amplitude=LINE_CRUISE_Z,
                balance=True, z_offset=0.028,
                hip_pitch=0.0, feet_pitch=-4.2
            )
        rate.sleep()

    return "shutdown"


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
            rospy.sleep(10)
            reset_imu()
            set_head(HEAD_TILT_WALK)

            if CALIBRATE_MODE:
                print("=" * 50)
                print("[CALIBRATE] {:.0f}s straight walk".format(CALIBRATE_TIME))
                print("=" * 50)
                robot.walkStart()
                t0 = time.time()
                while time.time() - t0 < CALIBRATE_TIME and not rospy.is_shutdown():
                    robot.walkVelocities(
                        x=WALK_FORWARD_X, y=WALK_BIAS_Y, th=WALK_BIAS_TH,
                        z_move_amplitude=0.03, balance=True, z_offset=0)
                    rate.sleep()
                robot.walkStop()
                currState = States.READY
            else:
                robot.walkStart()
                currState = States.WALK_FORWARD

    # ============================================================
    #  WALK_FORWARD(找槓鈴)
    # ============================================================
    elif currState == States.WALK_FORWARD:

        h_tilt = HEAD_TILT_WALK if go_back == 0 else HEAD_TILT_CLOSE
        set_head(h_tilt, wait=0.0)

        th_corr = np.clip(z_gyro * WALK_FORWARD_TH_SCALE,
                          -WALK_FORWARD_CLAMP, WALK_FORWARD_CLAMP)

        if cX is not None:
            cx_offset = cX - (fw // 2)
            y_corr = np.clip(-cx_offset * CX_KP, -CX_Y_CLAMP, CX_Y_CLAMP)
        else:
            cx_offset = 0
            y_corr = 0.0

        print("[WALK_FORWARD] go_back={} cY={} cX={} imu={:.2f} th={:.2f} y={:.2f}".format(
              go_back, cY, cX, z_gyro, th_corr, y_corr))

        if cX is None or cY is None:
            print("[WALK_FORWARD] target lost, slowing")
            robot.walkVelocities(x=0.5, y=0, th=0,
                                 z_move_amplitude=0.03, balance=True, z_offset=0)
            rate.sleep()
            continue

        robot.walkVelocities(
            x=WALK_FORWARD_X,
            y=y_corr + WALK_BIAS_Y,
            th=th_corr + WALK_BIAS_TH,
            z_move_amplitude=0.03,
            balance=True, z_offset=0
        )

        if go_back == 0 and cY >= CY_STOP_FIRST:
            robot.walkStop(); rospy.sleep(0.5)
            position_correct = False
            currState = States.Adjust_Mode

        elif go_back == 1 and cY >= CY_STOP_SECOND:
            robot.walkStop(); rospy.sleep(0.5)
            position_correct = False
            currState = States.Adjust_Mode

        elif go_back == 2 and cY >= CY_STOP_FINAL:
            robot.walkStop(); rospy.sleep(0.5)
            position_correct = False
            currState = States.CHECK_TOUCH

    # ============================================================
    #  Adjust_Mode
    # ============================================================
    elif currState == States.Adjust_Mode:
        print("[Adjust_Mode] go_back={} phase={} try={}".format(
              go_back, adjust_phase, adjust_try_count))

        h_tilt = HEAD_TILT_ADJUST_0 if go_back == 0 else HEAD_TILT_ADJUST_1
        set_head(h_tilt, wait=0.3)

        n_frames = VERIFY_FRAMES if adjust_phase == "verify" else ADJUST_FRAMES
        cx_now, _, ang_now = sample_bar(n_frames)
        angle    = ang_now if ang_now is not None else 0.0
        frame_cx = fw // 2
        offset   = (cx_now - frame_cx) if cx_now is not None else 0
        print("  angle={:.2f} cX={} offset={}".format(angle, cx_now, offset))

        if adjust_phase == "shift":
            if cx_now is None:
                pulse_move(x=-2.0, y=0, th=0, steps=2)
                adjust_try_count += 1
                if adjust_try_count >= ADJUST_MAX_SHIFTS:
                    adjust_phase = "verify"
                    adjust_try_count = 0
                    verify_retry = 0
            elif adjust_try_count >= ADJUST_MAX_SHIFTS:
                adjust_phase = "rotate"; adjust_try_count = 0
            elif offset > CENTER_TOL_ADJUST:
                pulse_move(x=ADJUST_SHIFT_X, y=-ADJUST_SHIFT_Y, th=0,
                           steps=ADJUST_SHIFT_STEPS)
                adjust_try_count += 1
            elif offset < -CENTER_TOL_ADJUST:
                pulse_move(x=ADJUST_SHIFT_X, y=ADJUST_SHIFT_Y, th=0,
                           steps=ADJUST_SHIFT_STEPS)
                adjust_try_count += 1
            else:
                adjust_phase = "rotate"; adjust_try_count = 0
                rospy.sleep(0.3)

        elif adjust_phase == "rotate":
            RECENTER_LIMIT = 120
            if cx_now is not None and abs(offset) > RECENTER_LIMIT:
                adjust_phase = "shift"; adjust_try_count = 0
            elif adjust_try_count >= ADJUST_MAX_TURNS:
                adjust_phase = "verify"; adjust_try_count = 0; verify_retry = 0
            elif abs(angle) <= ANGLE_THRESHOLD:
                adjust_phase = "verify"; adjust_try_count = 0; verify_retry = 0
                rospy.sleep(0.3)
            elif angle > ANGLE_THRESHOLD:
                pulse_move(x=ADJUST_TURN_X, y=0, th=-ADJUST_TURN_TH,
                           steps=ADJUST_TURN_STEPS)
                adjust_try_count += 1
            else:
                pulse_move(x=ADJUST_TURN_X, y=0, th=ADJUST_TURN_TH,
                           steps=ADJUST_TURN_STEPS)
                adjust_try_count += 1

        elif adjust_phase == "verify":
            if cx_now is None:
                position_correct = True
            else:
                angle_ok  = abs(angle) <= ANGLE_THRESHOLD
                center_ok = abs(offset) <= CENTER_TOL_ADJUST
                if angle_ok and center_ok:
                    position_correct = True; rospy.sleep(0.3)
                elif verify_retry >= VERIFY_MAX_RETRY:
                    position_correct = True
                else:
                    adjust_phase = "shift"; adjust_try_count = 0
                    verify_retry += 1; rospy.sleep(0.3)

        if position_correct:
            position_correct = False
            adjust_phase = "shift"; adjust_try_count = 0; verify_retry = 0
            if go_back == 0:
                go_back = 1; currState = States.WALK_FORWARD; rospy.sleep(1.0)
            elif go_back == 1:
                go_back = 2; currState = States.CHECK_TOUCH;  rospy.sleep(1.0)
            elif go_back == 2:
                currState = States.PICK_BAR; rospy.sleep(1.0)

    # ============================================================
    #  CHECK_TOUCH
    # ============================================================
    elif currState == States.CHECK_TOUCH:
        print("[CHECK_TOUCH]")
        robot.walkStop(); rospy.sleep(0.5)
        set_head(HEAD_TILT_CHECK, wait=0.3)

        cX2, cY2, _ = sample_bar(CHECK_DETECT_FRAMES)

        if cX2 is None:
            print("[CHECK_TOUCH] no detection, retry")
            rospy.sleep(0.5)
        else:
            touch, orien = check_center(cX2, fw)
            print("[CHECK_TOUCH] cX2={} cY2={} touch={} orien={}".format(
                  cX2, cY2, touch, orien))
            if touch and cY2 >= CY_PICK_READY:
                robot.walkStop()
                currState = States.PICK_BAR; rospy.sleep(1.0)
            elif touch and cY2 < CY_PICK_READY:
                walk_shift(CHECK_STEP_X, 0, 0, CHECK_STEP_TIME)
            elif orien == 0:
                walk_shift(0, CHECK_SHIFT_LEFT_Y, 0, CHECK_SHIFT_TIME)
            elif orien == 1:
                walk_shift(0, CHECK_SHIFT_RIGHT_Y, 0, CHECK_SHIFT_TIME)

    # ============================================================
    #  PICK_BAR
    # ============================================================
    elif currState == States.PICK_BAR:
        print("[PICK_BAR]")
        robot.walkStop(); rospy.sleep(2.0)

        robot.setGeneralControlModule("none"); rospy.sleep(1.0)
        robot.setGeneralControlModule("action_module"); rospy.sleep(0.5)
        robot.playMotion(MOTION_PICK, wait_for_end=True); rospy.sleep(1.0)

        reset_imu(); rospy.sleep(2.0)

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
            rate.sleep()

        robot.walkStop(); rospy.sleep(1.0)
        currState = States.WALK_WITH_BAR

    # ============================================================
    #  WALK_WITH_BAR(抬槓走短段、跨過撿起槓鈴附近的線)
    # ============================================================
    elif currState == States.WALK_WITH_BAR:
        print("[WALK_WITH_BAR] cross over the bar-side line for {:.1f}s".format(
              WALK_WITH_BAR_TIME))
        rospy.sleep(1.0)

        prepare_leg_walking()
        set_head(HEAD_TILT_LIFTWALK, wait=1.0)
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
            rate.sleep()

        robot.walkStop(); rospy.sleep(1.0)
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
            label="STOP_AT_LIFT_LINE"
        )
        rospy.sleep(1.0)
        currState = States.LIFT_BAR

    # ============================================================
    #  LIFT_BAR(舉過頭)
    # ============================================================
    elif currState == States.LIFT_BAR:
        print("[LIFT_BAR]")
        robot.walkStop(); rospy.sleep(0.3)

        robot.setGeneralControlModule("none"); rospy.sleep(1.0)
        robot.setGeneralControlModule("action_module"); rospy.sleep(0.5)
        robot.playMotion(MOTION_LIFT, wait_for_end=True); rospy.sleep(1.3)

        reset_imu(); rospy.sleep(0.8)

        prepare_leg_walking(); rospy.sleep(0.5)
        currState = States.WALK_2_FINISH

    # ============================================================
    #  WALK_2_FINISH(走到終點線停下放槓鈴)
    #  ● 暖身 + 啟動 + 加速段照舊,讓姿態穩
    #  ● 巡航段:看到最近的線 cY >= FINAL_LINE_STOP_CY 就停
    # ============================================================
    elif currState == States.WALK_2_FINISH:
        print("[WALK_2_FINISH] warmup + cruise to final line")

        # 暖身步
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
                x=3.8, y=LINE_CRUISE_Y,
                th=np.clip(z_gyro, -10, 10),
                z_move_amplitude=0.026,
                balance=True, z_offset=0.028,
                hip_pitch=0.0, feet_pitch=-4.2
            )
            rate.sleep()

        # 巡航段(共用 cruise_until_line)
        result = cruise_until_line(
            stop_cy=FINAL_LINE_STOP_CY,
            timeout_sec=FINAL_LINE_TIMEOUT,
            head_tilt=HEAD_TILT_LINE_NEAR,
            label="WALK_2_FINISH"
        )

        robot.walkStop(); rospy.sleep(1.0)
        finish_and_exit()

    # ============================================================
    #  END
    # ============================================================
    elif currState == States.END:
        finish_and_exit()

    rate.sleep()
