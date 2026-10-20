#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import rospy
import numpy as np
import time
import socket
import json
import threading

from sensor_msgs.msg import Imu

# This version intentionally uses the basketball walking utils/scale.
# If this import fails on OP3, copy op3_ros_utils1.py into this folder.
from op3_ros_utils_marathon import Robot

# ============================================================
#  OP3 Marathon Controller - basketball-front HBB approach
#  Detector side:
#    arrow_detector.py sends UDP JSON to 127.0.0.1:5006
#
#  Main idea copied from basketball front-chasing code:
#    scan target -> turn body to target -> continuous approach -> close fine adjust
#    HBB target uses cx/cy/w/h just like basketball ball detector.
#
#  Controller = Python2 / ROS Kinetic
#  Detector   = Python3 / OpenVINO
# ============================================================

DEBUG_MODE = True

DET_BIND_HOST = "0.0.0.0"
DET_BIND_PORT = 5006
DET_STALE_SEC = 0.50

YOLO_TO_DIR = {
    "forward_arrow": "forward",
    "left_arrow": "left",
    "right_arrow": "right",
    "forward": "forward",
    "left": "left",
    "right": "right",
}

APRILTAG_ID_TO_DIR = {
    1: "left",
    2: "forward",
    3: "right",
}

USE_TAG_CONFIRM = True
ALLOW_YOLO_ONLY = True
TAG_PRIORITY_MODE = True
YOLO_FALLBACK_MIN_SIZE = 120.0
YOLO_FALLBACK_MIN_Y = 380
CONFLICT_STOP_SEC = 0.8

# ============================================================
#  Basketball-style parameters
# ============================================================

def ae_to_rad(value):
    return (float(value) - 2048.0) * 2.0 * np.pi / 4096.0

# Head calibration copied from basketball front target approach.
HEAD_PAN_CENTER = 0.0
HEAD_TILT_READY = -0.65
SCAN_HEAD_TILT = ae_to_rad(1640)       # lower scan view
MARKER_HEAD_PAN = 0.0
MARKER_HEAD_TILT = ae_to_rad(1380)     # lower final fine adjust view
MARKER_APPROACH_HEAD_TILT = ae_to_rad(1600)  # lower continuous approach view
MARKER_NEAR_HEAD_TILT = ae_to_rad(1480)      # lower near slow approach view
HEAD_TILT_SEARCH = SCAN_HEAD_TILT

MOTION_INIT = 1

# Because op3_ros_utils1.walkVelocities internally does "th = th - 1",
# add 1 here so th_actual is closer to what we request.
WALK_TH_ZERO_OFFSET = 1.0

STALE = 0.5

SAMPLE_FRAMES   = 7
SAMPLE_DT       = 0.05
ADJ_SETTLE_TIME = 1.0

# ---------- Scan mode copied from basketball ----------
SCAN_HEAD_PANS = [0.0, -0.18, 0.18, -0.32, 0.32]
SCAN_SETTLE_TIME = 0.25
SCAN_INTERVAL = 0.60
MARKER_LOCK_FRAMES = 4
MARKER_SCAN_MIN_CONF = 0.75
MARKER_SCAN_MIN_Y = 55
MARKER_SCAN_MIN_SIZE = 30

# ---------- Turn-to-target copied from basketball ----------
TURN_PAN_KP = 0.8
TURN_BODY_KP = 3.5
TURN_TH_CLAMP = 5.0
TURN_TH_MIN = 1.8
TURN_RIGHT_EXTRA_GAIN = 1.4
TURN_Z_MOVE_AMPLITUDE = 0.014
TURN_Z_OFFSET = 0.034
TURN_HIP_PITCH = 5.0
TURN_DONE_PAN = 0.22
TURN_DONE_FRAMES = 2
TURN_LOST_TOL = 3
PAN_LIMIT = 1.35
TURN_SETTLE_TIME = 1.0

# ---------- Continuous approach copied from basketball ----------
MARKER_APPROACH_CX_TARGET = 331
MARKER_APPROACH_X = -0.022
MARKER_APPROACH_X_START = -0.010

MARKER_APPROACH_NEAR_SIZE = 110.0
MARKER_APPROACH_NEAR_Y = 390
MARKER_APPROACH_NEAR_X = -0.012

MARKER_APPROACH_STOP_SIZE = 125.0
MARKER_APPROACH_STOP_Y = 440
MARKER_APPROACH_STOP_FRAMES = 3
MARKER_APPROACH_TH_KP = 0.010
MARKER_APPROACH_TH_CLAMP = 0.70
MARKER_APPROACH_X_TOL = 35
MARKER_APPROACH_TH_SIGN = 1.0
MARKER_APPROACH_Z = 0.022
MARKER_APPROACH_Z_OFFSET = 0.030
MARKER_APPROACH_HIP_PITCH = 7.0
MARKER_APPROACH_LOST_MAX = 8
MARKER_APPROACH_TIMEOUT = 22.0

# ---------- Close Adjust_Mode copied from basketball ----------
# Same HBB image targets as basketball ball approach.
MARKER_CX_TARGET = 352
MARKER_CY_TARGET = 440
MARKER_W_TARGET = 96.0
MARKER_H_TARGET = 94.0

MARKER_X_TOL = 28#16
MARKER_Y_TOL = 25#12
MARKER_W_TOL = 22#14
MARKER_H_TOL = 24#16
MARKER_STABLE_FRAMES = 2

MARKER_FINE_ENTER_Y = 340
MARKER_FINE_ENTER_SIZE = 105.0

MARKER_FINE_TH_SIGN = 1.0
MARKER_FINE_TH_KP = 0.025
MARKER_FINE_TH_MIN = 1.5
MARKER_FINE_TH_MAX = 2.2
MARKER_FINE_TURN_TIME = 0.40

# op3_ros_utils1: negative x = forward, positive x = back.
MARKER_FINE_FORWARD_X = -0.010
MARKER_FINE_BACK_X = 0.015
MARKER_FINE_FORWARD_TIME = 0.12
MARKER_FINE_BACK_TIME = 1.0
MARKER_FINE_TURN_X_COMP = 0.006
MARKER_FINE_SETTLE_TIME = 1.0

MARKER_CX_LOCK_FRAMES = 2
MARKER_CX_RETURN_TOL = 24
MARKER_PHASE_SETTLE_TIME = 0.45
MARKER_SAFE_Y = 450
MARKER_FORCE_X_TOL = 35
MARKER_FORCE_Y_MIN = 380
MARKER_FORCE_Y_MAX = 425
MARKER_CX_FINE_ZONE = 40
MARKER_CX_TURN_TIME_NORMAL = 0.8
MARKER_CX_TURN_TIME_FINE = 0.4
MARKER_CX_TH_MIN = 1.2
MARKER_CX_TH_MAX = 1.6
MARKER_CX_SAFETY_Y_MARGIN = 30

MARKER_ADJ_MAX_ITER = 999
MARKER_LOST_MAX = 3

# ---------- After decision / movement ----------
# These use the same basketball walking scale.
FORWARD_MARKER_PASS_X = MARKER_APPROACH_X
FORWARD_MARKER_PASS_TIME = 8.0
AFTER_TURN_FORWARD_X = MARKER_APPROACH_X
AFTER_TURN_FORWARD_TIME = 10.0

ACTION_TURN_TH = 3.5
ACTION_TURN_TIME = 15.0
LEFT_TURN_SIGN = 1.0
RIGHT_TURN_SIGN = -1.0

AFTER_ACTION_WAIT = 0.35
BALANCE_ENABLE = True

# ============================================================
#  IMU yaw correction
# ============================================================

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_calc = 0.0

def imu_callback(data):
    global z_gyro, z_gyro_offset
    av = data.angular_velocity
    z_gyro = z_gyro - av.z / 2.0 + z_gyro_offset_for_calc
    z_gyro_offset = av.z / 2.0

def reset_imu():
    global z_gyro_offset_for_calc, z_gyro
    z_gyro_offset_for_calc = z_gyro_offset
    z_gyro = 0.0

# ============================================================
#  Detector UDP client
# ============================================================

class DetClient(object):
    def __init__(self, host=DET_BIND_HOST, port=DET_BIND_PORT, stale=DET_STALE_SEC):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((host, port))
        self.s.settimeout(0.2)
        self.stale = stale
        self.latest = None

        t = threading.Thread(target=self._loop)
        t.daemon = True
        t.start()

    def _loop(self):
        while True:
            try:
                data, _ = self.s.recvfrom(8192)
                if not isinstance(data, str):
                    data = data.decode("utf-8")
                self.latest = json.loads(data)
            except socket.timeout:
                continue
            except Exception as e:
                if DEBUG_MODE:
                    print("[DET] JSON/socket error:", e)
                continue

    def _fresh(self):
        m = self.latest
        if m is None:
            return None
        try:
            if time.time() - float(m.get("t", 0.0)) > self.stale:
                return None
        except Exception:
            return None
        return m

    def get_fw(self):
        m = self.latest
        if m is None:
            return 640
        return int(m.get("fw", 640))

    def get_fh(self):
        m = self.latest
        if m is None:
            return 480
        return int(m.get("fh", 480))

    def get_marker(self):
        m = self._fresh()
        if not m:
            return None

        a = m.get("arrow", {})
        tag = m.get("tag", {})

        arrow_found = bool(a.get("found", False))
        tag_found = bool(tag.get("found", False))

        if (not arrow_found) and (not tag_found):
            return None

        tag_id = None
        tag_dir = None
        tag_area = 0

        if tag_found:
            try:
                tag_id = int(tag.get("id"))
                tag_dir = APRILTAG_ID_TO_DIR.get(tag_id, None)
                tag_area = int(tag.get("area", 0))
            except Exception:
                tag_id = None
                tag_dir = None
                tag_area = 0

        if arrow_found:
            name = str(a.get("name", "none"))
            direction = YOLO_TO_DIR.get(name, None)
            conf = float(a.get("conf", 0.0))
            accepted = bool(a.get("accepted", False))

            cx = int(a.get("cx", 0))
            cy = int(a.get("cy", 0))
            w = int(a.get("w", 0))
            h = int(a.get("h", 0))
        else:
            name = "tag_only"
            direction = None
            conf = 1.0
            accepted = True

            cx = int(tag.get("cx", 0))
            cy = int(tag.get("cy", 0))

            tag_size = int(np.sqrt(max(tag_area, 0)))
            w = tag_size
            h = tag_size

        bottom_y = int(cy + h / 2)
        area = int(w * h)

        return {
            "name": name,
            "direction": direction,
            "conf": conf,
            "accepted": accepted,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "bottom_y": bottom_y,
            "area": area,
            "tag_found": tag_found,
            "tag_id": tag_id,
            "tag_dir": tag_dir,
            "fw": int(m.get("fw", 640)),
            "fh": int(m.get("fh", 480)),
        }

# ============================================================
#  State machine
# ============================================================

class States:
    INIT = -1
    READY = 0
    SCAN_ARROW = 1
    TURN_TO_ARROW = 2
    APPROACH_ARROW = 3
    FINE_ALIGN = 4
    EXEC_FORWARD = 5
    EXEC_LEFT = 6
    EXEC_RIGHT = 7
    END = 99

LEG_JOINTS = [
    "r_hip_yaw",   "l_hip_yaw",
    "r_hip_roll",  "l_hip_roll",
    "r_hip_pitch", "l_hip_pitch",
    "r_knee",      "l_knee",
    "r_ank_pitch", "l_ank_pitch",
    "r_ank_roll",  "l_ank_roll"
]

# ============================================================
#  Robot helper functions, basketball scale
# ============================================================

def set_head_rad(pan, tilt, wait=0.25):
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [pan, tilt])
    rospy.sleep(wait)

def prepare_leg_walking():
    robot.walkStop()
    rospy.sleep(0.2)
    robot.setJointsControlModule(LEG_JOINTS, ["walking_module"])
    rospy.sleep(0.2)
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    rospy.sleep(0.1)

def walk_cmd(x=0.0, y=0.0, th_actual=0.0, hip_pitch=10, balance=True,
             z_move_amplitude=None, z_offset=None):
    robot.walkVelocities(
        x=x,
        y=y,
        th=th_actual + WALK_TH_ZERO_OFFSET,
        hip_pitch=hip_pitch,
        balance=balance,
        z_move_amplitude=z_move_amplitude,
        z_offset=z_offset
    )

def stop_walk(wait=0.05):
    robot.walkStop()
    rospy.sleep(wait)

def turn_to_target_walk(th_actual, duration):
    print("[TURN_HIGH] th=%.2f duration=%.2f z_amp=%.3f z_offset=%.3f hip=%.1f" % (
          th_actual, duration, TURN_Z_MOVE_AMPLITUDE, TURN_Z_OFFSET, TURN_HIP_PITCH))

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
    while time.time() - t0 < duration and not rospy.is_shutdown():
        elapsed = time.time() - t0
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

def marker_fine_move(x_cmd, th_actual, duration):
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
    while time.time() - t0 < duration and not rospy.is_shutdown():
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
    rospy.sleep(MARKER_FINE_SETTLE_TIME)

def walk_straight_for(duration, x_cmd):
    print("[FORWARD_PASS] x=%.3f time=%.2f" % (x_cmd, duration))
    robot.walkStart()
    t0 = time.time()
    while time.time() - t0 < duration and not rospy.is_shutdown():
        walk_cmd(x=x_cmd, y=0.0, th_actual=0.0,
                 hip_pitch=MARKER_APPROACH_HIP_PITCH,
                 balance=True,
                 z_move_amplitude=MARKER_APPROACH_Z,
                 z_offset=MARKER_APPROACH_Z_OFFSET)
        rospy.sleep(0.05)
    robot.walkStop()
    rospy.sleep(AFTER_ACTION_WAIT)

def turn_action(direction):
    if direction == "left":
        sign = LEFT_TURN_SIGN
    else:
        sign = RIGHT_TURN_SIGN

    th_cmd = sign * ACTION_TURN_TH
    print("[EXEC_TURN_%s] th=%.2f time=%.2f" % (direction.upper(), th_cmd, ACTION_TURN_TIME))
    turn_to_target_walk(th_cmd, ACTION_TURN_TIME)

# ============================================================
#  Detection helpers
# ============================================================

track_marker_x = 320
track_marker_y = 240
track_marker_w = 0.0
track_marker_h = 0.0


def valid_marker_for_scan(info):
    if info is None:
        return False
    x = int(info["cx"])
    y = int(info["cy"])
    w = float(info["w"])
    h = float(info["h"])
    conf = float(info["conf"])

    if conf < MARKER_SCAN_MIN_CONF:
        return False
    if y < MARKER_SCAN_MIN_Y:
        return False
    if max(w, h) < MARKER_SCAN_MIN_SIZE:
        return False
    if x < 40 or x > 600:
        return False
    return True


def smooth_marker(info, alpha=0.30):
    global track_marker_x, track_marker_y, track_marker_w, track_marker_h

    x = int(info["cx"])
    y = int(info["cy"])
    w = float(info["w"])
    h = float(info["h"])

    track_marker_x = int(alpha * track_marker_x + (1.0 - alpha) * x)
    track_marker_y = int(alpha * track_marker_y + (1.0 - alpha) * y)
    track_marker_w = alpha * track_marker_w + (1.0 - alpha) * w
    track_marker_h = alpha * track_marker_h + (1.0 - alpha) * h

    return track_marker_x, track_marker_y, track_marker_w, track_marker_h


def sample_marker_median(n=SAMPLE_FRAMES, dt=SAMPLE_DT):
    xs, ys, ws, hs, bottoms, areas, infos = [], [], [], [], [], [], []
    dirs = []
    tag_dirs = []

    for _ in range(n):
        info = det.get_marker()
        if info is not None:
            xs.append(float(info["cx"]))
            ys.append(float(info["cy"]))
            ws.append(float(info["w"]))
            hs.append(float(info["h"]))
            bottoms.append(float(info["bottom_y"]))
            areas.append(float(info["area"]))
            infos.append(info)
            if info.get("direction") is not None:
                dirs.append(info.get("direction"))
            if info.get("tag_dir") is not None:
                tag_dirs.append(info.get("tag_dir"))
        rospy.sleep(dt)

    if not xs:
        return None

    info = infos[-1]
    info["cx"] = int(np.median(xs))
    info["cy"] = int(np.median(ys))
    info["w"] = float(np.median(ws))
    info["h"] = float(np.median(hs))
    info["bottom_y"] = int(np.median(bottoms))
    info["area"] = int(np.median(areas))

    if dirs:
        info["direction"] = max(set(dirs), key=dirs.count)
    if tag_dirs:
        info["tag_dir"] = max(set(tag_dirs), key=tag_dirs.count)

    return info


def decide_direction(info):
    global yolo_fallback_enabled

    if info is None:
        return None, False, "no_marker"

    yolo_dir = info.get("direction", None)
    tag_dir = info.get("tag_dir", None)
    accepted = bool(info.get("accepted", False))

    cy = int(info.get("cy", 0))
    w = float(info.get("w", 0))
    h = float(info.get("h", 0))
    size = max(w, h)

    if TAG_PRIORITY_MODE and tag_dir is not None:
        return tag_dir, True, "tag_priority"

    if not yolo_fallback_enabled:
        return None, False, "tag_missing_yolo_disabled"

    if yolo_dir is None:
        return None, False, "bad_yolo_name"

    if not accepted:
        return None, False, "not_accepted"

    if ALLOW_YOLO_ONLY:
        if size < YOLO_FALLBACK_MIN_SIZE and cy < YOLO_FALLBACK_MIN_Y:
            return None, False, "tag_missing_yolo_too_far"

        return yolo_dir, True, "yolo_fallback_enabled"

    return None, False, "tag_missing"

# ============================================================
#  Main
# ============================================================

rospy.init_node("marathon_controller")
rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_callback, queue_size=1)

robot = Robot()
rospy.sleep(0.3)

det = DetClient()
rate = rospy.Rate(60)

currState = States.INIT
selected_direction = None
yolo_fallback_enabled = False
last_user_button_time = 0.0

scan_idx = 0
scan_last_time = 0.0
scan_settle_until = 0.0
scan_pan_now = SCAN_HEAD_PANS[0]
marker_lock_count = 0
turn_pan = 0.0
turn_done_count = 0
lost_count = 0
marker_adj_iter = 0
marker_adjust_phase = "cx"
marker_cx_ok_count = 0
marker_ok_count = 0
last_good_marker_w = 0.0
last_marker_x = 320
last_marker_y = 240
last_marker_w = 0
last_marker_h = 0

print("[INFO] Marathon controller started")
print("[INFO] Waiting UDP on %s:%d" % (DET_BIND_HOST, DET_BIND_PORT))
print("[INFO] Using basketball utils scale: op3_ros_utils1")

while not rospy.is_shutdown():

    if currState != States.READY and robot.buttonCheck("user"):
        now_btn = time.time()
        if now_btn - last_user_button_time > 1.0:
            yolo_fallback_enabled = not yolo_fallback_enabled
            ALLOW_YOLO_ONLY = yolo_fallback_enabled
            last_user_button_time = now_btn
            print("[BUTTON] user -> YOLO fallback enabled = %s" % str(yolo_fallback_enabled))
            rospy.sleep(0.2)

    if robot.buttonCheck("mode"):
        print("[BUTTON] mode -> reset")
        robot.walkStop()
        currState = States.INIT
        selected_direction = None
        scan_idx = 0
        marker_lock_count = 0
        turn_done_count = 0
        lost_count = 0
        marker_adj_iter = 0
        marker_adjust_phase = "cx"
        marker_cx_ok_count = 0
        marker_ok_count = 0

    # ============================================================
    # INIT
    # ============================================================
    if currState == States.INIT:
        print("[INIT]")
        robot.walkStop()
        rospy.sleep(0.2)

        robot.setGeneralControlModule("action_module")
        rospy.sleep(0.3)
        robot.playMotion(MOTION_INIT, wait_for_end=True)
        rospy.sleep(0.3)

        robot.setGeneralControlModule("walking_module")
        rospy.sleep(0.3)
        prepare_leg_walking()
        set_head_rad(HEAD_PAN_CENTER, HEAD_TILT_READY, wait=0.8)

        currState = States.READY

    # ============================================================
    # READY
    # ============================================================
    elif currState == States.READY:
        print("[READY] waiting start/user button...")
        if robot.buttonCheck("start") or robot.buttonCheck("user"):
            print("[START]")
            reset_imu()
            scan_idx = 0
            scan_last_time = 0.0
            scan_settle_until = 0.0
            marker_lock_count = 0
            lost_count = 0
            marker_adj_iter = 0
            marker_adjust_phase = "cx"
            marker_cx_ok_count = 0
            marker_ok_count = 0
            scan_pan_now = SCAN_HEAD_PANS[0]
            set_head_rad(scan_pan_now, SCAN_HEAD_TILT, wait=0.4)
            stop_walk(0.2)
            currState = States.SCAN_ARROW

    # ============================================================
    # 0) Scan target: basketball scan_ball style
    # ============================================================
    elif currState == States.SCAN_ARROW:
        info = det.get_marker()
        now = time.time()

        if now < scan_settle_until:
            stop_walk(0.03)
            continue

        if valid_marker_for_scan(info):
            x, y, w, h = smooth_marker(info, alpha=0.30)
            marker_lock_count += 1
            print("[MARKER_SCAN_LOCK] pan=%.2f x=%d y=%d w=%.1f h=%.1f conf=%.2f lock=%d/%d" % (
                  scan_pan_now, x, y, w, h, info["conf"], marker_lock_count, MARKER_LOCK_FRAMES))
            stop_walk(0.05)

            if marker_lock_count >= MARKER_LOCK_FRAMES:
                px_corr = (320.0 - float(x)) * (45.0 / 320.0) * np.pi / 180.0
                turn_pan = float(np.clip(scan_pan_now + px_corr, -PAN_LIMIT, PAN_LIMIT))
                set_head_rad(turn_pan, SCAN_HEAD_TILT, wait=0.3)
                turn_done_count = 0
                lost_count = 0
                currState = States.TURN_TO_ARROW
                print("[STAGE] scan_arrow -> turn_to_arrow, marker direction pan=%.2f" % turn_pan)

        else:
            marker_lock_count = 0
            if now - scan_last_time >= SCAN_INTERVAL:
                pan = SCAN_HEAD_PANS[scan_idx % len(SCAN_HEAD_PANS)]
                scan_pan_now = pan
                set_head_rad(pan, SCAN_HEAD_TILT, wait=0.0)
                print("[MARKER_SCAN] no valid marker, head_pan=%.2f head_tilt=%.2f" % (pan, SCAN_HEAD_TILT))
                scan_idx += 1
                scan_last_time = now
                scan_settle_until = now + SCAN_SETTLE_TIME

                if scan_idx % len(SCAN_HEAD_PANS) == 0:
                    print("[MARKER_SCAN] full sweep no valid marker -> blind forward 5s and rescan")
                    walk_straight_for(5.0, -0.018)
                    scan_idx = 0
                    scan_last_time = 0.0
                    scan_settle_until = time.time() + SCAN_SETTLE_TIME
                    marker_lock_count = 0
                    scan_pan_now = SCAN_HEAD_PANS[0]
                    set_head_rad(scan_pan_now, SCAN_HEAD_TILT, wait=0.2)
                else:
                    stop_walk(0.05)

    # ============================================================
    # 0.5) Turn toward target: basketball turn_to_ball style
    # ============================================================
    elif currState == States.TURN_TO_ARROW:
        info = sample_marker_median()

        if info is not None:
            lost_count = 0
            last_good_marker_w = float(info["w"])

            bx = int(info["cx"])
            bw = float(info["w"])

            px_corr = (320.0 - float(bx)) * (45.0 / 320.0) * np.pi / 180.0
            turn_pan = float(np.clip(turn_pan + px_corr * TURN_PAN_KP, -PAN_LIMIT, PAN_LIMIT))
            set_head_rad(turn_pan, SCAN_HEAD_TILT, wait=0.25)

            if abs(turn_pan) < TURN_DONE_PAN:
                turn_done_count += 1
            else:
                turn_done_count = 0

            print("[TURN2MARKER] x=%d w=%.1f pan=%.2f done=%d/%d" % (
                  bx, bw, turn_pan, turn_done_count, TURN_DONE_FRAMES))

            if turn_done_count >= TURN_DONE_FRAMES:
                stop_walk(0.05)
                set_head_rad(MARKER_HEAD_PAN, MARKER_APPROACH_HEAD_TILT, wait=0.4)
                lost_count = 0
                marker_adj_iter = 0
                marker_adjust_phase = "cx"
                marker_cx_ok_count = 0
                marker_ok_count = 0
                currState = States.APPROACH_ARROW
                print("[STAGE] turn_to_arrow -> approach_arrow / continuous walk")
            else:
                raw_th = turn_pan * TURN_BODY_KP
                if abs(raw_th) < TURN_TH_MIN:
                    if raw_th > 0:
                        raw_th = TURN_TH_MIN
                    else:
                        raw_th = -TURN_TH_MIN

                th_cmd = float(np.clip(raw_th, -TURN_TH_CLAMP, TURN_TH_CLAMP))
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

                print("[TURN2MARKER_%s] pan=%.2f th=%.2f time=%.2f" % (
                      turn_mode, turn_pan, th_cmd, turn_time))

                turn_to_target_walk(th_actual=th_cmd, duration=turn_time)
                turn_pan = 0.0
                set_head_rad(turn_pan, SCAN_HEAD_TILT, wait=0.3)

        else:
            lost_count += 1
            stop_walk(0.05)
            print("[TURN2MARKER] lost marker (%d/%d)" % (lost_count, TURN_LOST_TOL))
            if lost_count >= TURN_LOST_TOL:
                currState = States.SCAN_ARROW
                scan_idx = 0
                scan_last_time = 0.0
                marker_lock_count = 0
                scan_pan_now = SCAN_HEAD_PANS[0]
                set_head_rad(scan_pan_now, SCAN_HEAD_TILT, wait=0.2)
                scan_settle_until = time.time() + SCAN_SETTLE_TIME
                print("[TURN2MARKER] lost too long -> scan_arrow")

    # ============================================================
    # 0.8) Continuous approach: basketball approach_ball style
    # ============================================================
    elif currState == States.APPROACH_ARROW:
        robot.setGeneralControlModule("walking_module")
        set_head_rad(MARKER_HEAD_PAN, MARKER_APPROACH_HEAD_TILT, wait=0.3)

        approach_lost = 0
        approach_close_count = 0
        approach_near_mode = False
        approach_start = time.time()

        robot.walkStart()
        while currState == States.APPROACH_ARROW and not rospy.is_shutdown():
            elapsed = time.time() - approach_start

            if elapsed >= MARKER_APPROACH_TIMEOUT:
                print("[MARKER_APPROACH] timeout -> stop and enter fine mode")
                robot.walkStop()
                marker_ok_count = 0
                marker_cx_ok_count = 0
                marker_adjust_phase = "cx"
                lost_count = 0
                marker_adj_iter = 0
                currState = States.FINE_ALIGN
                break

            info = det.get_marker()

            if info is None:
                approach_lost += 1
                walk_cmd(x=0.0, y=0.0, th_actual=0.0,
                         z_move_amplitude=MARKER_APPROACH_Z,
                         balance=True,
                         z_offset=MARKER_APPROACH_Z_OFFSET,
                         hip_pitch=MARKER_APPROACH_HIP_PITCH)

                print("[MARKER_APPROACH] lost %d/%d" % (approach_lost, MARKER_APPROACH_LOST_MAX))

                if approach_lost >= MARKER_APPROACH_LOST_MAX:
                    robot.walkStop()
                    currState = States.SCAN_ARROW
                    scan_idx = 0
                    scan_last_time = 0.0
                    marker_lock_count = 0
                    print("[MARKER_APPROACH] lost too long -> scan_arrow")
                    break

                rospy.sleep(0.05)
                continue

            approach_lost = 0

            bx = int(info["cx"])
            by = int(info["cy"])
            bw = float(info["w"])
            bh = float(info["h"])
            size = max(bw, bh)
            x_err = MARKER_APPROACH_CX_TARGET - bx

            last_good_marker_w = bw
            last_marker_x = bx
            last_marker_y = by
            last_marker_w = int(bw)
            last_marker_h = int(bh)

            if not approach_near_mode:
                if size >= MARKER_APPROACH_NEAR_SIZE or by >= MARKER_APPROACH_NEAR_Y:
                    approach_close_count += 1
                else:
                    approach_close_count = 0
            else:
                if size >= MARKER_APPROACH_STOP_SIZE or by >= MARKER_APPROACH_STOP_Y:
                    approach_close_count += 1
                else:
                    approach_close_count = 0

            print("[MARKER_APPROACH] x=%d y=%d w=%.1f h=%.1f size=%.1f near=%s close=%d/%d" % (
                bx, by, bw, bh, size, str(approach_near_mode), approach_close_count, MARKER_APPROACH_STOP_FRAMES))

            if (not approach_near_mode) and approach_close_count >= MARKER_APPROACH_STOP_FRAMES:
                approach_near_mode = True
                approach_close_count = 0
                set_head_rad(MARKER_HEAD_PAN, MARKER_NEAR_HEAD_TILT, wait=0.25)
                print("[MARKER_APPROACH] enter near slow approach")
            elif approach_near_mode and approach_close_count >= MARKER_APPROACH_STOP_FRAMES:
                robot.walkStop()
                rospy.sleep(0.5)

                set_head_rad(MARKER_HEAD_PAN, MARKER_HEAD_TILT, wait=0.6)
                marker_ok_count = 0
                marker_cx_ok_count = 0
                marker_adjust_phase = "cx"
                lost_count = 0
                marker_adj_iter = 0
                currState = States.FINE_ALIGN
                print("[STAGE] approach_arrow -> fine_align / closer Fine Adjust_Mode")
                break

            if abs(x_err) > MARKER_APPROACH_X_TOL:
                th_actual = float(np.clip(
                    MARKER_APPROACH_TH_SIGN * MARKER_APPROACH_TH_KP * float(x_err),
                    -MARKER_APPROACH_TH_CLAMP,
                    MARKER_APPROACH_TH_CLAMP
                ))
            else:
                th_actual = 0.0

            if elapsed < 0.8:
                ratio = elapsed / 0.8
                x_cmd = MARKER_APPROACH_X_START + (MARKER_APPROACH_X - MARKER_APPROACH_X_START) * ratio
            else:
                x_cmd = MARKER_APPROACH_X

            if abs(x_err) > 70:
                x_cmd *= 0.45

            if approach_near_mode:
                x_cmd = MARKER_APPROACH_NEAR_X
                if abs(x_err) > 70:
                    x_cmd *= 0.5

            walk_cmd(x=x_cmd, y=0.0, th_actual=th_actual,
                     z_move_amplitude=MARKER_APPROACH_Z,
                     balance=True,
                     z_offset=MARKER_APPROACH_Z_OFFSET,
                     hip_pitch=MARKER_APPROACH_HIP_PITCH)

            rospy.sleep(0.05)

    # ============================================================
    # 1) Fine alignment: basketball find_ball style, then decide arrow direction
    # ============================================================
    elif currState == States.FINE_ALIGN:
        marker_adj_iter += 1

        if marker_adj_iter > MARKER_ADJ_MAX_ITER:
            print("[MARKER_ADJ] max iter %d -> decide anyway" % MARKER_ADJ_MAX_ITER)
            stop_walk(0.05)
            final_info = sample_marker_median(5, 0.08)
            direction, ok, reason = decide_direction(final_info)
            if ok:
                selected_direction = direction
                if direction == "forward":
                    currState = States.EXEC_FORWARD
                elif direction == "left":
                    currState = States.EXEC_LEFT
                elif direction == "right":
                    currState = States.EXEC_RIGHT
                else:
                    currState = States.SCAN_ARROW
            else:
                print("[MARKER_ADJ] bad max-iter decision reason=%s -> scan" % reason)
                currState = States.SCAN_ARROW
            continue

        set_head_rad(MARKER_HEAD_PAN, MARKER_HEAD_TILT, wait=0.0)
        info = sample_marker_median()

        if info is not None:
            lost_count = 0
            last_good_marker_w = float(info["w"])
            last_marker_x = int(info["cx"])
            last_marker_y = int(info["cy"])
            last_marker_w = int(info["w"])
            last_marker_h = int(info["h"])

            bx = int(info["cx"])
            by = int(info["cy"])
            bw = float(info["w"])
            bh = float(info["h"])

            x_err = MARKER_CX_TARGET - bx
            y_err = MARKER_CY_TARGET - by
            w_err = MARKER_W_TARGET - bw
            h_err = MARKER_H_TARGET - bh

            fine_mode = (by >= MARKER_FINE_ENTER_Y or max(bw, bh) >= MARKER_FINE_ENTER_SIZE)
            cx_ok = abs(x_err) <= MARKER_X_TOL
            cy_ok = abs(y_err) <= MARKER_Y_TOL

            direction, ok, reason = decide_direction(info)

            print("[MARKER_ADJ] iter=%d phase=%s dir=%s ok=%s reason=%s x=%d y=%d w=%.1f h=%.1f ex=%d ey=%d ew=%.1f eh=%.1f cx_ok=%s cy_ok=%s final=%d/%d" % (
                  marker_adj_iter, marker_adjust_phase, str(direction), str(ok), reason,
                  bx, by, bw, bh, x_err, y_err, w_err, h_err,
                  str(cx_ok), str(cy_ok), marker_ok_count, MARKER_STABLE_FRAMES))

            if fine_mode:
                # Safety: marker is already very close. Stop and decide.
                if by >= MARKER_SAFE_Y:
                    stop_walk(0.05)
                    final_info = sample_marker_median(5, 0.08)
                    direction, ok, reason = decide_direction(final_info)
                    print("[MARKER_SAFE_DECIDE] direction=%s ok=%s reason=%s -> execute/scan" % (
                          str(direction), str(ok), reason))

                    if ok:
                        selected_direction = direction
                        if direction == "forward":
                            currState = States.EXEC_FORWARD
                        elif direction == "left":
                            currState = States.EXEC_LEFT
                        elif direction == "right":
                            currState = States.EXEC_RIGHT
                        else:
                            currState = States.SCAN_ARROW
                    else:
                        rospy.sleep(CONFLICT_STOP_SEC)
                        currState = States.SCAN_ARROW

                elif marker_adjust_phase == "cx":
                    marker_ok_count = 0

                    if by > MARKER_CY_TARGET + MARKER_CX_SAFETY_Y_MARGIN:
                        marker_cx_ok_count = 0
                        print("[MARKER_CX_SAFE_BACK] x=%d y=%d ex=%d ey=%d back=%.3f time=%.2f" % (
                              bx, by, x_err, y_err, MARKER_FINE_BACK_X, MARKER_FINE_BACK_TIME))
                        marker_fine_move(x_cmd=MARKER_FINE_BACK_X, th_actual=0.0, duration=MARKER_FINE_BACK_TIME)

                    elif cx_ok:
                        marker_cx_ok_count += 1
                        stop_walk(0.05)
                        print("[MARKER_CX_HOLD] x=%d target=%d ex=%d lock=%d/%d" % (
                              bx, MARKER_CX_TARGET, x_err, marker_cx_ok_count, MARKER_CX_LOCK_FRAMES))

                        if marker_cx_ok_count >= MARKER_CX_LOCK_FRAMES:
                            marker_adjust_phase = "cy"
                            marker_cx_ok_count = 0
                            marker_ok_count = 0
                            print("[MARKER_PHASE] cx aligned -> cy")
                        rospy.sleep(MARKER_PHASE_SETTLE_TIME)

                    else:
                        marker_cx_ok_count = 0
                        cx_pan_error = float(x_err) * (45.0 / 320.0) * np.pi / 180.0
                        raw_th = cx_pan_error * TURN_BODY_KP

                        if abs(raw_th) < MARKER_CX_TH_MIN:
                            if raw_th > 0:
                                raw_th = MARKER_CX_TH_MIN
                            else:
                                raw_th = -MARKER_CX_TH_MIN

                        th_cmd = float(np.clip(raw_th, -MARKER_CX_TH_MAX, MARKER_CX_TH_MAX))

                        if abs(x_err) <= MARKER_CX_FINE_ZONE:
                            turn_time = MARKER_CX_TURN_TIME_FINE
                            turn_label = "FINE"
                        else:
                            turn_time = MARKER_CX_TURN_TIME_NORMAL
                            turn_label = "NORMAL"

                        print("[MARKER_CX_TURN_%s] x=%d target=%d ex=%d pan_error=%.3f th=%.2f time=%.2f" % (
                              turn_label, bx, MARKER_CX_TARGET, x_err, cx_pan_error, th_cmd, turn_time))

                        turn_to_target_walk(th_actual=th_cmd, duration=turn_time)

                elif marker_adjust_phase == "cy":
                    if abs(x_err) > MARKER_CX_RETURN_TOL:
                        stop_walk(0.05)
                        marker_adjust_phase = "cx"
                        marker_cx_ok_count = 0
                        marker_ok_count = 0
                        print("[MARKER_PHASE] cx drifted x=%d target=%d ex=%d -> cx" % (
                              bx, MARKER_CX_TARGET, x_err))
                        rospy.sleep(MARKER_PHASE_SETTLE_TIME)

                    elif by < MARKER_CY_TARGET - MARKER_Y_TOL:
                        marker_ok_count = 0
                        print("[MARKER_CY_FORWARD] x=%d y=%d target_y=%d ex=%d ey=%d fwd=%.3f time=%.2f" % (
                              bx, by, MARKER_CY_TARGET, x_err, y_err, MARKER_FINE_FORWARD_X, MARKER_FINE_FORWARD_TIME))
                        marker_fine_move(x_cmd=MARKER_FINE_FORWARD_X, th_actual=0.0, duration=MARKER_FINE_FORWARD_TIME)

                    elif by > MARKER_CY_TARGET + MARKER_Y_TOL:
                        # Too close for the ideal CY. For marathon marker, decide instead of backing repeatedly.
                        stop_walk(0.05)
                        final_info = sample_marker_median(5, 0.08)
                        direction, ok, reason = decide_direction(final_info)
                        print("[MARKER_TOO_CLOSE_DECIDE] direction=%s ok=%s reason=%s" % (
                              str(direction), str(ok), reason))

                        if ok:
                            selected_direction = direction
                            if direction == "forward":
                                currState = States.EXEC_FORWARD
                            elif direction == "left":
                                currState = States.EXEC_LEFT
                            elif direction == "right":
                                currState = States.EXEC_RIGHT
                            else:
                                currState = States.SCAN_ARROW
                        else:
                            rospy.sleep(CONFLICT_STOP_SEC)
                            currState = States.SCAN_ARROW

                    else:
                        if cx_ok and cy_ok:
                            marker_ok_count += 1
                            stop_walk(0.05)
                            print("[MARKER_FINAL_HOLD] x=%d y=%d ok=%d/%d" % (
                                  bx, by, marker_ok_count, MARKER_STABLE_FRAMES))

                            if marker_ok_count >= MARKER_STABLE_FRAMES:
                                final_info = sample_marker_median(5, 0.08)
                                direction, ok, reason = decide_direction(final_info)
                                print("[DECIDE_FINAL] direction=%s ok=%s reason=%s" % (
                                      str(direction), str(ok), reason))
                                if ok:
                                    selected_direction = direction
                                    if direction == "forward":
                                        currState = States.EXEC_FORWARD
                                    elif direction == "left":
                                        currState = States.EXEC_LEFT
                                    elif direction == "right":
                                        currState = States.EXEC_RIGHT
                                    else:
                                        currState = States.SCAN_ARROW
                                else:
                                    rospy.sleep(CONFLICT_STOP_SEC)
                                    currState = States.SCAN_ARROW
                            else:
                                rospy.sleep(MARKER_FINE_SETTLE_TIME)

                        else:
                            stop_walk(0.05)
                            marker_adjust_phase = "cx"
                            marker_cx_ok_count = 0
                            marker_ok_count = 0
                            print("[MARKER_PHASE] cy ready but cx not ready x=%d ex=%d -> cx" % (bx, x_err))
                            rospy.sleep(MARKER_PHASE_SETTLE_TIME)

            else:
                # Not yet in close fine mode. Keep basketball distant fine approach.
                marker_adjust_phase = "cx"
                marker_cx_ok_count = 0
                marker_ok_count = 0

                if abs(x_err) > 40:
                    raw_th = MARKER_FINE_TH_SIGN * MARKER_FINE_TH_KP * float(x_err)
                    if abs(raw_th) < MARKER_FINE_TH_MIN:
                        if raw_th > 0:
                            raw_th = MARKER_FINE_TH_MIN
                        else:
                            raw_th = -MARKER_FINE_TH_MIN
                    th_cmd = float(np.clip(raw_th, -MARKER_FINE_TH_MAX, MARKER_FINE_TH_MAX))

                    print("[MARKER_APPROACH_TURN] ex=%d th=%.2f" % (x_err, th_cmd))
                    marker_fine_move(x_cmd=MARKER_FINE_TURN_X_COMP, th_actual=th_cmd, duration=0.55)
                else:
                    print("[MARKER_APPROACH_FORWARD] x=-0.012 time=0.15")
                    marker_fine_move(x_cmd=-0.012, th_actual=0.0, duration=0.15)

        else:
            lost_count += 1
            marker_ok_count = 0
            marker_cx_ok_count = 0
            stop_walk(0.05)
            print("[MARKER_ADJ] no detection (%d/%d), last_w=%.1f" % (
                  lost_count, MARKER_LOST_MAX, last_good_marker_w))

            if lost_count >= MARKER_LOST_MAX:
                if last_good_marker_w >= MARKER_FINE_ENTER_SIZE:
                    print("[MARKER_ADJ] close marker lost -> small back and retry")
                    marker_fine_move(x_cmd=MARKER_FINE_BACK_X, th_actual=0.0, duration=0.15)
                    lost_count = 0
                    marker_ok_count = 0
                    marker_cx_ok_count = 0
                    marker_adjust_phase = "cx"
                else:
                    print("[MARKER_ADJ] lost while far -> scan_arrow")
                    lost_count = 0
                    marker_adj_iter = 0
                    marker_ok_count = 0
                    marker_cx_ok_count = 0
                    marker_adjust_phase = "cx"
                    currState = States.SCAN_ARROW
                    scan_idx = 0
                    scan_last_time = 0.0
                    scan_settle_until = 0.0
                    marker_lock_count = 0
                    scan_pan_now = 0.0

    # ============================================================
    # EXEC_FORWARD / EXEC_LEFT / EXEC_RIGHT
    # ============================================================
    elif currState == States.EXEC_FORWARD:
        print("[EXEC_FORWARD] pass forward marker")
        walk_straight_for(FORWARD_MARKER_PASS_TIME, FORWARD_MARKER_PASS_X)
        selected_direction = None
        currState = States.SCAN_ARROW
        scan_idx = 0
        marker_lock_count = 0
        set_head_rad(SCAN_HEAD_PANS[0], SCAN_HEAD_TILT, wait=0.2)

    elif currState == States.EXEC_LEFT:
        print("[EXEC_LEFT]")
        turn_action("left")
        walk_straight_for(AFTER_TURN_FORWARD_TIME, AFTER_TURN_FORWARD_X)
        selected_direction = None
        currState = States.SCAN_ARROW
        scan_idx = 0
        marker_lock_count = 0
        set_head_rad(SCAN_HEAD_PANS[0], SCAN_HEAD_TILT, wait=0.2)

    elif currState == States.EXEC_RIGHT:
        print("[EXEC_RIGHT]")
        turn_action("right")
        walk_straight_for(AFTER_TURN_FORWARD_TIME, AFTER_TURN_FORWARD_X)
        selected_direction = None
        currState = States.SCAN_ARROW
        scan_idx = 0
        marker_lock_count = 0
        set_head_rad(SCAN_HEAD_PANS[0], SCAN_HEAD_TILT, wait=0.2)

    elif currState == States.END:
        robot.walkStop()
        rospy.signal_shutdown("END")

    rate.sleep()
