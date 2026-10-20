#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import math
import socket
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import Imu

from op3_ros_utils_marathon import Robot

# ============================================================
# OP3 Marathon Controller
# New strategy:
#   1. INIT locks the head at a low angle.
#   2. The robot continuously walks straight and does not scan for AprilTags.
#   3. A Tag is accepted only when it is close and near the image center.
#   4. Forward Tag -> keep walking.
#   5. Left/Right Tag -> finish the turn, reset IMU heading, then keep walking.
#   6. Arrow detection is independently controlled by True/False.
#   7. IMU yaw feedback corrects straight walking only; it does not affect turns.
#
# Controller: Python 2 / ROS Kinetic
# Detector:   Python 3 / OpenVINO, UDP JSON port 5006
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

# Keep the mapping from the user's current controller.
APRILTAG_ID_TO_DIR = {
    1: "left",
    2: "forward",
    3: "right",
}

# ============================================================
# Main True / False switches
# ============================================================

# True: keep the head locked at the low cruise angle during INIT and straight walking.
LOCK_HEAD_LOW_DURING_CRUISE = True

# AprilTag is checked only while cruising and only inside the trigger region.
ENABLE_APRILTAG_CRUISE = True

# Arrow recognition can be disabled when testing only AprilTags.
ENABLE_ARROW_DETECTION = False

# Use IMU yaw closed-loop correction during straight cruising.
USE_IMU_CRUISE_CORRECTION = True

# Keep the original close-approach style after an arrow appears.
USE_ARROW_APPROACH_FLOW = False

# ============================================================
# Head calibration and motion settings
# ============================================================

def ae_to_rad(value):
    return (float(value) - 2048.0) * 2.0 * np.pi / 4096.0

HEAD_PAN_CENTER = 0.0
HEAD_TILT_READY = -0.8

# Fixed low head angle for the passive Tag cruise.
# Smaller Action Editor value means the camera looks lower.
CRUISE_HEAD_PAN = 0.0
CRUISE_HEAD_TILT = ae_to_rad(1450)

# Arrow processing head angles retained from the current program.
ARROW_SCAN_HEAD_TILT = ae_to_rad(1700)
ARROW_APPROACH_HEAD_TILT = ae_to_rad(1690)
ARROW_NEAR_HEAD_TILT = ae_to_rad(1556)
ARROW_FINE_HEAD_TILT = ae_to_rad(1450)

MOTION_INIT = 1

# op3_ros_utils_marathon.walkVelocities internally applies th = th - 1.
# Adding 1 here makes th_actual approximately equal to the requested value.
WALK_TH_ZERO_OFFSET = 1.0

LEG_JOINTS = [
    "r_hip_yaw", "l_hip_yaw",
    "r_hip_roll", "l_hip_roll",
    "r_hip_pitch", "l_hip_pitch",
    "r_knee", "l_knee",
    "r_ank_pitch", "l_ank_pitch",
    "r_ank_roll", "l_ank_roll"
]

# ============================================================
# Passive straight cruise parameters
# ============================================================

CRUISE_X = -0.018
CRUISE_Y = 0.0
CRUISE_HIP_PITCH = 7.0
CRUISE_Z_MOVE_AMPLITUDE = 0.022
CRUISE_Z_OFFSET = 0.030
CRUISE_START_RAMP_SEC = 0.8

# Ignore the previous marker briefly after making a decision.
MARKER_IGNORE_SEC = 2.5

# ============================================================
# Passive AprilTag trigger parameters
# ============================================================

TAG_CENTER_X = 320
TAG_CENTER_X_TOL = 100
TAG_TRIGGER_MIN_Y = 300
TAG_TRIGGER_MAX_Y = 475
TAG_TRIGGER_MIN_AREA = 900
TAG_TRIGGER_LOCK_FRAMES = 3
TAG_STOP_SETTLE_SEC = 0.45
TAG_SAMPLE_FRAMES = 7
TAG_SAMPLE_DT = 0.06
TAG_LOST_AFTER_STOP_SEC = 0.60

# Require the same Tag ID during the lock sequence.
TAG_REQUIRE_SAME_ID = True

# ============================================================
# IMU yaw correction parameters
# ============================================================

# Original code accumulated -angular_velocity.z, so -1.0 preserves that sign.
IMU_GYRO_SIGN = -1.0

# 1.0 means normal negative feedback. If correction makes drift worse, use -1.0.
IMU_WALK_CORRECTION_SIGN = 1.0

IMU_CALIBRATION_SEC = 1.2
IMU_CALIBRATION_DT = 0.01
IMU_RATE_DEADBAND_RAD = 0.015
IMU_YAW_DEADBAND_DEG = 1.5
IMU_YAW_KP = 0.06
IMU_YAW_MAX_CORRECTION = 0.60
IMU_DT_MAX = 0.20
IMU_PRINT_INTERVAL = 0.50

# ============================================================
# Arrow trigger / approach parameters
# ============================================================

ARROW_TRIGGER_MIN_CONF = 0.75
ARROW_TRIGGER_MIN_Y = 55
ARROW_TRIGGER_MIN_SIZE = 30.0
ARROW_TRIGGER_LOCK_FRAMES = 4
ARROW_IGNORE_SEC = 2.5

ARROW_SAMPLE_FRAMES = 7
ARROW_SAMPLE_DT = 0.05

ARROW_TURN_PAN_KP = 0.8
ARROW_TURN_BODY_KP = 3.5
ARROW_TURN_TH_MIN = 1.8
ARROW_TURN_TH_CLAMP = 5.0
ARROW_TURN_DONE_PAN = 0.22
ARROW_TURN_DONE_FRAMES = 2
ARROW_TURN_LOST_MAX = 3
ARROW_PAN_LIMIT = 1.35

ARROW_APPROACH_CX_TARGET = 331
ARROW_APPROACH_X_START = -0.010
ARROW_APPROACH_X = -0.022
ARROW_APPROACH_NEAR_X = -0.012
ARROW_APPROACH_X_TOL = 35
ARROW_APPROACH_TH_KP = 0.010
ARROW_APPROACH_TH_CLAMP = 0.70
ARROW_APPROACH_NEAR_SIZE = 110.0
ARROW_APPROACH_NEAR_Y = 390
ARROW_APPROACH_STOP_SIZE = 125.0
ARROW_APPROACH_STOP_Y = 440
ARROW_APPROACH_STOP_FRAMES = 3
ARROW_APPROACH_LOST_MAX = 8
ARROW_APPROACH_TIMEOUT = 22.0

ARROW_FINE_CX_TARGET = 352
ARROW_FINE_CY_TARGET = 440
ARROW_FINE_X_TOL = 18
ARROW_FINE_Y_TOL = 14
ARROW_FINE_SAFE_Y = 450
ARROW_FINE_STABLE_FRAMES = 2
ARROW_FINE_LOST_MAX = 3
ARROW_FINE_MAX_ITER = 999
ARROW_FINE_TH_MIN = 1.2
ARROW_FINE_TH_MAX = 1.6
ARROW_FINE_TURN_TIME = 0.45
ARROW_FINE_FORWARD_X = -0.010
ARROW_FINE_FORWARD_TIME = 0.14
ARROW_FINE_BACK_X = 0.015
ARROW_FINE_BACK_TIME = 0.25
ARROW_FINE_SETTLE_TIME = 0.60

# ============================================================
# Turn action parameters
# ============================================================

ACTION_TURN_TH = 3.5
ACTION_TURN_TIME = 15.0
LEFT_TURN_SIGN = 1.0
RIGHT_TURN_SIGN = -1.0

TURN_Z_MOVE_AMPLITUDE = 0.014
TURN_Z_OFFSET = 0.034
TURN_HIP_PITCH = 5.0
TURN_SETTLE_TIME = 1.0
AFTER_ACTION_WAIT = 0.35
BALANCE_ENABLE = True

# ============================================================
# IMU state
# ============================================================

imu_lock = threading.Lock()
imu_raw_z = 0.0
imu_bias_z = 0.0
imu_yaw_deg = 0.0
imu_last_time = None
imu_have_data = False
imu_integrating = False


def imu_callback(data):
    global imu_raw_z, imu_yaw_deg, imu_last_time, imu_have_data

    now = time.time()
    raw_z = float(data.angular_velocity.z)

    with imu_lock:
        imu_raw_z = raw_z
        imu_have_data = True

        if imu_last_time is None:
            imu_last_time = now
            return

        dt = now - imu_last_time
        imu_last_time = now

        if (not imu_integrating) or dt <= 0.0 or dt > IMU_DT_MAX:
            return

        corrected_rate = raw_z - imu_bias_z
        if abs(corrected_rate) < IMU_RATE_DEADBAND_RAD:
            corrected_rate = 0.0

        imu_yaw_deg += IMU_GYRO_SIGN * corrected_rate * dt * 180.0 / math.pi


def calibrate_imu():
    global imu_bias_z, imu_last_time

    samples = []
    start = time.time()

    print("[IMU] calibration start; keep robot still")
    while time.time() - start < IMU_CALIBRATION_SEC and not rospy.is_shutdown():
        with imu_lock:
            if imu_have_data:
                samples.append(float(imu_raw_z))
        rospy.sleep(IMU_CALIBRATION_DT)

    with imu_lock:
        if samples:
            imu_bias_z = float(np.median(samples))
        else:
            imu_bias_z = 0.0
        imu_last_time = time.time()

    print("[IMU] calibration samples=%d bias_z=%.6f rad/s" % (len(samples), imu_bias_z))


def set_imu_integrating(enabled):
    global imu_integrating, imu_last_time
    with imu_lock:
        imu_integrating = bool(enabled)
        imu_last_time = time.time()


def reset_imu_heading(enable=True):
    global imu_yaw_deg, imu_last_time, imu_integrating
    with imu_lock:
        imu_yaw_deg = 0.0
        imu_last_time = time.time()
        imu_integrating = bool(enable)
    print("[IMU] heading reset; integrating=%s" % str(bool(enable)))


def get_imu_yaw():
    with imu_lock:
        return float(imu_yaw_deg)


def get_imu_cruise_correction():
    if not USE_IMU_CRUISE_CORRECTION:
        return 0.0

    yaw_error = get_imu_yaw()
    if abs(yaw_error) <= IMU_YAW_DEADBAND_DEG:
        return 0.0

    correction = -IMU_WALK_CORRECTION_SIGN * IMU_YAW_KP * yaw_error
    return float(np.clip(correction, -IMU_YAW_MAX_CORRECTION, IMU_YAW_MAX_CORRECTION))

# ============================================================
# Detector UDP client
# ============================================================

class DetClient(object):
    def __init__(self, host=DET_BIND_HOST, port=DET_BIND_PORT, stale=DET_STALE_SEC):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.bind((host, port))
        self.s.settimeout(0.2)
        self.stale = stale
        self.latest = None
        self.lock = threading.Lock()

        t = threading.Thread(target=self._loop)
        t.daemon = True
        t.start()

    def _loop(self):
        while not rospy.is_shutdown():
            try:
                data, _ = self.s.recvfrom(8192)
                if not isinstance(data, str):
                    data = data.decode("utf-8")
                packet = json.loads(data)
                with self.lock:
                    self.latest = packet
            except socket.timeout:
                continue
            except Exception as e:
                if DEBUG_MODE:
                    print("[DET] JSON/socket error:", e)

    def _fresh_packet(self):
        with self.lock:
            packet = self.latest

        if packet is None:
            return None

        try:
            if time.time() - float(packet.get("t", 0.0)) > self.stale:
                return None
        except Exception:
            return None

        return packet

    def get_tag(self):
        packet = self._fresh_packet()
        if packet is None:
            return None

        tag = packet.get("tag", {})
        if not bool(tag.get("found", False)):
            return None

        try:
            tag_id = int(tag.get("id"))
            tag_dir = APRILTAG_ID_TO_DIR.get(tag_id, None)
            cx = int(tag.get("cx", 0))
            cy = int(tag.get("cy", 0))
            area = int(tag.get("area", 0))
            size = float(np.sqrt(max(area, 0)))
        except Exception:
            return None

        if tag_dir is None:
            return None

        return {
            "tag_id": tag_id,
            "direction": tag_dir,
            "cx": cx,
            "cy": cy,
            "area": area,
            "size": size,
            "fw": int(packet.get("fw", 640)),
            "fh": int(packet.get("fh", 480)),
        }

    def get_arrow(self):
        packet = self._fresh_packet()
        if packet is None:
            return None

        arrow = packet.get("arrow", {})
        if not bool(arrow.get("found", False)):
            return None

        try:
            name = str(arrow.get("name", "none"))
            direction = YOLO_TO_DIR.get(name, None)
            conf = float(arrow.get("conf", 0.0))
            accepted = bool(arrow.get("accepted", False))
            cx = int(arrow.get("cx", 0))
            cy = int(arrow.get("cy", 0))
            w = float(arrow.get("w", 0.0))
            h = float(arrow.get("h", 0.0))
        except Exception:
            return None

        if direction is None:
            return None

        return {
            "name": name,
            "direction": direction,
            "conf": conf,
            "accepted": accepted,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "size": max(w, h),
            "bottom_y": int(cy + h / 2.0),
            "area": int(w * h),
            "fw": int(packet.get("fw", 640)),
            "fh": int(packet.get("fh", 480)),
        }

# ============================================================
# State machine
# ============================================================

class States:
    INIT = -1
    READY = 0
    TAG_CRUISE = 1
    TAG_DECIDE = 2
    ARROW_TURN = 3
    ARROW_APPROACH = 4
    ARROW_FINE = 5
    EXEC_FORWARD = 6
    EXEC_LEFT = 7
    EXEC_RIGHT = 8
    END = 99

# ============================================================
# Robot helpers
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


def walk_cmd(x=0.0, y=0.0, th_actual=0.0, hip_pitch=10.0, balance=True,
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
    set_imu_integrating(False)

    print("[TURN] th=%.2f duration=%.2f" % (th_actual, duration))
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

    start = time.time()
    while time.time() - start < duration and not rospy.is_shutdown():
        elapsed = time.time() - start
        if elapsed < 0.5:
            gain = 0.45 + 0.55 * elapsed / 0.5
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


def short_walk_move(x_cmd, th_actual, duration):
    set_imu_integrating(False)
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

    start = time.time()
    while time.time() - start < duration and not rospy.is_shutdown():
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
    rospy.sleep(ARROW_FINE_SETTLE_TIME)


def turn_action(direction):
    if direction == "left":
        sign = LEFT_TURN_SIGN
    else:
        sign = RIGHT_TURN_SIGN

    th_cmd = sign * ACTION_TURN_TH
    print("[EXEC_TURN_%s] th=%.2f time=%.2f" % (direction.upper(), th_cmd, ACTION_TURN_TIME))
    turn_to_target_walk(th_cmd, ACTION_TURN_TIME)

# ============================================================
# Detection helpers
# ============================================================

def valid_tag_for_cruise(tag):
    if (not ENABLE_APRILTAG_CRUISE) or tag is None:
        return False
    if abs(int(tag["cx"]) - TAG_CENTER_X) > TAG_CENTER_X_TOL:
        return False
    if int(tag["cy"]) < TAG_TRIGGER_MIN_Y or int(tag["cy"]) > TAG_TRIGGER_MAX_Y:
        return False
    if int(tag["area"]) < TAG_TRIGGER_MIN_AREA:
        return False
    return True


def valid_arrow_for_trigger(arrow):
    if (not ENABLE_ARROW_DETECTION) or arrow is None:
        return False
    if not bool(arrow.get("accepted", False)):
        return False
    if float(arrow.get("conf", 0.0)) < ARROW_TRIGGER_MIN_CONF:
        return False
    if int(arrow.get("cy", 0)) < ARROW_TRIGGER_MIN_Y:
        return False
    if float(arrow.get("size", 0.0)) < ARROW_TRIGGER_MIN_SIZE:
        return False
    return True


def sample_tag_median(n=TAG_SAMPLE_FRAMES, dt=TAG_SAMPLE_DT):
    ids = []
    directions = []
    xs = []
    ys = []
    areas = []

    for _ in range(n):
        tag = det.get_tag()
        if tag is not None:
            ids.append(int(tag["tag_id"]))
            directions.append(tag["direction"])
            xs.append(int(tag["cx"]))
            ys.append(int(tag["cy"]))
            areas.append(int(tag["area"]))
        rospy.sleep(dt)

    if not ids:
        return None

    tag_id = max(set(ids), key=ids.count)
    direction = max(set(directions), key=directions.count)

    return {
        "tag_id": tag_id,
        "direction": direction,
        "cx": int(np.median(xs)),
        "cy": int(np.median(ys)),
        "area": int(np.median(areas)),
    }


def sample_arrow_median(n=ARROW_SAMPLE_FRAMES, dt=ARROW_SAMPLE_DT):
    samples = []
    xs = []
    ys = []
    ws = []
    hs = []
    confs = []
    directions = []

    for _ in range(n):
        arrow = det.get_arrow()
        if arrow is not None and bool(arrow.get("accepted", False)):
            samples.append(arrow)
            xs.append(float(arrow["cx"]))
            ys.append(float(arrow["cy"]))
            ws.append(float(arrow["w"]))
            hs.append(float(arrow["h"]))
            confs.append(float(arrow["conf"]))
            directions.append(arrow["direction"])
        rospy.sleep(dt)

    if not samples:
        return None

    result = dict(samples[-1])
    result["cx"] = int(np.median(xs))
    result["cy"] = int(np.median(ys))
    result["w"] = float(np.median(ws))
    result["h"] = float(np.median(hs))
    result["size"] = max(result["w"], result["h"])
    result["conf"] = float(np.median(confs))
    result["direction"] = max(set(directions), key=directions.count)
    return result

# ============================================================
# Main
# ============================================================

rospy.init_node("marathon_controller_passive_imu")
rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_callback, queue_size=1)

robot = Robot()
rospy.sleep(0.3)

det = DetClient()
rate = rospy.Rate(60)

currState = States.INIT
selected_direction = None
pending_tag = None
pending_arrow = None

cruise_started = False
cruise_start_time = 0.0
marker_ignore_until = 0.0
last_imu_print_time = 0.0

# AprilTag lock state.
tag_lock_count = 0
tag_lock_id = None

# Arrow state.
arrow_lock_count = 0
arrow_turn_pan = 0.0
arrow_turn_done_count = 0
arrow_lost_count = 0
arrow_fine_stable_count = 0
arrow_fine_iter = 0

print("[INFO] Marathon passive-tag + IMU controller started")
print("[INFO] UDP detector: %s:%d" % (DET_BIND_HOST, DET_BIND_PORT))
print("[MODE] low_head=%s tag=%s arrow=%s imu=%s" % (
    str(LOCK_HEAD_LOW_DURING_CRUISE),
    str(ENABLE_APRILTAG_CRUISE),
    str(ENABLE_ARROW_DETECTION),
    str(USE_IMU_CRUISE_CORRECTION)
))

while not rospy.is_shutdown():

    # MODE button always returns to INIT when the main loop is active.
    if robot.buttonCheck("mode"):
        print("[BUTTON] mode -> reset")
        robot.walkStop()
        set_imu_integrating(False)
        currState = States.INIT
        selected_direction = None
        pending_tag = None
        pending_arrow = None
        cruise_started = False
        marker_ignore_until = 0.0
        tag_lock_count = 0
        tag_lock_id = None
        arrow_lock_count = 0
        arrow_turn_done_count = 0
        arrow_lost_count = 0
        arrow_fine_stable_count = 0
        arrow_fine_iter = 0

    # ========================================================
    # INIT
    # ========================================================
    if currState == States.INIT:
        print("[INIT]")
        robot.walkStop()
        set_imu_integrating(False)
        rospy.sleep(0.2)

        robot.setGeneralControlModule("action_module")
        rospy.sleep(0.3)
        robot.playMotion(MOTION_INIT, wait_for_end=True)
        rospy.sleep(0.3)

        robot.setGeneralControlModule("walking_module")
        rospy.sleep(0.3)
        prepare_leg_walking()

        if LOCK_HEAD_LOW_DURING_CRUISE:
            set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.8)
        else:
            set_head_rad(HEAD_PAN_CENTER, HEAD_TILT_READY, wait=0.8)

        calibrate_imu()
        currState = States.READY

    # ========================================================
    # READY
    # ========================================================
    elif currState == States.READY:
        print("[READY] waiting start/user button...")

        if robot.buttonCheck("start") or robot.buttonCheck("user"):
            print("[START]")
            stop_walk(0.2)
            set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.35)

            pending_tag = None
            pending_arrow = None
            marker_ignore_until = 0.0
            tag_lock_count = 0
            tag_lock_id = None
            arrow_lock_count = 0
            cruise_started = False
            reset_imu_heading(enable=True)
            currState = States.TAG_CRUISE

    # ========================================================
    # TAG_CRUISE: fixed low head + continuous IMU straight walk
    # ========================================================
    elif currState == States.TAG_CRUISE:
        now = time.time()

        if not cruise_started:
            robot.setGeneralControlModule("walking_module")
            set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.15)
            robot.walkStart()
            cruise_started = True
            cruise_start_time = now
            print("[TAG_CRUISE] continuous straight walking")

        elapsed = now - cruise_start_time
        if elapsed < CRUISE_START_RAMP_SEC:
            ratio = elapsed / CRUISE_START_RAMP_SEC
            x_cmd = -0.008 + (CRUISE_X + 0.008) * ratio
        else:
            x_cmd = CRUISE_X

        imu_th = get_imu_cruise_correction()
        walk_cmd(
            x=x_cmd,
            y=CRUISE_Y,
            th_actual=imu_th,
            hip_pitch=CRUISE_HIP_PITCH,
            balance=BALANCE_ENABLE,
            z_move_amplitude=CRUISE_Z_MOVE_AMPLITUDE,
            z_offset=CRUISE_Z_OFFSET
        )

        if DEBUG_MODE and now - last_imu_print_time >= IMU_PRINT_INTERVAL:
            print("[CRUISE_IMU] yaw=%.2f deg correction=%.3f" % (get_imu_yaw(), imu_th))
            last_imu_print_time = now

        tag = det.get_tag() if ENABLE_APRILTAG_CRUISE else None
        arrow = det.get_arrow() if ENABLE_ARROW_DETECTION else None

        # Ignore the previously processed sign until the robot has moved away.
        if now < marker_ignore_until:
            tag_lock_count = 0
            tag_lock_id = None
            arrow_lock_count = 0
            rate.sleep()
            continue

        # AprilTag has priority in cruise. It is not used for steering.
        if valid_tag_for_cruise(tag):
            current_id = int(tag["tag_id"])

            if TAG_REQUIRE_SAME_ID and tag_lock_id is not None and current_id != tag_lock_id:
                tag_lock_count = 0

            tag_lock_id = current_id
            tag_lock_count += 1
            arrow_lock_count = 0

            print("[TAG_LOCK] id=%d dir=%s x=%d y=%d area=%d lock=%d/%d" % (
                current_id,
                tag["direction"],
                int(tag["cx"]),
                int(tag["cy"]),
                int(tag["area"]),
                tag_lock_count,
                TAG_TRIGGER_LOCK_FRAMES
            ))

            if tag_lock_count >= TAG_TRIGGER_LOCK_FRAMES:
                pending_tag = dict(tag)
                robot.walkStop()
                cruise_started = False
                rospy.sleep(TAG_STOP_SETTLE_SEC)
                tag_lock_count = 0
                tag_lock_id = None
                arrow_lock_count = 0
                currState = States.TAG_DECIDE
                print("[STAGE] tag_cruise -> tag_decide")

        else:
            tag_lock_count = 0
            tag_lock_id = None

            # Arrow is processed only when there is no visible Tag.
            if tag is None and valid_arrow_for_trigger(arrow):
                arrow_lock_count += 1
                print("[ARROW_LOCK] dir=%s x=%d y=%d size=%.1f conf=%.2f lock=%d/%d" % (
                    arrow["direction"],
                    int(arrow["cx"]),
                    int(arrow["cy"]),
                    float(arrow["size"]),
                    float(arrow["conf"]),
                    arrow_lock_count,
                    ARROW_TRIGGER_LOCK_FRAMES
                ))

                if arrow_lock_count >= ARROW_TRIGGER_LOCK_FRAMES:
                    pending_arrow = dict(arrow)
                    robot.walkStop()
                    cruise_started = False
                    set_imu_integrating(False)
                    rospy.sleep(0.35)

                    if USE_ARROW_APPROACH_FLOW:
                        px_corr = (320.0 - float(arrow["cx"])) * (45.0 / 320.0) * np.pi / 180.0
                        arrow_turn_pan = float(np.clip(px_corr, -ARROW_PAN_LIMIT, ARROW_PAN_LIMIT))
                        arrow_turn_done_count = 0
                        arrow_lost_count = 0
                        set_head_rad(arrow_turn_pan, ARROW_SCAN_HEAD_TILT, wait=0.25)
                        currState = States.ARROW_TURN
                        print("[STAGE] tag_cruise -> arrow_turn pan=%.3f" % arrow_turn_pan)
                    else:
                        selected_direction = arrow["direction"]
                        if selected_direction == "left":
                            currState = States.EXEC_LEFT
                        elif selected_direction == "right":
                            currState = States.EXEC_RIGHT
                        else:
                            currState = States.EXEC_FORWARD

                    arrow_lock_count = 0
            else:
                arrow_lock_count = 0

    # ========================================================
    # TAG_DECIDE
    # ========================================================
    elif currState == States.TAG_DECIDE:
        stop_walk(0.05)
        set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.0)

        tag_result = sample_tag_median()
        if tag_result is None:
            tag_result = pending_tag

        if tag_result is None:
            print("[TAG_DECIDE] tag lost -> continue cruise")
            marker_ignore_until = time.time() + TAG_LOST_AFTER_STOP_SEC
            currState = States.TAG_CRUISE
            continue

        direction = tag_result["direction"]
        print("[TAG_DECIDE] id=%d dir=%s x=%d y=%d area=%d" % (
            int(tag_result["tag_id"]),
            direction,
            int(tag_result["cx"]),
            int(tag_result["cy"]),
            int(tag_result["area"])
        ))

        pending_tag = None
        selected_direction = direction
        marker_ignore_until = time.time() + MARKER_IGNORE_SEC

        if direction == "left":
            currState = States.EXEC_LEFT
        elif direction == "right":
            currState = States.EXEC_RIGHT
        else:
            # Forward Tag: do not turn and do not reset the original heading.
            print("[TAG_FORWARD] keep original heading and continue")
            selected_direction = None
            cruise_started = False
            set_imu_integrating(True)
            currState = States.TAG_CRUISE

    # ========================================================
    # ARROW_TURN
    # ========================================================
    elif currState == States.ARROW_TURN:
        arrow = sample_arrow_median()

        if arrow is None:
            arrow_lost_count += 1
            stop_walk(0.05)
            print("[ARROW_TURN] lost %d/%d" % (arrow_lost_count, ARROW_TURN_LOST_MAX))

            if arrow_lost_count >= ARROW_TURN_LOST_MAX:
                marker_ignore_until = time.time() + 0.6
                set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.2)
                reset_imu_heading(enable=True)
                currState = States.TAG_CRUISE
            continue

        arrow_lost_count = 0
        bx = int(arrow["cx"])
        px_corr = (320.0 - float(bx)) * (45.0 / 320.0) * np.pi / 180.0
        arrow_turn_pan = float(np.clip(arrow_turn_pan + px_corr * ARROW_TURN_PAN_KP,
                                       -ARROW_PAN_LIMIT, ARROW_PAN_LIMIT))
        set_head_rad(arrow_turn_pan, ARROW_SCAN_HEAD_TILT, wait=0.20)

        if abs(arrow_turn_pan) < ARROW_TURN_DONE_PAN:
            arrow_turn_done_count += 1
        else:
            arrow_turn_done_count = 0

        print("[ARROW_TURN] x=%d pan=%.3f done=%d/%d" % (
            bx, arrow_turn_pan, arrow_turn_done_count, ARROW_TURN_DONE_FRAMES
        ))

        if arrow_turn_done_count >= ARROW_TURN_DONE_FRAMES:
            stop_walk(0.05)
            set_head_rad(HEAD_PAN_CENTER, ARROW_APPROACH_HEAD_TILT, wait=0.35)
            currState = States.ARROW_APPROACH
        else:
            raw_th = arrow_turn_pan * ARROW_TURN_BODY_KP
            if abs(raw_th) < ARROW_TURN_TH_MIN:
                raw_th = ARROW_TURN_TH_MIN if raw_th >= 0.0 else -ARROW_TURN_TH_MIN

            th_cmd = float(np.clip(raw_th, -ARROW_TURN_TH_CLAMP, ARROW_TURN_TH_CLAMP))
            pan_abs = abs(arrow_turn_pan)

            if pan_abs >= 0.55:
                turn_time = 6.0
            elif pan_abs >= 0.30:
                turn_time = 5.0
            else:
                turn_time = 2.5

            turn_to_target_walk(th_cmd, turn_time)
            arrow_turn_pan = 0.0
            set_head_rad(0.0, ARROW_SCAN_HEAD_TILT, wait=0.25)

    # ========================================================
    # ARROW_APPROACH
    # ========================================================
    elif currState == States.ARROW_APPROACH:
        robot.setGeneralControlModule("walking_module")
        set_head_rad(HEAD_PAN_CENTER, ARROW_APPROACH_HEAD_TILT, wait=0.25)

        approach_start = time.time()
        approach_lost = 0
        approach_close_count = 0
        approach_near = False
        robot.walkStart()

        while currState == States.ARROW_APPROACH and not rospy.is_shutdown():
            elapsed = time.time() - approach_start

            if elapsed >= ARROW_APPROACH_TIMEOUT:
                robot.walkStop()
                arrow_fine_stable_count = 0
                arrow_fine_iter = 0
                currState = States.ARROW_FINE
                print("[ARROW_APPROACH] timeout -> arrow_fine")
                break

            arrow = det.get_arrow()
            if arrow is None or not bool(arrow.get("accepted", False)):
                approach_lost += 1
                walk_cmd(
                    x=0.0,
                    y=0.0,
                    th_actual=0.0,
                    hip_pitch=CRUISE_HIP_PITCH,
                    balance=True,
                    z_move_amplitude=CRUISE_Z_MOVE_AMPLITUDE,
                    z_offset=CRUISE_Z_OFFSET
                )
                print("[ARROW_APPROACH] lost %d/%d" % (approach_lost, ARROW_APPROACH_LOST_MAX))

                if approach_lost >= ARROW_APPROACH_LOST_MAX:
                    robot.walkStop()
                    marker_ignore_until = time.time() + 0.5
                    set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.2)
                    reset_imu_heading(enable=True)
                    currState = States.TAG_CRUISE
                    break

                rospy.sleep(0.05)
                continue

            approach_lost = 0
            bx = int(arrow["cx"])
            by = int(arrow["cy"])
            size = float(arrow["size"])
            x_err = ARROW_APPROACH_CX_TARGET - bx

            if not approach_near:
                if size >= ARROW_APPROACH_NEAR_SIZE or by >= ARROW_APPROACH_NEAR_Y:
                    approach_close_count += 1
                else:
                    approach_close_count = 0
            else:
                if size >= ARROW_APPROACH_STOP_SIZE or by >= ARROW_APPROACH_STOP_Y:
                    approach_close_count += 1
                else:
                    approach_close_count = 0

            print("[ARROW_APPROACH] x=%d y=%d size=%.1f near=%s close=%d/%d" % (
                bx, by, size, str(approach_near), approach_close_count, ARROW_APPROACH_STOP_FRAMES
            ))

            if (not approach_near) and approach_close_count >= ARROW_APPROACH_STOP_FRAMES:
                approach_near = True
                approach_close_count = 0
                set_head_rad(HEAD_PAN_CENTER, ARROW_NEAR_HEAD_TILT, wait=0.20)
            elif approach_near and approach_close_count >= ARROW_APPROACH_STOP_FRAMES:
                robot.walkStop()
                rospy.sleep(0.4)
                set_head_rad(HEAD_PAN_CENTER, ARROW_FINE_HEAD_TILT, wait=0.45)
                arrow_fine_stable_count = 0
                arrow_fine_iter = 0
                currState = States.ARROW_FINE
                break

            if abs(x_err) > ARROW_APPROACH_X_TOL:
                th_actual = float(np.clip(
                    ARROW_APPROACH_TH_KP * float(x_err),
                    -ARROW_APPROACH_TH_CLAMP,
                    ARROW_APPROACH_TH_CLAMP
                ))
            else:
                th_actual = 0.0

            if elapsed < 0.8:
                ratio = elapsed / 0.8
                x_cmd = ARROW_APPROACH_X_START + (ARROW_APPROACH_X - ARROW_APPROACH_X_START) * ratio
            else:
                x_cmd = ARROW_APPROACH_X

            if approach_near:
                x_cmd = ARROW_APPROACH_NEAR_X
            if abs(x_err) > 70:
                x_cmd *= 0.45

            walk_cmd(
                x=x_cmd,
                y=0.0,
                th_actual=th_actual,
                hip_pitch=CRUISE_HIP_PITCH,
                balance=True,
                z_move_amplitude=CRUISE_Z_MOVE_AMPLITUDE,
                z_offset=CRUISE_Z_OFFSET
            )
            rospy.sleep(0.05)

    # ========================================================
    # ARROW_FINE
    # ========================================================
    elif currState == States.ARROW_FINE:
        arrow_fine_iter += 1
        set_head_rad(HEAD_PAN_CENTER, ARROW_FINE_HEAD_TILT, wait=0.0)
        arrow = sample_arrow_median()

        if arrow is None:
            arrow_lost_count += 1
            stop_walk(0.05)
            print("[ARROW_FINE] lost %d/%d" % (arrow_lost_count, ARROW_FINE_LOST_MAX))

            if arrow_lost_count >= ARROW_FINE_LOST_MAX:
                marker_ignore_until = time.time() + 0.5
                set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.2)
                reset_imu_heading(enable=True)
                currState = States.TAG_CRUISE
            continue

        arrow_lost_count = 0
        bx = int(arrow["cx"])
        by = int(arrow["cy"])
        x_err = ARROW_FINE_CX_TARGET - bx
        y_err = ARROW_FINE_CY_TARGET - by
        cx_ok = abs(x_err) <= ARROW_FINE_X_TOL
        cy_ok = abs(y_err) <= ARROW_FINE_Y_TOL

        print("[ARROW_FINE] iter=%d dir=%s x=%d y=%d ex=%d ey=%d stable=%d/%d" % (
            arrow_fine_iter,
            arrow["direction"],
            bx,
            by,
            x_err,
            y_err,
            arrow_fine_stable_count,
            ARROW_FINE_STABLE_FRAMES
        ))

        # Close enough or already near the bottom of the frame: decide immediately.
        if by >= ARROW_FINE_SAFE_Y:
            selected_direction = arrow["direction"]
            stop_walk(0.05)
        elif not cx_ok:
            arrow_fine_stable_count = 0
            cx_pan_error = float(x_err) * (45.0 / 320.0) * np.pi / 180.0
            raw_th = cx_pan_error * ARROW_TURN_BODY_KP
            if abs(raw_th) < ARROW_FINE_TH_MIN:
                raw_th = ARROW_FINE_TH_MIN if raw_th >= 0.0 else -ARROW_FINE_TH_MIN
            th_cmd = float(np.clip(raw_th, -ARROW_FINE_TH_MAX, ARROW_FINE_TH_MAX))
            short_walk_move(0.0, th_cmd, ARROW_FINE_TURN_TIME)
            continue
        elif not cy_ok:
            arrow_fine_stable_count = 0
            if by < ARROW_FINE_CY_TARGET:
                short_walk_move(ARROW_FINE_FORWARD_X, 0.0, ARROW_FINE_FORWARD_TIME)
            else:
                short_walk_move(ARROW_FINE_BACK_X, 0.0, ARROW_FINE_BACK_TIME)
            continue
        else:
            arrow_fine_stable_count += 1
            stop_walk(0.05)
            if arrow_fine_stable_count >= ARROW_FINE_STABLE_FRAMES:
                selected_direction = arrow["direction"]

        if selected_direction is not None:
            print("[ARROW_DECIDE] direction=%s" % selected_direction)
            pending_arrow = None
            marker_ignore_until = time.time() + ARROW_IGNORE_SEC

            if selected_direction == "left":
                currState = States.EXEC_LEFT
            elif selected_direction == "right":
                currState = States.EXEC_RIGHT
            else:
                currState = States.EXEC_FORWARD

        if arrow_fine_iter > ARROW_FINE_MAX_ITER:
            print("[ARROW_FINE] max iter -> resume cruise")
            selected_direction = None
            marker_ignore_until = time.time() + 0.5
            set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.2)
            reset_imu_heading(enable=True)
            currState = States.TAG_CRUISE

    # ========================================================
    # EXEC_FORWARD
    # ========================================================
    elif currState == States.EXEC_FORWARD:
        print("[EXEC_FORWARD] continue straight")
        selected_direction = None
        marker_ignore_until = time.time() + MARKER_IGNORE_SEC
        cruise_started = False
        set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.2)

        # Arrow alignment may have changed the body heading, so use it as the new reference.
        reset_imu_heading(enable=True)
        currState = States.TAG_CRUISE

    # ========================================================
    # EXEC_LEFT
    # ========================================================
    elif currState == States.EXEC_LEFT:
        print("[EXEC_LEFT]")
        turn_action("left")
        selected_direction = None
        marker_ignore_until = time.time() + MARKER_IGNORE_SEC
        cruise_started = False
        set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.2)

        # After a 90-degree turn, the new direction becomes yaw = 0.
        reset_imu_heading(enable=True)
        currState = States.TAG_CRUISE

    # ========================================================
    # EXEC_RIGHT
    # ========================================================
    elif currState == States.EXEC_RIGHT:
        print("[EXEC_RIGHT]")
        turn_action("right")
        selected_direction = None
        marker_ignore_until = time.time() + MARKER_IGNORE_SEC
        cruise_started = False
        set_head_rad(CRUISE_HEAD_PAN, CRUISE_HEAD_TILT, wait=0.2)

        # After a 90-degree turn, the new direction becomes yaw = 0.
        reset_imu_heading(enable=True)
        currState = States.TAG_CRUISE

    # ========================================================
    # END
    # ========================================================
    elif currState == States.END:
        robot.walkStop()
        set_imu_integrating(False)
        rospy.signal_shutdown("END")

    rate.sleep()