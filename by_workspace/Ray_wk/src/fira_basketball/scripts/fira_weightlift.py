#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Refactored FIRA Sprint controller for OP3.
- Keeps original behavior but fixes crashes, globals, and magic numbers.
- Python 2.7 + ROS1 friendly.
- Key fixes:
  * Robust camera/IMU handling and state loop structure
  * OpenCV 2/3/4 compatible contour API
  * Eliminate NameError/UnboundLocalError (e.g., detect_touch, comb_red, cX usage)
  * Safer head PD and clipping
  * Clear HSV presets + utility functions
  * Centralized Robot/State control with readable transitions

Usage:
  rosrun <your_pkg> fira_sprint_refactor.py
"""
from __future__ import division
import rospy
import numpy as np
import cv2
import time
import math

from sensor_msgs.msg import Imu

# External project deps (keep names as in your project)
from op3_ros_utils import Robot
# If you still need getWalkingParams: from op3_ros_utils import getWalkingParams

###############################################################################
# Configuration
###############################################################################
DEBUG_MODE = False
DEGREE2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# Camera index
CAM_INDEX = 0

# Head PD (screen-centered tracking)
HEAD_KP = 20.0
HEAD_KD = 0.0
PAN_LIMIT_DEG = 95.0
TILT_LIMIT_DEG = 95.0

# Alignment acceptance (for Adjust_Mode_V2 → CHECK_SLOPE_V2)
ALIGN_TILT_TARGET_DEG = -78.0  # original code checked 78 + tilt(deg) ≈ 0
ALIGN_TILT_TOL_DEG = 7.5
ALIGN_PAN_TOL_DEG = 5.0
ALIGN_YAW_TOL_DEG = 10.0

# Walking params
WALK_X_BASE = 0.08
WALK_Y_PAN_GAIN = 0.43  # scales pan(deg) to lateral velocity
WALK_TH_YAW_GAIN = 2.0  # scales z_gyro (deg) to yaw cmd
WALK_Y_LIMIT = 3.0
WALK_TH_LIMIT = 3.0

# Final segments
WALK_WITH_BAR_DURATION = 9.5  # seconds
WALK_2_FINISH_DURATION = 50.0

# HSV presets (BGR→HSV space)
# Yellow (original main target range provided by user comments)
HSV_Y_LOW = np.array([0, 69, 128], dtype=np.uint8)
HSV_Y_HIGH = np.array([10, 255, 255], dtype=np.uint8)

# Alternative ranges
HSV_G_LOW = np.array([68, 57, 48], dtype=np.uint8)
HSV_G_HIGH = np.array([96, 255, 255], dtype=np.uint8)

HSV_W_LOW = np.array([0, 0, 190], dtype=np.uint8)
HSV_W_HIGH = np.array([180, 255, 255], dtype=np.uint8)

# Red split (wrap-around)
HSV_R1_LOW = np.array([0, 69, 90], dtype=np.uint8)
HSV_R1_HIGH = np.array([5, 255, 255], dtype=np.uint8)
HSV_R2_LOW = np.array([160, 69, 90], dtype=np.uint8)
HSV_R2_HIGH = np.array([179, 255, 255], dtype=np.uint8)

# Contour filtering
MIN_AREA = 300
HORIZONTAL_RATIO = 2.0  # width > ratio * height → horizontal

###############################################################################
# States
###############################################################################
class States(object):
    INIT = -1
    READY = 0
    WALK_FORWARD = 1
    PICK_BAR = 5
    WALK_WITH_BAR = 6
    LIFT_BAR = 7
    WALK_2_FINISH = 8
    CHECK_SLOPE = 9
    CHECK_PAN = 10
    Adjust_Mode = 11
    CHECK_TOUCH = 12
    Adjust_Mode_V2 = 13
    CHECK_SLOPE_V2 = 14
    Buffer = 20
    END = 99

###############################################################################
# Utilities
###############################################################################

def clip_deg(v, lo, hi):
    return max(lo, min(hi, v))


def cv_find_contours(mask):
    """OpenCV 2/3/4 compatible findContours."""
    out = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(out) == 3:
        # OpenCV 3: image, contours, hierarchy
        _, contours, _ = out
    else:
        # OpenCV 4: contours, hierarchy
        contours, _ = out
    return contours


def compute_angle(width, height, angle):
    # Match original logic for minAreaRect angle normalization
    if angle < -90 or (width > height and angle < 0):
        return 90 + angle
    if width < height and angle > 0:
        return (90 - angle) * -1
    return angle

###############################################################################
# Perception
###############################################################################
class Perception(object):
    def __init__(self, cam_index):
        self.cap = cv2.VideoCapture(cam_index)
        self.frame = None
        self.last_cx = None
        self.last_cy = None
        self.last_angle = None
        if not self.cap.isOpened():
            rospy.logwarn("Camera index %d failed to open." % cam_index)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.frame = frame
        return frame

    def _centroid_and_angle_from_contour(self, frame, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None, None, None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        rect = cv2.minAreaRect(contour)
        (x, y), (w, h), a = rect
        ang = compute_angle(w, h, a)
        return cx, cy, ang

    def detect_red_bar(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, HSV_R1_LOW, HSV_R1_HIGH)
        mask2 = cv2.inRange(hsv, HSV_R2_LOW, HSV_R2_HIGH)
        mask = cv2.add(mask1, mask2)
        contours = cv_find_contours(mask)
        # Filter by horizontal shape and min area
        best = None
        best_area = 0
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue
            (x, y), (w, h), a = cv2.minAreaRect(c)
            if w > h * HORIZONTAL_RATIO:
                if area > best_area:
                    best = c
                    best_area = area
        if best is None:
            return None, None, None
        cx, cy, ang = self._centroid_and_angle_from_contour(frame, best)
        if cx is None:
            return None, None, None
        # Draw for debugging
        if DEBUG_MODE:
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.drawContours(frame, [best], -1, (0, 255, 0), 2)
        self.last_cx, self.last_cy, self.last_angle = cx, cy, ang
        return cx, cy, ang

    def detect_middle(self, frame, cx):
        h, w = frame.shape[:2]
        mid_x = w // 2
        detect_touch = False
        orientation = 2  # 0:right, 1:left, 2:center
        if cx is None:
            return False, orientation
        # draw guide for debug
        if DEBUG_MODE:
            cv2.line(frame, (int(cx), 0), (int(cx), h), (255, 0, 0), 2)
        if (mid_x - 4) <= cx <= (mid_x + 4):
            detect_touch = True
        elif cx > mid_x + 4:
            orientation = 0  # on the right
        else:
            orientation = 1  # on the left
        return detect_touch, orientation

###############################################################################
# Controller
###############################################################################
class Controller(object):
    def __init__(self):
        self.robot = Robot()
        self.z_gyro = 0.0
        self.z_gyro_offset = 0.0
        self.z_bias = 0.0
        self.last_px = 0.0
        self.last_py = 0.0
        self.last_pd_t = time.time()
        rospy.Subscriber("/robotis/open_cr/imu", Imu, self._imu_cb, queue_size=1)

    # IMU integration (mirror original logic)
    def _imu_cb(self, data):
        ang_v = data.angular_velocity
        # original: z_gyro = z_gyro - (angular_velocity.z / 2) + z_gyro_offset_for_caculate
        self.z_gyro = self.z_gyro - (ang_v.z / 2.0) + self.z_bias
        self.z_gyro_offset = (ang_v.z / 2.0)

    def reset_yaw_integration(self):
        self.z_bias = self.z_gyro_offset
        self.z_gyro = 0.0

    def init_pose(self):
        # action module to reset pose
        self.robot.setGeneralControlModule("action_module")
        self.robot.playMotion(1, wait_for_end=True)
        # walking legs; head → none (direct control)
        self.robot.setGeneralControlModule("walking_module")
        self.robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        # neutral head
        self.robot.setJointPos(["head_tilt", "head_pan"], [-0.4, 0.0])
        rospy.sleep(1.0)

    def head_track(self, cx, cy, fw, fh):
        # normalized screen coords relative to center
        px = -1.0 * ((cx / float(fw)) - 0.5)
        py = -1.0 * ((cy / float(fh)) - 0.5)
        pan_old = self.robot.joint_pos["head_pan"] * RAD2DEG
        tilt_old = self.robot.joint_pos["head_tilt"] * RAD2DEG
        now = time.time()
        dt = max(1e-3, now - self.last_pd_t)
        dx = (px - self.last_px) / dt
        dy = (py - self.last_py) / dt
        self.last_px, self.last_py, self.last_pd_t = px, py, now
        pan = pan_old + (px * HEAD_KP + dx * HEAD_KD)
        tilt = tilt_old + (py * HEAD_KP + dy * HEAD_KD)
        pan = clip_deg(pan, -PAN_LIMIT_DEG, PAN_LIMIT_DEG)
        tilt = clip_deg(tilt, -TILT_LIMIT_DEG, TILT_LIMIT_DEG)
        self.robot.setJointPos(["head_pan", "head_tilt"], [pan * DEGREE2RAD, tilt * DEGREE2RAD])
        return pan, tilt

    def aligned_for_pick(self, pan_deg, tilt_deg):
        tilt_ok = abs(ALIGN_TILT_TARGET_DEG - tilt_deg) < ALIGN_TILT_TOL_DEG
        pan_ok = abs(pan_deg) < ALIGN_PAN_TOL_DEG
        yaw_ok = abs(self.z_gyro) < ALIGN_YAW_TOL_DEG
        return tilt_ok and pan_ok and yaw_ok

    def walk_adjust(self):
        # start and adjust
        self.robot.walkStart()
        pan_deg = self.robot.joint_pos["head_pan"] * RAD2DEG
        y_cmd = clip_deg(pan_deg, -WALK_Y_LIMIT, WALK_Y_LIMIT) * WALK_Y_PAN_GAIN
        th_cmd = clip_deg(self.z_gyro, -WALK_TH_LIMIT, WALK_TH_LIMIT) * WALK_TH_YAW_GAIN
        self.robot.walkVelocities(x=WALK_X_BASE, y=y_cmd, th=th_cmd, balance=True)
        if DEBUG_MODE:
            rospy.loginfo("walk_adjust: tilt=%.2f pan=%.2f yaw=%.2f" % (
                self.robot.joint_pos["head_tilt"] * RAD2DEG,
                pan_deg,
                self.z_gyro,
            ))

    def walk_forward_constant(self, x=5.0, dur=2.0, hip_pitch=3.0):
        self.robot.walkStart()
        t0 = time.time()
        while time.time() - t0 < dur and not rospy.is_shutdown():
            self.robot.walkVelocities(x=x, y=0.0,
                                      th=np.clip(self.z_gyro, -10.0, 10.0),
                                      z_move_amplitude=0.045,
                                      balance=True,
                                      z_offset=0.0,
                                      hip_pitch=hip_pitch)
            rospy.sleep(0.01)
        self.robot.walkStop()

###############################################################################
# Node
###############################################################################
class SprintNode(object):
    def __init__(self):
        rospy.init_node("fira_sprint")
        self.perc = Perception(CAM_INDEX)
        self.ctrl = Controller()
        self.state = States.INIT
        self.tickrate = 60
        self.rate = rospy.Rate(self.tickrate)
        self._count = 0
        self.last_pan = 0.0
        self.last_tilt = 0.0

    def _frame_dims(self, cap):
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return fw, fh

    def run(self):
        while not rospy.is_shutdown():
            frame = self.perc.read()
            if frame is not None:
                fw, fh = self._frame_dims(self.perc.cap)
                cx, cy, ang = self.perc.detect_red_bar(frame)
                if cx is not None and cy is not None:
                    self.last_pan, self.last_tilt = self.ctrl.head_track(cx, cy, fw, fh)
                if DEBUG_MODE:
                    cv2.imshow('frame', frame)
                    cv2.waitKey(1)

            if self.ctrl.robot.buttonCheck("mode"):
                self.state = States.INIT
                self._count = 0

            # State machine
            if self.state == States.INIT:
                rospy.loginfo("[INIT]")
                self.ctrl.init_pose()
                self.state = States.READY
                self._count = 0

            elif self.state == States.READY:
                rospy.loginfo("[READY]")
                if self.ctrl.robot.buttonCheck("start"):
                    rospy.sleep(5.0)  # for video taping
                    self.ctrl.reset_yaw_integration()
                    self.ctrl.robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
                    self.ctrl.robot.setJointPos(["head_pan", "head_tilt"], [0.0, -0.4])
                    self._count = 0
                    self.ctrl.robot.walkStart()
                    self.state = States.Adjust_Mode_V2

            elif self.state == States.Adjust_Mode_V2:
                self._count += 1
                self.ctrl.walk_adjust()
                # Consider aligned when close to target tilt/pan/yaw
                if self.ctrl.aligned_for_pick(self.last_pan, self.last_tilt):
                    self.ctrl.robot.walkStop()
                    self._count = 0
                    self.state = States.CHECK_SLOPE_V2
                    rospy.sleep(1.0)

            elif self.state == States.CHECK_SLOPE_V2:
                self.ctrl.robot.walkStop()
                if self._count < 60:
                    self._count += 1
                else:
                    # Original code proceeded to PICK_BAR unconditionally when not aligned too
                    self.state = States.PICK_BAR
                    self._count = 0

            elif self.state == States.PICK_BAR:
                rospy.loginfo("[PICK_BAR]")
                rospy.sleep(2.0)
                self.ctrl.robot.setGeneralControlModule("none")
                rospy.sleep(2.0)
                self.ctrl.robot.setGeneralControlModule("action_module")
                # Prepare & pick (IDs from original script)
                self.ctrl.robot.playMotion(192, wait_for_end=True)
                time.sleep(2.0)
                self.ctrl.reset_yaw_integration()
                rospy.sleep(1.0)
                # transition
                self.state = States.WALK_WITH_BAR

                # Start walking straight a bit (moved from original inline loop)
                self.ctrl.robot.setJointsControlModule([
                    "r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll",
                    "r_hip_pitch","l_hip_pitch","r_knee","l_knee",
                    "r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],
                    ["walking_module"])
                self.ctrl.walk_forward_constant(x=5.0, dur=2.0, hip_pitch=3.0)

            elif self.state == States.WALK_WITH_BAR:
                rospy.loginfo("[WALK_WITH_BAR]")
                rospy.sleep(1.0)
                self.ctrl.robot.setJointsControlModule([
                    "r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll",
                    "r_hip_pitch","l_hip_pitch","r_knee","l_knee",
                    "r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],
                    ["walking_module"]) 
                self.ctrl.robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
                self.ctrl.robot.setJointPos(["head_tilt", "head_pan"], [-1.5, 0.0])
                rospy.sleep(1.0)

                # Constant forward for a fixed time window (original 9.5s)
                t0 = time.time()
                self.ctrl.robot.walkStart()
                while time.time() - t0 < WALK_WITH_BAR_DURATION and not rospy.is_shutdown():
                    self.ctrl.robot.walkVelocities(
                        x=5.0, y=0.0,
                        th=np.clip(self.ctrl.z_gyro, -10.0, 10.0),
                        z_move_amplitude=0.045,
                        balance=True,
                        z_offset=0.0,
                        hip_pitch=5.0)
                    rospy.sleep(0.01)
                self.ctrl.robot.walkStop()
                self.state = States.LIFT_BAR
                rospy.sleep(1.0)

            elif self.state == States.LIFT_BAR:
                rospy.loginfo("[LIFT_BAR]")
                self.ctrl.robot.setGeneralControlModule("none")
                rospy.sleep(2.0)
                self.ctrl.robot.setGeneralControlModule("action_module")
                # Lift IDs (kept as in original)
                self.ctrl.robot.playMotion(191, wait_for_end=True)
                time.sleep(2.0)
                rospy.sleep(1.0)
                self.state = States.WALK_2_FINISH

            elif self.state == States.WALK_2_FINISH:
                rospy.loginfo("[WALK_2_FINISH]")
                self.ctrl.robot.setJointsControlModule([
                    "r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll",
                    "r_hip_pitch","l_hip_pitch","r_knee","l_knee",
                    "r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],
                    ["walking_module"]) 
                t0 = time.time()
                self.ctrl.robot.walkStart()
                while time.time() - t0 < WALK_2_FINISH_DURATION and not rospy.is_shutdown():
                    self.ctrl.robot.walkVelocities(
                        x=5.0, y=0.0,
                        th=np.clip(self.ctrl.z_gyro, -10.0, 10.0),
                        z_move_amplitude=0.045,
                        balance=True,
                        z_offset=0.0,
                        hip_pitch=7.0)
                    rospy.sleep(0.01)
                self.ctrl.robot.walkStop()
                self.state = States.END

            elif self.state == States.END:
                rospy.loginfo("[END]")
                # Idle; keep node alive

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = SprintNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
