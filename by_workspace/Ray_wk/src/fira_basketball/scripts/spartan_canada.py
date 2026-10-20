#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import rospy
import cv2
import numpy as np
import time
import math

from op3_ros_utils import Robot

# ============================================================
#  Blue range + blue boundary slope walking demo
#  For ROBOTIS OP3 / ROS Kinetic / Python2.7
#  Put this file in Ray_wk package scripts folder and run with python.
# ============================================================

# ----------------------------
# Camera / display
# ----------------------------
CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480
SHOW_IMAGE = True       # If SSH/X11 has no display, change to False.
USE_TRACKBAR = False    # Competition tuning: True opens HSV sliders.
DISPLAY_DELAY_MS = 1

# ----------------------------
# Blue HSV range
# OpenCV HSV: H 0~179, S/V 0~255
# Adjust these values on site if blue mask is not clean.
# ----------------------------
BLUE_H_MIN = 90
BLUE_S_MIN = 60
BLUE_V_MIN = 40
BLUE_H_MAX = 130
BLUE_S_MAX = 255
BLUE_V_MAX = 255

# ----------------------------
# Blue geometry detection
# ----------------------------
ROI_TOP_RATIO = 0.25
ROI_BOTTOM_RATIO = 1.00
MIN_BLUE_AREA = 800
BOUNDARY_MODE = "upper"     # "upper" or "lower" blue boundary slope.
BOUNDARY_STEP_X = 4
BLUE_CENTER_TOL = 35
BLUE_SLOPE_TOL_DEG = 5.0
TARGET_CX_OFFSET = 0         # Positive means target is right of image center.

# Stop condition. Camera cannot measure 5 cm directly without calibration,
# so BLUE_STOP_BOTTOM_Y is the image-space stop threshold.
# Put the robot about 5 cm from the blue area, check the printed bottom_y,
# then replace this value if needed. For 640x480, 430~455 is usually near.
BLUE_STOP_BOTTOM_Y = 440
BLUE_STOP_ANGLE_TOL_DEG = 3.0
REQUIRE_CENTER_TO_STOP = False

# ----------------------------
# Head / motion
# ----------------------------
HEAD_TILT_FINE = -0.9
HEAD_PAN_CENTER = 0.0
MOTION_INIT = 1

# ============================================================
#  Walking parameters copied from weightlifting Adjust_Mode
# ============================================================
FINE_CY_TARGET      = 450
FINE_CX_TOL         = 15
FINE_SAMPLE_FRAMES  = 7

FINE_X_STEP         = 0.04
FINE_Y_KP           = 0.006
FINE_Y_CLAMP        = 0.18

FINE_TH_KP          = 0.10
FINE_TH_CLAMP       = 3.0
FINE_TH_SIGN        = -1.0
FINE_TH_BIAS        = 1.0

FINE_TURN_X         = -1.0
FINE_TURN_Y         = 0.8
FINE_TURN_TH        = 3.5
FINE_TURN_TIME      = 3.0
FINE_ACCEPT_CY      = 445
FINE_MIN_PICK_CY    = 442
FINE_ACCEPT_ANG     = 4.5

FINE_STEP_TIME      = 0.6
FINE_LOST_MAX       = 2
FINE_MAX_ITER       = 24

ANGLE_THRESHOLD         = 2.0
ANGLE_VALID_MAX         = 35.0
ANGLE_MIN_VALID_FRAMES  = 4
ANGLE_SIGN_RATIO        = 0.6

# This script uses the Adjust_Mode walking style but detects blue area/edge instead of bar.
# For safer field test, these two values are separated from the copied constants.
BLUE_FORWARD_X = FINE_X_STEP
BLUE_STEP_TIME = FINE_STEP_TIME
BLUE_Z_AMP = 0.018
BLUE_TURN_TIME_SCALE = 1.00   # 1.00 = fully copy weightlifting Adjust_Mode turn time.

# ============================================================
#  ROS / Robot
# ============================================================
rospy.init_node("blue_adjust_slope_raywk")
robot = Robot()
rospy.sleep(0.5)
rate = rospy.Rate(60)


def set_head(tilt_rad, pan_rad=0.0, wait=0.25):
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_tilt", "head_pan"], [tilt_rad, pan_rad])
    rospy.sleep(wait)


def init_robot_pose():
    print("[INIT] action init -> walking module")
    robot.setGeneralControlModule("action_module")
    rospy.sleep(0.2)
    robot.playMotion(MOTION_INIT, wait_for_end=True)
    rospy.sleep(0.2)
    robot.setGeneralControlModule("walking_module")
    rospy.sleep(0.2)
    set_head(HEAD_TILT_FINE, HEAD_PAN_CENTER, wait=0.5)


def nothing(x):
    pass


def setup_trackbars():
    cv2.namedWindow("BlueAdjust", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("H min", "BlueAdjust", BLUE_H_MIN, 179, nothing)
    cv2.createTrackbar("S min", "BlueAdjust", BLUE_S_MIN, 255, nothing)
    cv2.createTrackbar("V min", "BlueAdjust", BLUE_V_MIN, 255, nothing)
    cv2.createTrackbar("H max", "BlueAdjust", BLUE_H_MAX, 179, nothing)
    cv2.createTrackbar("S max", "BlueAdjust", BLUE_S_MAX, 255, nothing)
    cv2.createTrackbar("V max", "BlueAdjust", BLUE_V_MAX, 255, nothing)


def get_hsv_range():
    if USE_TRACKBAR and SHOW_IMAGE:
        hmin = cv2.getTrackbarPos("H min", "BlueAdjust")
        smin = cv2.getTrackbarPos("S min", "BlueAdjust")
        vmin = cv2.getTrackbarPos("V min", "BlueAdjust")
        hmax = cv2.getTrackbarPos("H max", "BlueAdjust")
        smax = cv2.getTrackbarPos("S max", "BlueAdjust")
        vmax = cv2.getTrackbarPos("V max", "BlueAdjust")
        return np.array([hmin, smin, vmin]), np.array([hmax, smax, vmax])
    return np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN]), np.array([BLUE_H_MAX, BLUE_S_MAX, BLUE_V_MAX])


def find_contours_compat(mask):
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 3:
        return result[1]
    return result[0]


def fit_boundary_slope(mask, contour, mode):
    h, w = mask.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    pts_x = []
    pts_y = []

    x0 = max(0, x)
    x1 = min(w - 1, x + bw)
    y0 = max(0, y)
    y1 = min(h - 1, y + bh)

    for col in range(x0, x1, BOUNDARY_STEP_X):
        col_pixels = np.where(mask[y0:y1, col] > 0)[0]
        if len(col_pixels) == 0:
            continue
        if mode == "lower":
            by = y0 + int(col_pixels[-1])
        else:
            by = y0 + int(col_pixels[0])
        pts_x.append(col)
        pts_y.append(by)

    if len(pts_x) >= 10:
        slope, intercept = np.polyfit(np.array(pts_x), np.array(pts_y), 1)
        angle = math.degrees(math.atan(float(slope)))
        p1 = (int(x0), int(slope * x0 + intercept))
        p2 = (int(x1), int(slope * x1 + intercept))
        return float(slope), float(angle), p1, p2, list(zip(pts_x, pts_y))

    pts = contour.reshape(-1, 2)
    if len(pts) >= 10:
        vx, vy, x_fit, y_fit = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01)
        vx = float(vx[0]); vy = float(vy[0]); x_fit = float(x_fit[0]); y_fit = float(y_fit[0])
        if abs(vx) < 1e-6:
            slope = 999.0
            angle = 90.0
        else:
            slope = vy / vx
            angle = math.degrees(math.atan(slope))
        p1 = (int(x_fit - 120 * vx), int(y_fit - 120 * vy))
        p2 = (int(x_fit + 120 * vx), int(y_fit + 120 * vy))
        return float(slope), float(angle), p1, p2, []

    return None, None, None, None, []


def detect_blue(frame):
    hsv_low, hsv_high = get_hsv_range()
    h, w = frame.shape[:2]
    roi_y0 = int(h * ROI_TOP_RATIO)
    roi_y1 = int(h * ROI_BOTTOM_RATIO)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    # ROI mask
    mask[:roi_y0, :] = 0
    mask[roi_y1:, :] = 0

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours = find_contours_compat(mask)
    if len(contours) == 0:
        return None, mask

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < MIN_BLUE_AREA:
        return None, mask

    M = cv2.moments(contour)
    if abs(M["m00"]) < 1e-6:
        return None, mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    slope, angle, p1, p2, boundary_pts = fit_boundary_slope(mask, contour, BOUNDARY_MODE)

    x, y, bw, bh = cv2.boundingRect(contour)
    bottom_y = int(y + bh)
    info = {
        "cx": cx,
        "cy": cy,
        "bottom_y": bottom_y,
        "area": area,
        "slope": slope,
        "angle": angle,
        "contour": contour,
        "bbox": (x, y, bw, bh),
        "line_p1": p1,
        "line_p2": p2,
        "boundary_pts": boundary_pts,
        "fw": w,
        "fh": h,
        "roi_y0": roi_y0,
        "roi_y1": roi_y1
    }
    return info, mask


def draw_debug(frame, info, mask):
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.line(vis, (0, int(h * ROI_TOP_RATIO)), (w, int(h * ROI_TOP_RATIO)), (255, 255, 0), 1)
    cv2.line(vis, (w // 2 + TARGET_CX_OFFSET, 0), (w // 2 + TARGET_CX_OFFSET, h), (0, 255, 255), 1)

    if info is not None:
        cv2.drawContours(vis, [info["contour"]], -1, (255, 0, 0), 2)
        x, y, bw, bh = info["bbox"]
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.circle(vis, (info["cx"], info["cy"]), 5, (0, 0, 255), -1)
        if info["line_p1"] is not None and info["line_p2"] is not None:
            cv2.line(vis, info["line_p1"], info["line_p2"], (0, 0, 255), 2)
        for p in info["boundary_pts"]:
            cv2.circle(vis, p, 1, (0, 255, 255), -1)
        text = "cx={} cy={} bottom={} area={:.0f} slope={} ang={}".format(
            info["cx"], info["cy"], info.get("bottom_y", 0), info["area"],
            "None" if info["slope"] is None else "{:.3f}".format(info["slope"]),
            "None" if info["angle"] is None else "{:.1f}".format(info["angle"])
        )
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    else:
        cv2.putText(vis, "No blue found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    small_mask = cv2.resize(mask_bgr, (w // 3, h // 3))
    vis[0:h // 3, 0:w // 3] = small_mask
    return vis


def sample_blue_stable(cap, n=FINE_SAMPLE_FRAMES, dt=0.04):
    infos = []
    last_frame = None
    last_mask = None

    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            rospy.sleep(dt)
            continue
        last_frame = frame
        info, mask = detect_blue(frame)
        last_mask = mask
        if info is not None and info["angle"] is not None and abs(info["angle"]) <= ANGLE_VALID_MAX:
            infos.append(info)
        elif info is not None:
            infos.append(info)
        rospy.sleep(dt)

    if len(infos) == 0:
        return None, last_frame, last_mask

    cxs = [i["cx"] for i in infos]
    cys = [i["cy"] for i in infos]
    bottoms = [i.get("bottom_y", i["cy"]) for i in infos]
    areas = [i["area"] for i in infos]
    angles = [i["angle"] for i in infos if i["angle"] is not None and abs(i["angle"]) <= ANGLE_VALID_MAX]

    info_med = infos[-1]
    info_med["cx"] = int(np.median(cxs))
    info_med["cy"] = int(np.median(cys))
    info_med["bottom_y"] = int(np.median(bottoms))
    info_med["area"] = float(np.median(areas))
    if len(angles) >= ANGLE_MIN_VALID_FRAMES:
        info_med["angle"] = float(np.median(angles))
        info_med["angle_stable"] = True
    elif len(angles) > 0:
        info_med["angle"] = float(np.median(angles))
        info_med["angle_stable"] = False
    else:
        info_med["angle"] = 0.0
        info_med["angle_stable"] = False
    return info_med, last_frame, last_mask


def walk_step(x_cmd, y_cmd, th_cmd, step_time, z_amp=BLUE_Z_AMP):
    robot.walkStart()
    t0 = time.time()
    while time.time() - t0 < step_time and not rospy.is_shutdown():
        robot.walkVelocities(x=x_cmd, y=y_cmd, th=th_cmd, z_move_amplitude=z_amp, balance=True, z_offset=0.0)
        rate.sleep()
    robot.walkStop()
    rospy.sleep(0.3)


def main():
    if SHOW_IMAGE and USE_TRACKBAR:
        setup_trackbars()
    elif SHOW_IMAGE:
        cv2.namedWindow("BlueAdjust", cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    else:
        cap.set(3, FRAME_W)
        cap.set(4, FRAME_H)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera index {}".format(CAMERA_INDEX))
        return

    init_robot_pose()
    print("[READY] Press start button to run blue adjust. Press mode button or q to stop.")

    running = False
    lost_count = 0
    iter_count = 0

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rate.sleep()
            continue

        info, mask = detect_blue(frame)
        if SHOW_IMAGE:
            vis = draw_debug(frame, info, mask)
            cv2.imshow("BlueAdjust", vis)
            key = cv2.waitKey(DISPLAY_DELAY_MS) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                running = True
            elif key == ord('p'):
                running = False
                robot.walkStop()

        if robot.buttonCheck("mode"):
            print("[MODE] stop")
            running = False
            robot.walkStop()

        if (not running) and robot.buttonCheck("start"):
            print("[START] blue adjust start")
            running = True
            lost_count = 0
            iter_count = 0
            set_head(HEAD_TILT_FINE, HEAD_PAN_CENTER, wait=0.3)

        if not running:
            rate.sleep()
            continue

        iter_count += 1
        if iter_count >= FINE_MAX_ITER:
            print("[BLUE_ADJUST] max iter {}, stop".format(FINE_MAX_ITER))
            robot.walkStop()
            running = False
            continue

        info_stable, last_frame, last_mask = sample_blue_stable(cap, FINE_SAMPLE_FRAMES)
        if info_stable is None:
            lost_count += 1
            print("[BLUE_ADJUST] no blue ({}/{})".format(lost_count, FINE_LOST_MAX))
            robot.walkStop()
            if lost_count >= FINE_LOST_MAX:
                print("[BLUE_ADJUST] lost too many -> stop")
                running = False
            continue

        lost_count = 0
        fw = info_stable["fw"]
        target_cx = fw // 2 + TARGET_CX_OFFSET
        offset = info_stable["cx"] - target_cx
        angle = info_stable["angle"] if info_stable["angle"] is not None else 0.0
        angle_stable = info_stable.get("angle_stable", False)

        center_ok = abs(offset) <= BLUE_CENTER_TOL
        angle_ok = (not angle_stable) or abs(angle) <= BLUE_SLOPE_TOL_DEG
        stop_angle_ok = abs(angle) <= BLUE_STOP_ANGLE_TOL_DEG
        distance_ok = info_stable.get("bottom_y", 0) >= BLUE_STOP_BOTTOM_Y
        stop_center_ok = (center_ok or (not REQUIRE_CENTER_TO_STOP))

        print("[BLUE_ADJUST] cx={} offset={} bottom_y={} stop_y={} angle={:.1f} stable={} center_ok={} angle_ok={} distance_ok={}".format(
              info_stable["cx"], offset, info_stable.get("bottom_y", 0), BLUE_STOP_BOTTOM_Y, angle, angle_stable, center_ok, angle_ok, distance_ok))

        if distance_ok and stop_angle_ok and stop_center_ok:
            print("[BLUE_ADJUST] STOP: about 5cm and angle near 0 deg, bottom_y={} angle={:.1f}".format(
                  info_stable.get("bottom_y", 0), angle))
            robot.walkStop()
            running = False
            continue

        # Same style as weightlifting Adjust_Mode: angle pulse first, then small forward/side correction.
        if angle_stable and abs(angle) > BLUE_SLOPE_TOL_DEG:
            if angle < 0:
                x_cmd = FINE_TURN_X
                y_cmd = FINE_TURN_Y
                th_cmd = FINE_TURN_TH
            else:
                x_cmd = FINE_TURN_X
                y_cmd = -FINE_TURN_Y
                th_cmd = -FINE_TURN_TH
            step_time = FINE_TURN_TIME * BLUE_TURN_TIME_SCALE
            print("  [ANGLE_PULSE] x={:.2f} y={:.2f} th={:.2f} angle={:.1f} time={:.2f}".format(
                  x_cmd, y_cmd, th_cmd, angle, step_time))
        else:
            x_cmd = BLUE_FORWARD_X
            y_cmd = np.clip(offset * FINE_Y_KP, -FINE_Y_CLAMP, FINE_Y_CLAMP)
            th_cmd = 0.0
            step_time = BLUE_STEP_TIME
            print("  [FINE_BLUE_FORWARD] x={:.2f} y={:.2f} th={:.2f} time={:.2f}".format(
                  x_cmd, y_cmd, th_cmd, step_time))

        walk_step(x_cmd, y_cmd, th_cmd, step_time, BLUE_Z_AMP)

    robot.walkStop()
    cap.release()
    if SHOW_IMAGE:
        cv2.destroyAllWindows()
    print("[END]")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
