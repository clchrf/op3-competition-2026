#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import math
import time
import cv2
import numpy as np
import rospy
from scipy import optimize
from op3_ros_utils2 import Robot


# ============================================================
# Basic settings
# ============================================================

CAMERA_INDEX = 0
TICK_RATE = 60
DISPLAY_DELAY_MS = 10
SHOOT_DELAY_TIME = 1.8

HEAD_PAN = 1.6
HEAD_TILT = 0.0

STAND_MOTION = 1
SHOOT_READY_MOTION = 202

# False: do not execute any standing motion automatically.
# When False, place the robot manually in a stable walking-ready pose before pressing START.
# Set True only after STAND_MOTION is confirmed to be a safe transition motion.
RUN_STAND_MOTION = True

# Enable the OpenCV preview window.
SHOW_IMAGE = True

# True: alignment test only. False: continue to sitting and shooting.
ALIGN_ONLY_TEST = True


# ============================================================
# Position alignment settings
# ============================================================

# Calibrate this value at the correct standing shooting position.
ALIGN_TARGET_X = 320.0
ALIGN_TOLERANCE_PX = 15.0
ALIGN_STABLE_REQUIRED = 2

ALIGN_MIN_POINTS = 12
ALIGN_COLLECT_TIME = 1.5
ALIGN_MAX_POINTS = 60
ALIGN_BATCH_TIMEOUT = 5.0

ALIGN_MIN_RADIUS = 8.0
ALIGN_MAX_RADIUS = 100.0
ALIGN_MAX_FIT_STD = 10.0

ALIGN_COARSE_ERROR_PX = 50.0
ALIGN_COARSE_X = 3.0
ALIGN_COARSE_TIME = 1.0
ALIGN_FINE_X = 1.5
ALIGN_FINE_TIME = 0.8

ALIGN_SETTLE_TIME = 0.8

# Change to -1.0 if the correction direction is reversed.
ALIGN_MOVE_SIGN = 1.0
ALIGN_MAX_STEPS = 12

ALIGN_Z_MOVE_AMPLITUDE = 0.025
ALIGN_BALANCE = True
ALIGN_Z_OFFSET = 0.0


# ============================================================
# Original target detection settings
# ============================================================

LOWER_RED = np.array([151, 100, 100])
UPPER_RED = np.array([179, 255, 154])
LOWER_YELLOW = np.array([31, 132, 100])
UPPER_YELLOW = np.array([44, 255, 255])


class States(object):
    WAIT_START = 0
    ALIGN_POSITION = 1
    PREPARE_SHOOTING = 2
    CALCULATE_TARGET = 3
    END = 99


def find_contours_compat(mask):
    """Support both OpenCV 3 and OpenCV 4."""
    result = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[1] if len(result) == 3 else result[0]


def calculate_center(mask):
    contours = find_contours_compat(mask)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def fit_circle(points):
    """Fit a circle to the rotating marker trajectory."""
    if len(points) < 3:
        return None

    x_coords = np.asarray([point[0] for point in points], dtype=np.float64)
    y_coords = np.asarray([point[1] for point in points], dtype=np.float64)

    def calc_radius(center):
        center_x, center_y = center
        return np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

    def residual(center):
        radius_array = calc_radius(center)
        return radius_array - np.mean(radius_array)

    center_estimate = (np.mean(x_coords), np.mean(y_coords))

    try:
        center, _ = optimize.leastsq(residual, center_estimate)
    except Exception as error:
        print("[FIT] leastsq failed: {}".format(error))
        return None

    center_x = float(center[0])
    center_y = float(center[1])
    radius_array = calc_radius(center)
    radius = float(np.mean(radius_array))
    residual_std = float(np.std(radius_array - radius))

    if not (np.isfinite(center_x) and np.isfinite(center_y) and np.isfinite(radius) and np.isfinite(residual_std)):
        return None

    return center_x, center_y, radius, residual_std


def reset_tracking():
    now = time.time()
    return {
        "positions": [],
        "angular_velocity_history": [],
        "prev_angle": None,
        "prev_time": now,
        "start_time": now,
        "batch_start_time": now
    }


def set_side_head(robot):
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [HEAD_PAN, HEAD_TILT])


def walk_x_pulse(robot, rate, x_command, duration):
    """Move a short distance along the robot X axis."""
    print("[ALIGN_MOVE] x={:.2f}, duration={:.2f}".format(x_command, duration))

    robot.setGeneralControlModule("walking_module")
    rospy.sleep(0.2)
    set_side_head(robot)

    robot.walkStart()
    move_start_time = time.time()

    while not rospy.is_shutdown() and time.time() - move_start_time < duration:
        robot.walkVelocities(x=x_command, y=0.0, th=0.0, z_move_amplitude=ALIGN_Z_MOVE_AMPLITUDE,
                             balance=ALIGN_BALANCE, z_offset=ALIGN_Z_OFFSET)
        rate.sleep()

    robot.walkStop()
    rospy.sleep(ALIGN_SETTLE_TIME)
    set_side_head(robot)
    rospy.sleep(0.2)


def prepare_standing_alignment(robot):
    if RUN_STAND_MOTION:
        print("[INIT] Execute standing motion {}".format(STAND_MOTION))
        robot.setGeneralControlModule("action_module")
        rospy.sleep(0.5)
        robot.playMotion(STAND_MOTION, wait_for_end=True)
        rospy.sleep(0.8)
    else:
        print("[INIT] Automatic standing motion disabled")
        print("[INIT] Make sure the robot is already in a stable walking-ready pose")

    robot.setGeneralControlModule("walking_module")
    rospy.sleep(0.5)
    set_side_head(robot)
    rospy.sleep(1.0)


def prepare_camera_preview(robot):
    """Turn only the head for camera preview without moving the body."""
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [HEAD_PAN, HEAD_TILT])
    rospy.sleep(1.0)


def prepare_original_shooting_pose(robot):
    robot.walkStop()
    rospy.sleep(0.5)

    robot.setGeneralControlModule("action_module")
    rospy.sleep(0.5)
    robot.playMotion(SHOOT_READY_MOTION, wait_for_end=True)
    rospy.sleep(2.0)

    set_side_head(robot)
    rospy.sleep(1.0)


def draw_status(image, state, align_step, stable_count):
    if state == States.WAIT_START:
        status_text = "PRESS START"
    elif state == States.ALIGN_POSITION:
        status_text = "ALIGN step={} stable={}/{}".format(align_step, stable_count, ALIGN_STABLE_REQUIRED)
    elif state == States.PREPARE_SHOOTING:
        status_text = "PREPARE SHOOTING"
    elif state == States.CALCULATE_TARGET:
        status_text = "TRACKING / SHOOTING"
    else:
        status_text = "END"

    cv2.putText(image, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


def main():
    rospy.init_node("fira_archery_position_correct")

    robot = Robot()
    rate = rospy.Rate(TICK_RATE)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        rospy.logerr("Cannot open camera index {}".format(CAMERA_INDEX))
        return

    rospy.sleep(1.0)
    prepare_camera_preview(robot)

    current_state = States.WAIT_START
    tracking = reset_tracking()
    align_step_count = 0
    align_stable_count = 0
    shoot = False

    try:
        while not rospy.is_shutdown():
            ret, image = cap.read()
            if not ret:
                print("[CAMERA] frame read failed")
                break

            image = cv2.flip(image, 1)
            current_time = time.time()

            # HSV segmentation
            image_blur = cv2.GaussianBlur(image, (3, 3), 5)
            hsv_image = cv2.cvtColor(image_blur, cv2.COLOR_BGR2HSV)

            mask_red = cv2.inRange(hsv_image, LOWER_RED, UPPER_RED)
            mask_yellow = cv2.inRange(hsv_image, LOWER_YELLOW, UPPER_YELLOW)
            combined_mask = cv2.add(mask_red, mask_yellow)

            contours_red = find_contours_compat(mask_red)
            contours_yellow = find_contours_compat(mask_yellow)
            cv2.drawContours(image, contours_red, -1, (0, 0, 255), 2)
            cv2.drawContours(image, contours_yellow, -1, (0, 255, 255), 2)

            center_red = calculate_center(mask_red)
            center_yellow = calculate_center(mask_yellow)

            # Hough circle detection
            hough_image = cv2.GaussianBlur(combined_mask, (7, 7), 1)
            hough_image = cv2.medianBlur(hough_image, 5)
            hough_image = cv2.blur(hough_image, (5, 5))

            detected_circles = cv2.HoughCircles(hough_image, cv2.HOUGH_GRADIENT, 1, 1000, param1=30,
                                                param2=15, minRadius=1, maxRadius=70)
            latest_marker_radius = 5

            if detected_circles is not None:
                circles = np.uint16(np.around(detected_circles))

                for circle in circles[0, :]:
                    marker_x = int(circle[0])
                    marker_y = int(circle[1])
                    marker_radius = int(circle[2])
                    latest_marker_radius = max(marker_radius, 3)

                    tracking["positions"].append((marker_x, marker_y))
                    cv2.circle(image, (marker_x, marker_y), marker_radius, (0, 255, 0), 3)
                    cv2.circle(image, (marker_x, marker_y), 2, (0, 100, 100), 3)

            if current_state == States.ALIGN_POSITION:
                tracking["positions"] = tracking["positions"][-ALIGN_MAX_POINTS:]

            circle_fit = fit_circle(tracking["positions"])

            if circle_fit is not None:
                center_x, center_y, fitted_radius, fit_std = circle_fit
                cv2.circle(image, (int(center_x), int(center_y)), int(fitted_radius), (255, 255, 255), 2)
                cv2.circle(image, (int(center_x), int(center_y)), 3, (255, 0, 255), 3)
            else:
                center_x = None
                center_y = None
                fitted_radius = None
                fit_std = None

            if current_state in (States.WAIT_START, States.ALIGN_POSITION):
                cv2.line(image, (int(ALIGN_TARGET_X), 0), (int(ALIGN_TARGET_X), image.shape[0] - 1),
                         (255, 0, 0), 2)

            # Wait for the start button
            if current_state == States.WAIT_START:
                if robot.buttonCheck("start"):
                    print("[STATE] START pressed")
                    prepare_standing_alignment(robot)
                    print("[STATE] START -> ALIGN_POSITION")
                    tracking = reset_tracking()
                    align_step_count = 0
                    align_stable_count = 0
                    current_state = States.ALIGN_POSITION

            # Standing position alignment
            elif current_state == States.ALIGN_POSITION:
                enough_points = len(tracking["positions"]) >= ALIGN_MIN_POINTS
                enough_time = current_time - tracking["batch_start_time"] >= ALIGN_COLLECT_TIME
                valid_fit = (circle_fit is not None and ALIGN_MIN_RADIUS <= fitted_radius <= ALIGN_MAX_RADIUS
                             and fit_std <= ALIGN_MAX_FIT_STD)

                if enough_points and enough_time and valid_fit:
                    position_error = center_x - ALIGN_TARGET_X
                    absolute_error = abs(position_error)

                    print("[ALIGN] center_x={:.1f}, target={:.1f}, error={:.1f}, radius={:.1f}, fit_std={:.2f}".format(
                        center_x, ALIGN_TARGET_X, position_error, fitted_radius, fit_std))

                    if absolute_error <= ALIGN_TOLERANCE_PX:
                        align_stable_count += 1
                        print("[ALIGN] inside tolerance {}/{}".format(align_stable_count, ALIGN_STABLE_REQUIRED))

                        if align_stable_count >= ALIGN_STABLE_REQUIRED:
                            print("[ALIGN] position correction completed")
                            robot.walkStop()

                            if ALIGN_ONLY_TEST:
                                print("[ALIGN_ONLY_TEST] Stop here. Set ALIGN_ONLY_TEST=False after validation.")
                                current_state = States.END
                            else:
                                current_state = States.PREPARE_SHOOTING
                        else:
                            tracking = reset_tracking()

                    else:
                        align_stable_count = 0

                        if align_step_count >= ALIGN_MAX_STEPS:
                            print("[ALIGN] maximum step count reached; stop")
                            robot.walkStop()
                            current_state = States.END
                        else:
                            if absolute_error > ALIGN_COARSE_ERROR_PX:
                                move_speed = ALIGN_COARSE_X
                                move_duration = ALIGN_COARSE_TIME
                                move_mode = "COARSE"
                            else:
                                move_speed = ALIGN_FINE_X
                                move_duration = ALIGN_FINE_TIME
                                move_mode = "FINE"

                            direction = 1.0 if position_error > 0 else -1.0
                            x_command = ALIGN_MOVE_SIGN * direction * move_speed

                            print("[ALIGN_{}] error={:.1f}, x_command={:.2f}".format(
                                move_mode, position_error, x_command))

                            walk_x_pulse(robot, rate, x_command, move_duration)
                            align_step_count += 1
                            tracking = reset_tracking()

                elif current_time - tracking["batch_start_time"] > ALIGN_BATCH_TIMEOUT:
                    print("[ALIGN] insufficient or invalid circle data; reset batch")
                    tracking = reset_tracking()

            # Execute the original sitting motion
            elif current_state == States.PREPARE_SHOOTING:
                print("[STATE] PREPARE_SHOOTING")
                prepare_original_shooting_pose(robot)
                tracking = reset_tracking()
                current_state = States.CALCULATE_TARGET
                print("[STATE] CALCULATE_TARGET")

            # Original path prediction and shooting logic
            elif current_state == States.CALCULATE_TARGET:
                if circle_fit is not None and len(tracking["positions"]) > 3:
                    latest_x, latest_y = tracking["positions"][-1]
                    angle = np.arctan2(latest_y - center_y, latest_x - center_x)

                    elapsed_from_shoot_start = current_time - tracking["start_time"]
                    elapsed_time = current_time - tracking["prev_time"]

                    if elapsed_from_shoot_start > 0.5 and elapsed_time > 0.000001:
                        previous_angle = tracking["prev_angle"]

                        if previous_angle is not None:
                            angle_change = angle - previous_angle

                            if angle_change > np.pi:
                                angle_change -= 2.0 * np.pi
                            elif angle_change < -np.pi:
                                angle_change += 2.0 * np.pi

                            angular_velocity = angle_change / elapsed_time
                            tracking["angular_velocity_history"].append(angular_velocity)
                            average_angular_velocity = np.mean(tracking["angular_velocity_history"])

                            predicted_angle = previous_angle + average_angular_velocity * float(SHOOT_DELAY_TIME)
                            predicted_x = int(center_x + fitted_radius * np.cos(predicted_angle))
                            predicted_y = int(center_y + fitted_radius * np.sin(predicted_angle))

                            cv2.circle(image, (predicted_x, predicted_y), latest_marker_radius, (0, 0, 255), 3)

                            target_bottom = (center_x, center_y + fitted_radius)
                            prediction_distance = calculate_distance((predicted_x, predicted_y), target_bottom)

                            should_check_shoot = ((prediction_distance <= 5.0 and elapsed_from_shoot_start > 15.0)
                                                  or elapsed_from_shoot_start > 30.0)

                            if should_check_shoot and center_red is not None and center_yellow is not None:
                                red_yellow_distance = calculate_distance(center_red, center_yellow)

                                if red_yellow_distance < 5.0:
                                    print("All centers overlapped. Shoot")
                                    shoot = True
                                    robot.setGeneralControlModule("action_module")
                                    robot.playMotion(209, wait_for_end=True)
                                    current_state = States.END
                                    rospy.sleep(1.0)

                        tracking["prev_time"] = current_time
                        tracking["prev_angle"] = angle

            draw_status(image, current_state, align_step_count, align_stable_count)

            if SHOW_IMAGE:
                cv2.imshow("Archery Alignment", image)
                key = cv2.waitKey(DISPLAY_DELAY_MS) & 0xFF
            else:
                key = 255
            if key == 27 or key == ord("q"):
                print("[KEY] quit")
                break

            if shoot or current_state == States.END:
                break

    finally:
        try:
            robot.walkStop()
        except Exception:
            pass

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()