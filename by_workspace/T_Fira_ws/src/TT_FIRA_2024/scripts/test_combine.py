#! /usr/bin/python

import rospy
from op3_ros_utils import getWalkingParams, Robot
import cv2
import numpy as np
import time
import math
from scipy import optimize

cap = cv2.VideoCapture(0)

class States:
    INIT = -1
    READY = 0 
    WALK_FORWARD = 1
    CORRECT_ANGLE = 2
    END = 99

rospy.init_node("walk_op3")

# Create robot
robot = Robot()

rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages

def init():
    # Set ctrl modules of all actions to joint, so we can reset robot position
    robot.setGeneralControlModule("action_module")

    # Call initial robot position
    robot.playMotion(1, wait_for_end=True)

    # Set ctrl module to walking, this actually only sets the legs
    robot.setGeneralControlModule("walking_module")
    
    # Set joint modules of head joints to none so we can control them directly
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [0, -0.5])

    rospy.sleep(1.0)

def move_robot(line):
    x1, y1, x2, y2 = line
    # Calculate the angle of the line relative to the horizontal axis
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    
    if angle > 10:
        print("turn right")
        robot.walkVelocities(x=0, th=5, balance=True, hip_pitch=7)
    elif angle < -10:
        print("turn left")
        robot.walkVelocities(x=0, th=-5, balance=True, hip_pitch=7)
    else:
        print("position correct")
        position_correct = True

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x_coords-xc)**2 + (y_coords-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    xc, yc = c  # Unpack the tuple into xc and yc
    Ri = calc_R(xc, yc)  # Call calc_R with individual coordinates
    return Ri - np.mean(Ri)

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

tickrate = 60
rate = rospy.Rate(tickrate)

cap = cv2.VideoCapture(0)

delay_between_frames = 10
# delay_time = input("delay time : ")
delay_time = 0.5

detected_circle_positions = []

start_time = time.time()
current_time = start_time
prev_time = start_time

prev_angle = None
angular_velocity = None
angular_velocity_history = []

# robot.setGeneralControlModule("action_module")
# robot.playMotion(1, wait_for_end=True)
# time.sleep(2)

robot.setGeneralControlModule("action_module")
robot.playMotion(94, wait_for_end=True)
time.sleep(2)

robot.setGeneralControlModule("action_module")
robot.playMotion(95, wait_for_end=True)
time.sleep(2)

robot.setGeneralControlModule("action_module")
robot.playMotion(96, wait_for_end=True)
time.sleep(2)

shoot = []
position_correct = []

currState = States.INIT

while not rospy.is_shutdown():
    
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([22, 69, 128])
    upper_yellow = np.array([38, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    blurred = cv2.medianBlur(mask, 5)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5,5),np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

            slope = float(y2 - y1) / float(x2 - x1)
            print(slope)
    
    cv2.imshow('Video', frame)
    cv2.waitKey(1)

    if currState == States.INIT:
        print("[INIT]")
        init()

        direction = False
        currState = States.READY

    elif currState == States.READY:
        
        print("[READY]")
        if robot.buttonCheck("start"):
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.5])
            
            robot.walkStart()
            currState = States.WALK_FORWARD

    elif currState == States.WALK_FORWARD:
        
        print("[WALK_FORWARD]")
        if robot.buttonCheck("start"):
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.5])
            
            robot.walkStart()
            robot.walkVelocities(x=5, th=0, balance=True, hip_pitch=7)

            if lines is not None:
                robot.walkStop()
                currState = States.CORRECT_ANGLE
        
    elif currState == States.CORRECT_ANGLE:
        print("[WALK_FORWARD]")
            
        if slope < 0.1:
            print("turn left")
            robot.walkStart()
            robot.walkVelocities(x=0, th=15, balance=True, hip_pitch=7)
            robot.walkStop()
        elif slope > -0.1:
            print("turn right")
            robot.walkStart()
            robot.walkVelocities(x=0, th=-15, balance=True, hip_pitch=7)
            robot.walkStop()
        else:
            print("position correct")
            position_correct = True
            rospy.sleep(1)

        if position_correct == True:
            robot.walkStop()
            currState = States.END
            rospy.sleep(1)

    elif currState == States.END:
        print("[END]")
        break

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    current_time = time.time()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    gray_blurred = cv2.medianBlur(gray_blurred, 5)
    gray_blurred = cv2.blur(gray_blurred, (5, 5))
    # gray_blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT,
                                        1, 1000, param1=50, param2=30,
                                        minRadius=1, maxRadius=25)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for i in detected_circles[0, :]:
            a, b, r = i[0], i[1], i[2]
            
            detected_circle_positions.append((a, b))

            cv2.circle(img, (a, b), r, (0, 255, 0), 3)
            cv2.circle(img, (a, b), 1, (0, 100, 100), 3)
            print(a, b)

    if len(detected_circle_positions) >= 3:
        x_coords, y_coords = zip(*detected_circle_positions)

        center_estimate = (np.mean(x_coords), np.mean(y_coords))  # Estimate initial center
        center, ier = optimize.leastsq(f_2, center_estimate)

        xc, yc = center

        Ri_2 = calc_R(*center)
        R_2 = Ri_2.mean()

        radius = np.mean(R_2)
        center_x, center_y = xc, yc

        if current_time - start_time > 0.5:
            cv2.circle(img, (int(center_x), int(center_y)), int(radius), (255, 255, 255), 3)
            cv2.circle(img, (int(center_x), int(center_y)), 3, (0, 255, 0), 3)

    if len(detected_circle_positions) > 3:
        a, b = detected_circle_positions[-1]
        displacement_vector = (a - center_x, b - center_y)

        angle = np.arctan2(displacement_vector[1], displacement_vector[0])

        current_time = time.time()
        elapsed_time = current_time - prev_time

        if current_time - start_time > 0.5:
            if prev_angle is not None:
                angle_change = angle - prev_angle

                if angle_change > np.pi:
                    angle_change -= 2 * np.pi
                elif angle_change < -np.pi:
                    angle_change += 2 * np.pi

                angular_velocity = angle_change / elapsed_time
                angular_velocity_history.append(angular_velocity)
                avg_angular_velocity = np.mean(angular_velocity_history)

                delay_time += delay_time * angular_velocity
                angle_change = avg_angular_velocity * float(delay_time)
                new_angle = prev_angle + angle_change
                new_x = int(center_x + radius * np.cos(new_angle))
                new_y = int(center_y + radius * np.sin(new_angle))

                cv2.circle(img, (new_x, new_y), r, (0, 0, 255), 3)

                distance = calculate_distance((new_x, new_y), (center_x, (center_y + radius)))
                target_height = center_y + radius
                # print(distance)
                # cv2.circle(img, (int(center_x), int(center_y + radius)), r, (255, 255, 255), 3)

                if distance <= 5:
                    cv2.circle(img, (new_x, new_y), r, (255, 255, 255), 3)

                    if target_height < 250:
                        shoot = True
                        print("shoot")
                        print("high target : ", target_height)
                        robot.setGeneralControlModule("action_module")
                        robot.playMotion(97, wait_for_end=True)
                        time.sleep(2)
                    elif target_height > 300:
                        shoot = True
                        print("shoot")
                        print("low target : ", target_height)
                        robot.setGeneralControlModule("action_module")
                        robot.playMotion(97, wait_for_end=True)
                        time.sleep(2)
                    else:
                        print("shoot")
                        shoot = True
                        print("normal target : ", target_height)
                        robot.setGeneralControlModule("action_module")
                        robot.playMotion(97, wait_for_end=True)
                        time.sleep(2)

            prev_time = current_time
            prev_angle = angle

    cv2.imshow("show image", img)

    key = cv2.waitKey(delay_between_frames)
    # if current_time - start_time > 30:
    if shoot is True:
        break

cap.release()
cv2.destroyAllWindows()
