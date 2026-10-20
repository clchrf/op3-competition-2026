#! /usr/bin/python

import rospy
import numpy as np
import cv2
from op3_ros_utils import getWalkingParams, Robot

cap = cv2.VideoCapture(0)

class States:
    INIT = -1
    READY = 0 # Waits for start button
    WALK_FORWARD = 1 # Moves the head, looking for the ball
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

tickrate = 60
rate = rospy.Rate(tickrate)

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
            
        if slope < -0.1:
            print("turn left")
            robot.walkStart()
            robot.walkVelocities(x=0, th=5, balance=True, hip_pitch=7)
            robot.walkStop()
            rospy.sleep(0.2)
        elif slope > 0.1:
            print("turn right")
            robot.walkStart()
            robot.walkVelocities(x=0, th=-5, balance=True, hip_pitch=7)
            robot.walkStop()
            rospy.sleep(0.2)
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
        
    rate.sleep()
