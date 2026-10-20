#! /usr/bin/python

import rospy
import numpy as np
import cv2
from op3_ros_utils import getWalkingParams, Robot
from sensor_msgs.msg import Imu

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0

def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    # print("aa")
    z_gyro = z_gyro - angular_velocity.z/2 + z_gyro_offset_for_caculate
    # z_gyro = angular_velocity.z/2
    # print("z_gyro_offset_for_caculate:",z_gyro_offset_for_caculate)
    z_gyro_offset = angular_velocity.z/2

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

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
    robot.setJointPos(["head_pan", "head_tilt"], [0, -1.2])

    rospy.sleep(1.0)

# Set ctrl modules of all actions to joint, so we can reset robot position
robot.setGeneralControlModule("action_module")

# Call initial robot position
robot.playMotion(1, wait_for_end=True)

# Set ctrl module to walking, this actually only sets the legs
robot.setGeneralControlModule("walking_module")

# Set joint modules of head joints to none so we can control them directly
robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
robot.setJointPos(["head_pan", "head_tilt"], [0, -1.5])

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


tickrate = 60
rate = rospy.Rate(tickrate)

position_correct = []
detect_line = []

currState = States.INIT

rospy.sleep(1)

z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0


cap = cv2.VideoCapture(0)

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
        detect_line = True
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
            
            robot.walkStart()
            currState = States.WALK_FORWARD


    elif currState == States.WALK_FORWARD:
        
        print("[WALK_FORWARD]")
        
        robot.walkStart()
        # robot.walkVelocities(x=5, th=0, balance=True, hip_pitch=7)
        robot.walkVelocities(x=10, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
        robot.walkStop()

        if detect_line is True:
            print("STOP!!!")
            robot.walkStop()
            rospy.sleep(1)
            currState = States.CORRECT_ANGLE
            rospy.sleep(2)
        
    elif currState == States.CORRECT_ANGLE:
        print("[CORRECT_ANGLE]")
            
        if slope < -0.1:
            print("turn left")
            robot.walkStart()
            robot.walkVelocities(x=0, th=3, balance=True, hip_pitch=7)
            robot.walkStop()
            rospy.sleep(0.2)
        elif slope > 0.1:
            print("turn right")
            robot.walkStart()
            robot.walkVelocities(x=0, th=-3, balance=True, hip_pitch=7)
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
