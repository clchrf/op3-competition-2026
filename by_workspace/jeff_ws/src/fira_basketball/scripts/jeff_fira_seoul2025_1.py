#! /usr/bin/python

##adjust line 469 482 684
import rospy
import numpy as np
import cv2
import time
import math


from op3_ros_utils import getWalkingParams, Robot
from vision import *
from copy import copy
from sensor_msgs.msg import Imu
import sys

DEBUG_MODE = False
# DEBUG_MODE = True 
MIN_AREA = 300
CROSS_AREA = 1165
DEGREE2RADIAN = np.pi / 180

class States:
    INIT = -1
    READY = 0 # Waits for start button
    WALK_FORWARD = 1 # Moves the head, looking for the ball
    PICK_BAR = 5
    WALK_WITH_BAR = 6
    LIFT_BAR = 7
    WALK_2_FINISH = 8
    CHECK_SLOPE = 9
    Adjust_Mode = 11
    Adjust_Mode_V2 = 13
    CHECK_SLOPE_V2 = 14
    CHECK_PAN = 10
    CHECK_TOUCH = 12
    Buffer = 20
    END = 99

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0

_count = 0
def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    # print("aa")
    # z_gyro = z_gyro - math.floor(angular_velocity.z/2*100)/100 + z_gyro_offset_for_caculate
    z_gyro = z_gyro - (angular_velocity.z/2) + z_gyro_offset_for_caculate
    # z_gyro = angular_velocity.z/2
    # print("z_gyro_offset_for_caculate:",z_gyro_offset_for_caculate)
    # print(z_gyro)
    z_gyro_offset = angular_velocity.z/2

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

global last_Px, last_Py, time_for_Pd
last_Px = 0.0
last_Py = 0.0
time_for_Pd = time.time()

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# Define your color range for detection

# lower_hsv = np.array([16, 77, 0])
# upper_hsv = np.array([43, 255, 198])
# (np.array([104, 86, 93]), np.arr
# ay([179, 255, 255]
lower_hsv = np.array([109, 36, 42])
upper_hsv = np.array([136 ,230 ,255])

lower_g=np.array([68, 57, 48])
upper_g=np.array([ 96, 255, 255])

# lower_r=np.array([0, 59, 137])
# upper_r=np.array([ 180, 255, 181])

lower_w = np.array([ 0, 0, 190])
upper_w = np.array([180, 255, 255])

# lower value of yel1:[ 22  69 128]
# upper value of yel1:[ 38 255 255]

mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

go_back = 0
position_correct = []

# Iinitialize Node
rospy.init_node("fira_sprint")

# Create robot
robot = Robot()



rospy.sleep(3) # Make sure every publisher has registered to their topic,
               # avoiding lost messages
# rospy.sleep(2) # Make sure every publisher has registered to their topic,               # avoiding lost messages
z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0
#sure
def detect_single_color(frame, lower_hsv, upper_hsv):
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([163, 196, 65])
    upper_red1 = np.array([179 ,255 ,198])
    lower_red2 = np.array([163, 196, 65])
    upper_red2 = np.array([179 ,255 ,198])

    mask_R1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_R2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_comb = cv2.add(mask_R1, mask_R2)
    # Create a mask with the given HSV range
    # mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    kernel = np.ones((5,5), np.uint8) 
    mask_comb = cv2.morphologyEx(mask_comb, cv2.MORPH_OPEN, kernel)
    #eliminating fine red dot noise
    
    # Convert mask to a 3-channel image to put a colored dot
    colored_mask = cv2.cvtColor(mask_comb, cv2.COLOR_GRAY2BGR)
    
    # Find contours in the mask
    # Note: cv2.findContours returns 3 values in older OpenCV versions used with Python 2.7
    _, contours, _ = cv2.findContours(mask_comb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    # Ensure at least one contour was found
    if contours:
        # Find the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate the center of the contour
        M = cv2.moments(largest_contour)

        selection = cv2.minAreaRect(largest_contour)
        (x, y), (width, height), slope = selection    
        angle = compute_angle(width, height, slope)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Draw a circle at the center on the original frame
            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)  # Blue dot
            # Draw the largest contour on the original frame
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)  # Green contour
            return cX, cY, angle
        
        else:
            # cX, cY = 0, 0  # Default to (0,0) if contour is a line
            return None, None,None
        
def detect_single_color_V2(frame, lower_hsv, upper_hsv):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ratio = 2.0
    hor_contour = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (x, y), (width, height), slope = rect

        if width > height * ratio:
            hor_contour.append(contour)

    if hor_contour:

        largest_contour = max(hor_contour, key=cv2.contourArea)
        
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)  # Blue dot
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)  # Green contour

            angle = rect[2]
            if width < height:
                angle += 90
            
            cv2.imshow("frame",frame)
                
            return cX, cY, angle
        else:
            return None, None,None

def detect_line(frame,hsv_l,hsv_h):
    height,width=frame.shape[:2]
    point_x=width//2
    point_y=height-1-320

    color = frame[point_y,point_x]
    hsv_color=cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_BGR2HSV)

    detected = cv2.inRange(hsv_color,hsv_l,hsv_h).any()
    circle_color = (0,0,0)
    cv2.circle(frame,(point_x,point_y),5,circle_color,-1)
    # return frame,detected
    return frame,detected

def compute_angle(width, height, angle):
    if angle < -90 or (width > height and angle < 0):
        return 90 + angle
    if width < height and angle > 0:
        return (90 - angle) * -1 
    return angle

#sure
def camera_theta(cX, cY, frame_width, frame_height):
    global last_Px, last_Py, time_for_Pd
    
    # Convert cX and cY to normalized coordinates relative to the frame center
    Px = -1 * ((cX / float(frame_width)) - 0.5)
    Py = -1 * ((cY / float(frame_height)) - 0.5)
    
    # Assuming robot.joint_pos is accessible and contains the current angles in radians
    pan_angle_old = np.degrees(robot.joint_pos["head_pan"])
    tilt_angle_old = np.degrees(robot.joint_pos["head_tilt"])
    
    # Calculate derivative of position
    Dx = (Px - last_Px) / (time.time() - time_for_Pd)
    Dy = (Py - last_Py) / (time.time() - time_for_Pd)
    time_for_Pd = time.time()
    
    # Proportional and derivative gains (tune these according to your robot's characteristics)
    kp = 15
    kd = 0
    
    # Calculate new angles
    pan_angle = pan_angle_old + Px * kp + Dx * kd
    tilt_angle = tilt_angle_old + Py * kp + Dy * kd
    
    # Update last position for next iteration
    last_Px, last_Py = Px, Py
    
    # Apply the new angles, ensuring they're within the mechanical limits of the robot
    robot.setJointPos(["head_pan", "head_tilt"], [np.clip(np.radians(pan_angle), np.radians(-95), np.radians(95)), np.clip(np.radians(tilt_angle), np.radians(-95), np.radians(95))])
    
    # print("Pan angle:", pan_angle)
    # print("Tilt angle:", tilt_angle)
    return pan_angle, tilt_angle

def detect_middle(frame,Cx):
    height, width = frame.shape[:2]
    frame_middle_x = width // 2
    orientation = 2
    cv2.line(frame,(int(Cx),0),(Cx,height),(255,0,0),2)
    if (frame_middle_x+4) >= Cx >= (frame_middle_x-4):
        print("touching")
        detect_touch = True
    elif Cx > frame_middle_x+4:           #line on the right
        orientation = 0
        detect_touch = False
    elif Cx < frame_middle_x-4:
        orientation = 1
        detect_touch = False
    # cv2.imshow('Frame_touch',frame)
    return frame,detect_touch, orientation


def is_line_detected(frame, lower_color, upper_color, threshold=5000):

    #check bottom area color,slope and show 
    height, width = frame.shape[:2]
    # detect low bottom
    roi = frame[height//2:height, :] 
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, lower_color, upper_color)
    
    #detect color pixcel
    pixel_count = cv2.countNonZero(mask)
    
    # Debug mode 
    text = "Area: " + str(pixel_count) + " / Thresh: " + str(threshold)
    # area enough => green else red
    text_color = (0, 255, 0) if pixel_count > threshold else (0, 0, 255)
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
    # yellow line 
    cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 255), 2)

    
    if pixel_count > threshold:
        return True
    return False


def init():
    # Set ctrl modules of all actions to joint, so we can reset robot position 
    robot.setGeneralControlModule("action_module")

    # robot.setGrippersPos(left=0.0, right=0.0)

    # Call initial robot position
    robot.playMotion(1, wait_for_end=True)

    # Set ctrl module to walking, this actually only sets the legs
    robot.setGeneralControlModule("walking_module")
    
    # Set joint modules of head joints to none so we can control them directly
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])

    robot.setJointPos(["head_tilt" , "head_pan"] , [-0.45 , 0])

    rospy.sleep(1.0)

tickrate = 60
rate = rospy.Rate(tickrate)

# TODO remember to put to 0
STEP_LEVEL = 0

currState = States.INIT
while not rospy.is_shutdown():
    ret, frame = cap.read()
    if ret:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result = detect_single_color(frame, lower_hsv, upper_hsv)
        if result is not None:  # Ensure there's a result to unpack
            cX, cY ,ang= result
            # print("computed:",ang)
            if ang is not None:
                if ang>0:
                    ang=90.0-ang
                elif ang<=0:
                    ang=(-90.0)-ang
                    print("slope:",ang)
            if cX is not None and cY is not None:
                # Now it's safe to use cX and cY
                pan_angle, tilt_angle = camera_theta(cX, cY, frame_width, frame_height)
        # cv2.imshow('MASK',comb_red)
        cv2.imshow('frame',frame)
        cv2.waitKey(1)
    
    if robot.buttonCheck("mode"):
        STEP_LEVEL = 0
        currState = States.INIT
    
    if DEBUG_MODE:
        # print(np.degrees(robot.joint_pos["head_tilt"]))
        lower_hsv = np.array([100,  69, 128])
        upper_hsv = np.array([ 140 ,255 ,255])
        lower_g=np.array([68, 57, 48])
        upper_g=np.array([ 96, 255, 255])

        lower_r=np.array([0, 59, 137])
        upper_r=np.array([ 180, 255, 181])

        lower_w = np.array([ 0, 0, 231])
        upper_w = np.array([180, 255, 255])
        
        # lower value of yellow1:[ 11  11 164]
        # upper value of yellow1:[ 68 255 255]
        # lower value of red1:[95 81  0]
# upper value of red1:[124 255 255]
        # ret, frame = cap.read()
        # detect_single_color(frame,lower_hsv,upper_hsv)
        
        ret, frame = cap.read()
        # print("ret,",ret)
        height, width = frame.shape[:2]
        frame_middle_x = width // 2
        # print(frame_middle_x)
        # print("width",width)
        # line_frame,detect = detect_line(frame,lower_w,upper_w)
        # print("WHATATATAT")
        # touch_frame,_,_=detect_middle(frame,cX)        
        # cv2.imshow("god",touch_frame)

        if ret:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # result = detect_single_color(frame, lower_hsv, upper_hsv)
            # result = detect_single_color(frame, lower_g, upper_g)
            result = detect_single_color(frame, lower_hsv, upper_hsv)

            # cX, cY ,ang= detect_single_color(frame, lower_hsv, upper_hsv)  # This function should return the centroid coordinates\
            
            if result is not None:  # Ensure there's a result to unpack
                cX, cY , ang = result
                if ang>0:
                    ang=90-ang
                elif ang<=0:
                    ang=(-90)-ang
                if cX is not None and cY is not None:
                    # Now it's safe to use cX and cY
                    # pan_angle,_= camera_theta(cX, cY, frame_width, frame_height)
                    # print("slope:",ang)
                    # print("robot_pan_ang:",abs(np.degrees(robot.joint_pos["head_pan"])))
                    touch_frame,tf,what=detect_middle(frame,cX)        
                    # print("tf:",tf)
                    # print("what:",what)
                    cv2.imshow("god",touch_frame)
        # cv2.imshow('MASK',mask)
        # cv2.imshow('frame',frame)
        cv2.waitKey(1)



    if currState == States.INIT:
        print("[INIT]")
        init()

        # Transition
        tick_count = 0
        direction = False
        currState = States.READY

    elif currState == States.READY:
        print("[READY]")
        if robot.buttonCheck("start"):
            # rospy.sleep(1)
            rospy.sleep(5)                      #for video taping
            z_gyro_offset_for_caculate = z_gyro_offset
            z_gyro = 0.0
            print(z_gyro)
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none","none"])
            robot.setJointPos(["head_pan", "head_tilt"], [0, -0.45])
            tick_count = 0
            robot.walkStart()
            # currState = States.WALK_FORWARD
            currState = States.Adjust_Mode_V2
            


    elif currState == States.Adjust_Mode_V2:
        
        _count = _count+1
        #robot.walkStart()  
        # y =  np.clip(3*np.degrees(robot.joint_pos["head_pan"] ), -18, 10)  th=np.clip(z_gyro*1,-3,3) x = (-65-np.degrees(robot.joint_pos["head_tilt"]))*0.005
        ##adjst error in y 
        # robot.walkVelocities(x=10, y=0, th=np.clip(_the, -10, 10), z_move_amplitude=0.045, balance=True, z_offset=0)# y+:left,-:right the+:left, -:right
        # robot.walkVelocities(x = 5, y = np.clip(np.degrees(robot.joint_pos["head_pan"]),-5,0)*3.0, th=np.clip(z_gyro, -10 , 10) ,z_move_amplitude=0.045,balance=True, feet_pitch = 3)# ,balance=True
        #robot.walkVelocities(x = 5, y = 9, th=np.clip(z_gyro, -5 , 5) ,z_offset= 0 ,z_move_amplitude=0.045,balance=True, feet_pitch = 3)# ,balance=True
        # robot.walkVelocities(x = 0.1, y = np.clip(np.degrees(robot.joint_pos["head_pan"]),-3,3)*2.0, th=-10,hip_pitch=10)# ,balance=True
        
        #read head angle
        #pan_angle_deg = np.degrees(robot.joint_pos["head_pan"]) 
        #tilt_angle_deg = np.degrees(robot.joint_pos["head_tilt"])
        raw_pan = np.degrees(robot.joint_pos["head_pan"]) 
        raw_tilt = np.degrees(robot.joint_pos["head_tilt"])
        
        if raw_pan < -150 or raw_tilt < -150:
            print("WARNING: Motor read error! Ignoring values.")
            pan_angle_deg = 0.0
            tilt_angle_deg = 0.0
        else:
            pan_angle_deg = raw_pan
            tilt_angle_deg = raw_tilt
        
        #IMU balance /p controler
        pan_offset = 0.0
        pan_error = pan_angle_deg - pan_offset

        if abs(pan_error) > 8.0:
            dynamic_x = 1.5 
            
            # Y axis shift
            dynamic_y = np.clip(pan_error * -1.8, -18, 18) 
            
            print(">>> [Aligning] X: Stop | Y: Moving to align | pan:", pan_error)
            
        else:
            dynamic_x = 5.0
            
            dynamic_y = np.clip(pan_error * -0.5, -5, 5)
            
            print(">>> [Approaching] X: Forward | Y: Micro-adjust | tilt:", tilt_angle_deg)

      
        dynamic_th = np.clip(z_gyro, -8, 8)
        robot.walkVelocities(x = dynamic_x, y = dynamic_y, th = dynamic_th, z_offset= 0, z_move_amplitude=0.045, balance=True, feet_pitch = 3)
        target_tilt = 65.0  
    
        #if(abs(78+np.degrees(robot.joint_pos["head_tilt"] ))  < 5.8):
        if(abs(target_tilt + tilt_angle_deg) <5.8) and (abs(pan_error) < 15.0) :
            print("Target locked Stopping to pick up the bar")
            robot.walkStop()
            rospy.sleep(2)  
            currState = States.PICK_BAR
            rospy.sleep(1)
            show_tilt = 1
            show_gyro = 0


    elif currState == States.PICK_BAR:
        rospy.loginfo("[PICK_BAR]")
        rospy.sleep(2)
        robot.setGeneralControlModule("none")
        rospy.sleep(2)
        robot.setGeneralControlModule("action_module")
        # robot.playMotion(178, wait_for_end=True)     #wait pos
        # time.sleep(2)
        robot.playMotion(192, wait_for_end=True)     #wait pos 100,192
        time.sleep(2)
        z_gyro_offset_for_caculate = z_gyro_offset
        # z_gyro = 0.0
        rospy.sleep(1)
        currState = States.WALK_WITH_BAR

        robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch","l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],["walking_module"])
        robot.walkStart()
        prev_time = time.time()
        while time.time() - prev_time < 2:
            # robot.walkVelocities(x = 5, y = 0, th=np.clip(z_gyro, -10, 10), z_move_amplitude=0.045, balance=True, z_offset=0, hip_pitch=-16.5)     #100 disks
            # robot.walkVelocities(x = 7, y = 0, th=np.clip(z_gyro, -10, 10), z_move_amplitude=0.045, balance=True, z_offset=0,hip_pitch=-10)     #60 disks
            robot.walkVelocities(x = 5, y = 0, th=np.clip(z_gyro, -10, 10), z_move_amplitude=0.045, balance=True, z_offset=0, hip_pitch=-3)     #20 disks
            print(z_gyro)
        robot.walkStop()
        rospy.sleep(1)

    elif currState == States.WALK_WITH_BAR:
        
        rospy.sleep(1)
        print("[WALK_WITH_BAR]")
        robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch",
                                      "l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],["walking_module"])
        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        robot.setJointPos(["head_tilt" , "head_pan"] , [-1.5 , 0])
        rospy.sleep(1)
        robot.walkStart()

        line_found = False
        start_walk_time = time.time()
        while not line_found and (time.time() - start_walk_time < 20.0) and not rospy.is_shutdown():
            ret, frame = cap.read()
            if ret:
                # detect whitw line 
                if is_line_detected(frame, lower_w, upper_w, threshold=8000):
                    print("see the pick up line pick the bar")
                    line_found = True
                
                # display Debug vision
                cv2.imshow('frame', frame)
                cv2.waitKey(1)
            
            # walking stage
            robot.walkVelocities(x = 5, y = 0, th=np.clip(z_gyro, -10, 10), z_move_amplitude=0.045, balance=True, z_offset=0, hip_pitch= -15)     
        
        robot.walkStop()
        currState = States.LIFT_BAR
        rospy.sleep(1)          
        
    elif currState == States.LIFT_BAR:
        print("[LIFT_BAR]")
        robot.setGeneralControlModule("none")
        rospy.sleep(2)
        robot.setGeneralControlModule("action_module")
        robot.playMotion(191, wait_for_end=True)
        time.sleep(2)
        rospy.sleep(1)
        currState = States.WALK_2_FINISH

    elif currState == States.WALK_2_FINISH:
        print("WALK_2_FINISH")
        robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch","l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],["walking_module"])
        
        prev_time2 = time.time()
        while time.time() - prev_time2 < 2:
            robot.walkStart()
            robot.walkVelocities(x = 10, y =0, th=np.clip(z_gyro, -10, 10), z_move_amplitude=0.035, balance=True, z_offset=0, hip_pitch=4.8)     #30 disks
            # robot.walkVelocities(x = 7, y = 0, th=np.clip(z_gyro, -10, 10), z_move_amplitude=0.045, balance=True, z_offset=0,hip_pitch=-10)     #60 disks
        
        finish_line_found = False
        start_finish_time = time.time()
        while not finish_line_found and (time.time() - start_finish_time < 30.0) and not rospy.is_shutdown():
            robot.walkStart()
            ret, frame = cap.read()
            if ret :
                if is_line_detected(frame, lower_w,upper_w,threshold=0000):
                    print('find finish line')
                    finish_line_found = True

                cv2.imshow('frame', frame)
                cv2.waitKey(1)    
            robot.walkVelocities(x=10,y=0, th=np.clip(z_gyro,-10,10),z_move_amplitude=0.035, balance=True, z_offset=0, hip_pitch = 5.05)

        robot.walkStop()
        currState = States.END    

    elif currState == States.END:
        print("[END]")  

        
    rate.sleep()
()