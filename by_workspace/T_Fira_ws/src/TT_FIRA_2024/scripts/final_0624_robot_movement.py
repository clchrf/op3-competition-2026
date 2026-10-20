#! /usr/bin/env python
import rospy
from op3_ros_utils import getWalkingParams, Robot
import cv2
import numpy as np
import time
import math
from scipy import optimize
from sensor_msgs.msg import Imu
from std_msgs.msg import Int32

rospy.init_node("fira_Archery_movement")

robot = Robot()

class States:
    INIT = -1
    READY = 0
    COTRRECT_POSITION = 1
    SHOOT = 2
    END = 99

z_gyro = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0

def imu_get_yaw_by_integral(data):
    global z_gyro
    global z_gyro_offset
    angular_velocity = data.angular_velocity
    z_gyro = z_gyro - angular_velocity.z / 2 + z_gyro_offset_for_caculate
    z_gyro_offset = angular_velocity.z / 2

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

data = 0

def get_data_from_Tina_Archery_vision(msg):
    global data
    data = msg.data

rospy.Subscriber("Tina_Archery_Vision", Int32, get_data_from_Tina_Archery_vision, queue_size=1)

def calc_R(xc, yc):
    """ Calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x_coords - xc) ** 2 + (y_coords - yc) ** 2)

def f_2(c):
    """ Calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    xc, yc = c  # Unpack the tuple into xc and yc
    Ri = calc_R(xc, yc)  # Call calc_R with individual coordinates
    return Ri - np.mean(Ri)

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def init():
    robot.setGeneralControlModule("action_module")
    robot.playMotion(1, wait_for_end=True)
    robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [0, -0.8])
    rospy.sleep(1.0)

rospy.sleep(1)

z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0

count = 0

currState = States.INIT

while not rospy.is_shutdown():

    current_time = time.time()

    if robot.buttonCheck("mode"):
        print("stop!")
        robot.walkStop()
        currState = States.END

    if currState == States.INIT:
        print("[INIT]")
        robot.setGeneralControlModule("action_module")
        robot.playMotion(169, wait_for_end=True)
        robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch","l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],["walking_module"])
        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        robot.setJointPos(["head_pan", "head_tilt"], [0, -0.8])
        time.sleep(1)
        currState = States.READY
        z_gyro_offset_for_caculate = z_gyro_offset
        z_gyro = 0.0    

    elif currState == States.READY:
        print("[READY]")
        if robot.buttonCheck("start"):
            detect_line = []
            robot.walkStart()
            currState = States.COTRRECT_POSITION
            start_time = current_time

    elif currState == States.COTRRECT_POSITION:
        print("[COTRRECT_POSITION]")
        
        if data == 1:
            print("walk forward")
            robot.walkStart()
            robot.walkVelocities(x=3, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            rospy.sleep(0.5)
            robot.walkStop()
            rospy.sleep(1)
        elif data == 2:
            print("walk backward")
            robot.walkStart()
            robot.walkVelocities(x=-5, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            rospy.sleep(0.5)
            robot.walkStop()
            rospy.sleep(1)
        elif data == 3:
            print("turn left")
            robot.walkStart()
            robot.walkVelocities(x=0, y=5, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            rospy.sleep(0.5)
            robot.walkStop()
            rospy.sleep(1)
        elif data == 4:
            print("turn right")
            robot.walkStart()
            robot.walkVelocities(x=0, y=-5, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            rospy.sleep(0.5)
            robot.walkStop()
            rospy.sleep(1)
        elif data == 5:
            print("init shooting position")
            robot.walkStop()
            robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
            robot.setJointPos(["head_pan", "head_tilt"], [1.4, 0])
        elif data == 6:
            print("shoot!!!")
            robot.setGeneralControlModule("action_module")
            robot.playMotion(174, wait_for_end=True)
            rospy.sleep(0.1)

            robot.setGeneralControlModule("action_module")
            robot.playMotion(176, wait_for_end=True)
            time.sleep(2)

            currState = States.END
            rospy.sleep(1)

    elif currState == States.END:
        print("[END]")
        break
  
    cv2.waitKey(1)
