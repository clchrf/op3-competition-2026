import rospy
from op3_ros_utils import getWalkingParams, Robot
import cv2
import numpy as np
import time
import math
from scipy import optimize
from sensor_msgs.msg import Imu

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

def calc_R(xc, yc):
    """ Calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x_coords - xc) ** 2 + (y_coords - yc) ** 2)

def f_2(c):
    """ Calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    xc, yc = c  # Unpack the tuple into xc and yc
    Ri = calc_R(xc, yc)  # Call calc_R with individual coordinates
    return Ri - np.mean(Ri)

def init():
    robot.setGeneralControlModule("action_module")
    robot.playMotion(1, wait_for_end=True)
    robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [0, -0.8])
    rospy.sleep(1.0)

rospy.init_node("fira_Archery")

robot = Robot()

class States:
    INIT = -1
    READY = 0
    WALK_TO_THE_MIDDLE = 1
    WALK_TO_THE_MIDDLE_2 = 2
    CORRECT_ANGLE = 3
    INIT_SHOOTING_POSITION = 4
    CORRECT_POSITION = 5
    CALCULATE_TARGET = 6
    END = 99

cap = cv2.VideoCapture(0)

delay_between_frames = 10
delay_time = 0.5

detected_circle_positions = []

start_time = time.time()
current_time = start_time
prev_time = start_time

prev_angle = None
angular_velocity = None
angular_velocity_history = []

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

shoot = []

tickrate = 60
rate = rospy.Rate(tickrate)

position_correct = []
position_correct_2 = []
position_correct_3 = []
turn_left_1 = []
count = 0
Sleep_and_Stop_Waliking = False

currState = States.INIT

rospy.sleep(1)

z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0
count = 0
count_sleep = 0


while not rospy.is_shutdown():
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if Sleep_and_Stop_Waliking == True:
        robot.walkStop()
        if count_sleep == 10:
            Sleep_and_Stop_Waliking = False
            count_sleep = 0
        continue

    if currState == States.INIT:
        print("[INIT]")
        init()
        currState = States.READY

    elif currState == States.READY:
        print("[READY]")
        if robot.buttonCheck("start"):
            detect_line = []
            robot.walkStart()
            currState = States.WALK_TO_THE_MIDDLE
            start_time = current_time

    elif currState == States.WALK_TO_THE_MIDDLE:
        print("[WALK_TO_THE_MIDDLE]")
        
        if robot.buttonCheck("mode"):
            print("stop!")
            robot.walkStop()
            currState = States.END
        if position_correct == True:
            robot.walkStop()
            currState = States.WALK_TO_THE_MIDDLE_2
            rospy.sleep(2)
        if smallest_y_value < 260:
            print("walk forward")
            robot.walkStart()
            robot.walkVelocities(x=3, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            Sleep_and_Stop_Waliking = True
            robot.walkStop()
        elif smallest_y_value > 270:
            print("walk backward")
            robot.walkStart()
            robot.walkVelocities(x=-5, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            Sleep_and_Stop_Waliking = True
            robot.walkStop()
        else:
            print("walk position correct")
            robot.walkStop()
            position_correct = True
            Sleep_and_Stop_Waliking = True
            rospy.sleep(1)

    elif currState == States.WALK_TO_THE_MIDDLE_2:
        print("[WALK_TO_THE_MIDDLE_2]")
        print(smallest_y_value)
        
        if robot.buttonCheck("mode"):
            print("stop!")
            robot.walkStop()
            currState = States.END
        if position_correct_2 == True:
            robot.walkStop()
            currState = States.CORRECT_ANGLE
            rospy.sleep(2)
        if smallest_y_value < 265:
            print("walk forward")
            robot.walkStart()
            robot.walkVelocities(x=5, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            Sleep_and_Stop_Waliking = True
            robot.walkStop()
            position_correct_2 = True
        elif smallest_y_value > 265:
            print("walk backward")
            robot.walkStart()
            robot.walkVelocities(x=-7, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            Sleep_and_Stop_Waliking = True
            robot.walkStop()
            position_correct_2 = True
        else:
            print("walk position correct")
            robot.walkStop()
            position_correct_2 = True
            rospy.sleep(1)
        
    elif currState == States.CORRECT_ANGLE:
        print("[CORRECT_ANGLE]")
        print(slope)
        
        if turn_left_1 is not True:
            print("turn left 1")
            robot.walkStart()
            robot.walkVelocities(x=0, y=3, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            Sleep_and_Stop_Waliking = True
            robot.walkStop()
            turn_left_1 = True
        if slope < -0.008:
            print("turn left")
            robot.walkStart()
            robot.walkVelocities(x=0, y=3, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            Sleep_and_Stop_Waliking = True
            robot.walkStop()
            time.sleep(1)
        elif slope > 0.008:
            print("turn right")
            robot.walkStart()
            robot.walkVelocities(x=0, y=-3, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            Sleep_and_Stop_Waliking = True
            robot.walkStop()
            time.sleep(1)
        else:
            print("angle position correct")
            position_correct_3 = True
            time.sleep(1)

        if position_correct_3 == True:
            robot.walkStop()
            currState = States.END
            rospy.sleep(2)

    elif currState == States.END:
        print("[END]")
        break

    cv2.waitKey(1)

    key = cv2.waitKey(delay_between_frames)
    if key == ord("q") or current_time - start_time > 10:
        break

cap.release()
cv2.destroyAllWindows()
