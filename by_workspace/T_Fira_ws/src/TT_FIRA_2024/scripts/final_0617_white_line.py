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

currState = States.INIT

rospy.sleep(1)

z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0
count = 0

while not rospy.is_shutdown():
    ret, frame = cap.read()

    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower_white = np.array([0, 0, 210], dtype=np.uint8)
    upper_white = np.array([180, 25, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(dilated_edges, rho = 1, theta = np.pi / 180, threshold=50, minLineLength=30, maxLineGap=20)
    
    smallest_y_line = None
    smallest_y_value = float('inf')
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = float(y2 - y1) / float(x2 - x1 + 1e-6)
            
            if -0.2 < slope < 0.2:  # Check if the slope is within the horizontal range
                midpoint_y = (y1 + y2) / 2
                if midpoint_y < smallest_y_value:
                    smallest_y_value = midpoint_y
                    smallest_y_line = ((x1 + x2) / 2, midpoint_y, x2 - x1, y2 - y1)
    
    if smallest_y_line is not None:
        x_center, y_center, x_length, y_length = smallest_y_line
        x1 = int(x_center - x_length / 2)
        y1 = int(y_center - y_length / 2)
        x2 = int(x_center + x_length / 2)
        y2 = int(y_center + y_length / 2)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
        # print("Smallest y line parameters: ({}, {}, {}, {})".format(smallest_y_line[0], smallest_y_line[1], smallest_y_line[2], smallest_y_line[3]))

    cv2.imshow('Video', frame)
    cv2.waitKey(1)

    if lines is None and count > 10:
        print("no line, walk backward")
        robot.walkStart()
        robot.walkVelocities(x=-5, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
        rospy.sleep(0.2)
        robot.walkStop()
        count += 1

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
        print(smallest_y_value)
        
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
            rospy.sleep(0.2)
            robot.walkStop()
        elif smallest_y_value > 270:
            print("walk backward")
            robot.walkStart()
            robot.walkVelocities(x=-5, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            rospy.sleep(0.2)
            robot.walkStop()
        else:
            print("walk position correct")
            robot.walkStop()
            position_correct = True
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
            robot.walkStop()
            position_correct_2 = True
        elif smallest_y_value > 265:
            print("walk backward")
            robot.walkStart()
            robot.walkVelocities(x=-7, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
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
            robot.walkStop()
            turn_left_1 = True
        if slope < -0.008:
            print("turn left")
            robot.walkStart()
            robot.walkVelocities(x=0, y=3, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            robot.walkStop()
            time.sleep(1)
        elif slope > 0.008:
            print("turn right")
            robot.walkStart()
            robot.walkVelocities(x=0, y=-3, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
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
