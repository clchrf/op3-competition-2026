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
    z_gyro = z_gyro - angular_velocity.z/2 + z_gyro_offset_for_caculate
    z_gyro_offset = angular_velocity.z/2

rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_get_yaw_by_integral, queue_size=1)

def empty(v):
    pass

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

# def init():
#     robot.setGeneralControlModule("action_module")
#     robot.playMotion(1, wait_for_end=True)
#     robot.setGeneralControlModule("walking_module")
#     robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
#     robot.setJointPos(["head_pan", "head_tilt"], [0, -1.2])
#     rospy.sleep(1.0)

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

rospy.init_node("fira_Archery")

robot = Robot()

class States:
    INIT = -1
    READY = 0
    WALK_TO_THE_MIDDLE = 1
    WALK_TO_THE_MIDDLE_2 = 2
    CORRECT_ANGLE = 3
    INIT_SHOOTING_POSITION = 4
    INIT_SHOOTING_POSITION = 5
    AIM_AND_SHOOT = 5
    END = 99

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0)

delay_between_frames = 10
# delay_time = input("delay time : ")
delay_time = 1.8

# tickrate = 60
# rate = rospy.Rate(tickrate)

# position_correct = None
position_correct = True

currState = States.INIT

rospy.sleep(1)

z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0

# robot.setGeneralControlModule("action_module")
# robot.playMotion(169, wait_for_end=True)
# time.sleep(2)

# # robot.setGeneralControlModule("walking_module")
# robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
# robot.setJointPos(["head_pan", "head_tilt"], [1.4, 0])
# rospy.sleep(1.0)

detected_circle_positions = []

start_time = time.time()
current_time = start_time
prev_time = start_time

prev_angle = None
angular_velocity = None
angular_velocity_history = []

start = False
shoot = []
position_correct = []
position_correct_2 = []
position_correct_3 = []
turn_left_1 = []
count = 0

# cv2.namedWindow("track_bar")
# cv2.resizeWindow("track_bar", 640,320)
# cv2.createTrackbar("hue min", "track_bar",   0 , 179, empty)
# cv2.createTrackbar("hue max", "track_bar", 179 , 179, empty)
# cv2.createTrackbar("sat min", "track_bar",   0 , 255, empty)
# cv2.createTrackbar("sat max", "track_bar", 255 , 255, empty)
# cv2.createTrackbar("val min", "track_bar",   0 , 255, empty)
# cv2.createTrackbar("val max", "track_bar", 255 , 255, empty)

while not rospy.is_shutdown():
# while True:
    ret, img = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
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
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
        # print("Smallest y line parameters: ({}, {}, {}, {})".format(smallest_y_line[0], smallest_y_line[1], smallest_y_line[2], smallest_y_line[3]))

    cv2.imshow('Video', img)
    cv2.waitKey(1)

    if currState == States.INIT:
        print("[INIT]")
        # init()
        robot.setGeneralControlModule("action_module")
        robot.playMotion(169, wait_for_end=True)
        robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch","l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],["walking_module"])
        robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
        robot.setJointPos(["head_pan", "head_tilt"], [0, -0.8])
        time.sleep(1)
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
            time.sleep(1)
        if smallest_y_value < 260:
            print("walk forward")
            robot.walkStart()
            robot.walkVelocities(x=3, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            time.sleep(0.2)
            robot.walkStop()
        elif smallest_y_value > 270:
            print("walk backward")
            robot.walkStart()
            robot.walkVelocities(x=-5, y=0, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            time.sleep(0.2)
            robot.walkStop()
        else:
            print("walk position correct")
            robot.walkStop()
            position_correct = True
            time.sleep(1)

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
            time.sleep(1)
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
            time.sleep(1)
        
    elif currState == States.CORRECT_ANGLE:
        print("[CORRECT_ANGLE]")
        print(slope)
        
        if turn_left_1 is not True:
            print("turn left 1")
            robot.walkStart()
            robot.walkVelocities(x=0, y=3, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            robot.walkStop()
            turn_left_1 = True
        if slope < -0.05:
            print("turn left")
            robot.walkStart()
            robot.walkVelocities(x=0, y=3, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            robot.walkStop()
        elif slope > 0.05:
            print("turn right")
            robot.walkStart()
            robot.walkVelocities(x=0, y=-3, th=np.clip(z_gyro, -15, 15), z_move_amplitude=0.045, balance=True, z_offset=0)
            robot.walkStop()
        else:
            print("angle position correct")
            position_correct_3 = True
            time.sleep(1)

        if position_correct_3 == True:
            robot.walkStop()
            currState = States.END
            time.sleep(1)

    elif currState == States.END:
        print("[END]")
        break
    
rospy.sleep(1)
time.sleep(1)
start_time = time.time()

currState = States.AIM_AND_SHOOT
 # robot.setGeneralControlModule("walking_module")
robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
robot.setJointPos(["head_pan", "head_tilt"], [1.4, 0])
rospy.sleep(1.0)

while True:

    ret, img = cap.read()

    if not ret:
        break
        
    current_time = time.time()

    # while start is not True:
    #     if robot.buttonCheck("start"):
    #         start = True
    #         start_time = time.time()
    #         break

    img_blur = cv2.GaussianBlur(img, (3,3) , 5)
    img = cv2.GaussianBlur(img_blur, (5,5) , 5)
    hsv_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    # h_min = cv2.getTrackbarPos("hue min", "track_bar")
    # h_max = cv2.getTrackbarPos("hue max", "track_bar")
    # s_min = cv2.getTrackbarPos("sat min", "track_bar")
    # s_max = cv2.getTrackbarPos("sat max", "track_bar")
    # v_min = cv2.getTrackbarPos("val min", "track_bar")
    # v_max = cv2.getTrackbarPos("val max", "track_bar")
    h_min, h_max, s_min, s_max, v_min, v_max = 0, 10, 100, 255, 100, 255
    # print(h_min, h_max, s_min, s_max, v_min, v_max )
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])

    mask_img = cv2.inRange(hsv_img, lower, upper)
    
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    img_blurred = cv2.GaussianBlur(mask_img, (7, 7), 0)
    img_blurred = cv2.GaussianBlur(mask_img, (7, 7), 0)
    img_blurred = cv2.medianBlur(img_blurred, 5)
    img_blurred = cv2.blur(img_blurred, (5, 5))
    img_blurred = cv2.blur(img_blurred, (5, 5))
    # img_blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    # detected_circles = cv2.HoughCircles(img_blurred, cv2.HOUGH_GRADIENT,
    #                                     1, 1000, param1=50, param2=30,
    #                                     minRadius=1, maxRadius=25)
    detected_circles = cv2.HoughCircles(
        img_blurred, 
        cv2.HOUGH_GRADIENT, 
        1, 
        1000, 
        param1=50, 
        param2=20,  # Lowering param2 to make detection less precise
        minRadius=1, 
        maxRadius=50  # Increasing maxRadius to detect larger circles
    )

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for i in detected_circles[0, :]:
            a, b, r = i[0], i[1], i[2]
            
            detected_circle_positions.append((a, b))

            cv2.circle(img, (a, b), r, (0, 255, 0), 3)
            cv2.circle(img, (a, b), 1, (0, 100, 100), 3)

            cv2.circle(mask_img, (a, b), r, (0, 255, 0), 3)
            cv2.circle(mask_img, (a, b), 1, (0, 100, 100), 3)

    if len(detected_circle_positions) >= 3:
        x_coords, y_coords = zip(*detected_circle_positions)

        center_estimate = (np.mean(x_coords), np.mean(y_coords))  # Estimate initial center
        center, ier = optimize.leastsq(f_2, center_estimate)

        xc, yc = center

        Ri_2 = calc_R(*center)
        R_2 = Ri_2.mean()

        radius = np.mean(R_2)
        center_x, center_y = xc, yc

        if current_time - start_time > 1:
            cv2.circle(img, (int(center_x), int(center_y)), int(radius), (255, 255, 255), 3)
            cv2.circle(img, (int(center_x), int(center_y)), 3, (0, 255, 0), 3)
            cv2.circle(mask_img, (int(center_x), int(center_y)), int(radius), (255, 255, 255), 3)
            cv2.circle(mask_img, (int(center_x), int(center_y)), 3, (0, 255, 0), 3)

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

                angle_change = avg_angular_velocity * float(delay_time)
                new_angle = prev_angle + angle_change
                new_x = int(center_x + radius * np.cos(new_angle))
                new_y = int(center_y + radius * np.sin(new_angle))

                cv2.circle(img, (new_x, new_y), r, (0, 0, 255), 3)
                cv2.circle(mask_img, (new_x, new_y), r, (0, 0, 255), 3)

                distance = calculate_distance((new_x, new_y), (center_x, (center_y + radius)))
                target_height = center_y + radius
                # print(distance)
                # cv2.circle(img, (int(center_x), int(center_y + radius)), r, (255, 255, 255), 3)

                if distance <= 5 and current_time - start_time > 7:
                    cv2.circle(img, (new_x, new_y), r, (255, 255, 255), 3)
                    cv2.circle(mask_img, (new_x, new_y), r, (255, 255, 255), 3)

                    shoot = True
                    print("shoot")

                    robot.setGeneralControlModule("action_module")
                    robot.playMotion(174, wait_for_end=True)

                    robot.setGeneralControlModule("action_module")
                    robot.playMotion(176, wait_for_end=True)
                    time.sleep(2)

                    currState = States.END
                    rospy.sleep(1)

            prev_time = current_time
            prev_angle = angle

    cv2.imshow("img",img)
    cv2.imshow("mask",mask_img)

    key = cv2.waitKey(delay_between_frames)
    # if current_time - start_time > 30:
    if shoot is True:
        break

    cv2.waitKey(1)