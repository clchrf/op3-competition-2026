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

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x_coords-xc)**2 + (y_coords-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    xc, yc = c  # Unpack the tuple into xc and yc
    Ri = calc_R(xc, yc)  # Call calc_R with individual coordinates
    return Ri - np.mean(Ri)
    
def init():
    robot.setGeneralControlModule("action_module")
    robot.playMotion(1, wait_for_end=True)
    robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [0, -1.2])
    rospy.sleep(1.0)

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

rospy.init_node("fira_Archery_PF")

robot = Robot()

class States:
    INIT = -1
    READY = 0 
    WALK_FORWARD = 1
    CORRECT_ANGLE = 2
    TURN_AROUND = 3
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

shoot = []

tickrate = 60
rate = rospy.Rate(tickrate)

# position_correct = None
position_correct = True

currState = States.INIT

rospy.sleep(1)

z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0
    
robot.setGeneralControlModule("walking_module")
robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
robot.setJointPos(["head_pan", "head_tilt"], [1.2, 0])
rospy.sleep(1.0)

robot.setGeneralControlModule("action_module")
robot.playMotion(170, wait_for_end=True)
time.sleep(2)

while not rospy.is_shutdown():

    current_time = time.time()

    ret, img = cap.read()
    if not ret:
        break
    
    current_time = time.time()

    if position_correct is not True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])

        # lower_yellow = np.array([22, 69, 128])
        # upper_yellow = np.array([38, 255, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)
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
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

                slope = float(y2 - y1) / float(x2 - x1)
                # print(slope)

    if position_correct is True:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        gray_blurred = cv2.medianBlur(gray_blurred, 5)
        gray_blurred = cv2.blur(gray_blurred, (5, 5))
        # gray_blurred = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT,
                                            1, 1000, param1=50, param2=30,
                                            minRadius=15, maxRadius=35)

        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))

            for i in detected_circles[0, :]:
                a, b, r = i[0], i[1], i[2]
                
                detected_circle_positions.append((a, b))

                cv2.circle(img, (a, b), r, (0, 255, 0), 3)
                cv2.circle(img, (a, b), 1, (0, 100, 100), 3)

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

    cv2.imshow("show image", img)
    cv2.waitKey(1)
    
    if current_time - start_time > 30:
            print("time's up!")
            robot.setGeneralControlModule("action_module")
            robot.playMotion(173, wait_for_end=True)
            time.sleep(2)
            
            currState = States.END

    if currState == States.INIT:
        print("[INIT]")
        init()

        currState = States.READY

    elif currState == States.READY:
        print("[READY]")

        if robot.buttonCheck("start"):
            
            currState = States.CALCULATE_TARGET
            
        
    elif currState == States.CALCULATE_TARGETE:
        print("[CALCULATE_PREDICTED_TARGET]")
        
        if len(detected_circle_positions) > 3:
            a, b = detected_circle_positions[-1]
            displacement_vector = (a - center_x, b - center_y)

            angle = np.arctan2(displacement_vector[1], displacement_vector[0])

            current_time = time.time()
            elapsed_time = current_time - prev_time

            if current_time - start_time > 15:
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

                    distance = calculate_distance((new_x, new_y), (center_x, (center_y + radius)))
                    # print(distance)
                    # cv2.circle(img, (int(center_x), int(center_y + radius)), r, (255, 255, 255), 3)

                    if distance <= 5:
                        cv2.circle(img, (new_x, new_y), r, (255, 255, 255), 3)
                        if current_time - start_time > 10:
                            currState = States.SHOOT
                            rospy.sleep(2)

                prev_time = current_time
                prev_angle = angle
            
    elif currState == States.SHOOT:
        print("[SHOOT]")
        
        robot.setGeneralControlModule("action_module")
        robot.playMotion(173, wait_for_end=True)
        time.sleep(2)

        currState = States.END
        rospy.sleep(1)

    elif currState == States.END:
        print("[END]")
        break

    rate.sleep()
