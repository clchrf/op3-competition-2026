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
#     robot.setGeneralControlModule("action_module")page
#     robot.playMotion(1, wait_for_end=True)
#     robot.setGeneralControlModule("walking_module")
#     robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
#     robot.setJointPos(["head_pan", "head_tilt"], [0, -1.2])
#     rospy.sleep(1.0)

# def calculate_distance(point1, point2):
#     x1, y1 = point1
#     x2, y2 = point2
#     return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

rospy.init_node("fira_Archery")

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

video_path = "/home/robotis/T_Fira_ws/src/TT_FIRA_2024/scripts/Target_04.mp4"

# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

delay_between_frames = 10
# delay_time = input("delay time : ")
delay_time = 2 #2 #1.8 #1.6-1.8
                                                       
tickrate = 60
rate = rospy.Rate(tickrate) 

# position_correct = None
position_correct = True

xzcurrState = States.INIT

rospy.sleep(1)

z_gyro_offset_for_caculate = z_gyro_offset

z_gyro = 0.0

robot.setGeneralControlModule("action_module")
robot.playMotion(202, wait_for_end=True)
time.sleep(2)

# robot.setGeneralControlModule("walking_module")
robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
robot.setJointPos(["head_pan", "head_tilt"], [1.6, 0])
rospy.sleep(1.0)

detected_circle_positions = []

start_time = time.time()
current_time = start_time
prev_time = start_time

prev_angle = None
angular_velocity = None
angular_velocity_history = []

start = False
shoot = False

while not rospy.is_shutdown():
#video looping test
# while True:
    
    ret, img = cap.read()

    if not ret:
        break

    #video looping test
    # if not ret:
       
    #     cap = cv2.VideoCapture(video_path)
    #     continue
        
    current_time = time.time()

    while start is not True:
        # cv2.imshow("img", img) 
        if robot.buttonCheck("start"):
            start = True
            start_time = time.time()
            break

    img_blur = cv2.GaussianBlur(img, (3, 3), 5)
    img = cv2.GaussianBlur(img_blur, (5, 5), 5)
    hsv_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    lower_red = np.array([162, 218, 191])
    upper_red = np.array([179, 249, 207])

    yellow_lower = np.array([10, 177, 219])
    yellow_upper = np.array([51, 249, 255])
 
    # blue_lower = np.array([100, 150, 100])
    # blue_upper = np.array([130, 255, 255])

    mask_red = cv2.inRange(hsv_img, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    # mask_blue = cv2.inRange(hsv_img, blue_lower, blue_upper)

    combined_mask = cv2.add(mask_red, mask_yellow)
    # combined_mask = cv2.add(combined_mask, mask_blue)

    _, contours_red, _ = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_yellow, _ = cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # _, contours_blue, _ = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours_red, -1, (0, 0, 255), 2)
    cv2.drawContours(img, contours_yellow, -1, (0, 255, 255), 2)
    # cv2.drawContours(img, contours_blue, -1, (255, 0, 0), 2)

    img_blurred = cv2.GaussianBlur(combined_mask, (7, 7), 0)
    img_blurred = cv2.medianBlur(img_blurred, 5)
    img_blurred = cv2.blur(img_blurred, (5, 5))

    def calculate_center(mask):
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None

    center_red = calculate_center(mask_red)
    center_yellow = calculate_center(mask_yellow)
    # center_blue = calculate_center(mask_blue)

    img_blurred = cv2.GaussianBlur(combined_mask, (7, 7), 1)
    img_blurred = cv2.medianBlur(img_blurred, 5)
    img_blurred = cv2.blur(img_blurred, (5, 5))

    detected_circles = cv2.HoughCircles(
        img_blurred,
        cv2.HOUGH_GRADIENT,
        1,
        1000,
        param1=30,
        param2=15,
        minRadius=1,
        maxRadius=70
    )

    circle_mask = np.zeros_like(combined_mask)

# =====================================test======================================================
    
    # if current_time - start_time > 5:
    #     cv2.circle(img, (new_x, new_y), r, (255, 255, 255), 3)
    #     cv2.circle(mask_img, (new_x, new_y), r, (255, 255, 255), 3)

    #     shoot = True
    #     print("shoot")

    #     robot.setGeneralControlModule("action_module")
    #     robot.playMotion(189, wait_for_end=True)

    #     currState = States.END
    #     rospy.sleep(1)

# =====================================test======================================================

    # print("shoot")

    # robot.setGeneralControlModule("action_module")

    # robot.playMotion(203, wait_for_end=True)

    # break
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for i in detected_circles[0, :]:
            a, b, r = i[0], i[1], i[2]
            
            detected_circle_positions.append((a, b))

            cv2.circle(img, (a, b), r, (0, 255, 0), 3)
            cv2.circle(img, (a, b), 1, (0, 100, 100), 3)

            cv2.circle(circle_mask, (a, b), r, 255, -1)

    filtered_mask = cv2.bitwise_and(combined_mask, combined_mask, mask=circle_mask)

    filtered_img = cv2.bitwise_and(img, img, mask=circle_mask)

    # cv2.imshow("Filtered Mask", filtered_mask)
    # cv2.imshow("Filtered Image", filtered_img)

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

                distance = calculate_distance((new_x, new_y), (center_x, (center_y + radius)))
                target_height = center_y + radius
                # print(distance)
                # cv2.circle(img, (int(center_x), int(center_y + radius)), r, (255, 255, 255), 3)
                if (distance <= 5 and current_time - start_time > 10) or current_time - start_time > 30:
                    cv2.circle(filtered_img, (new_x, new_y), r, (255, 255, 255), 3)
                    
                    if center_red and center_yellow:
                        distance_red_yellow = calculate_distance(center_red, center_yellow)
                        
                        if distance_red_yellow < 5:
                            print("All centers overlapped. Shoot")
                            shoot = True
                            robot.setGeneralControlModule("action_module")
                            # robot.playMotion(203, wait_for_end=True) #lowest
                            # robot.playMotion(209, wait_for_end=True) #lower
                            # robot.playMotion(208, wait_for_end=True) #low
                            # robot.playMotion(207, wait_for_end=True) #medium
                            # robot.playMotion(206, wait_for_end=True) #taller
                            robot.playMotion(205, wait_for_end=True) #tall

                    currState = States.END
                    rospy.sleep(1)

            prev_time = current_time
            prev_angle = angle

    cv2.imshow("img", img) 
    # cv2.imshow("mask", combined_mask)

    key = cv2.waitKey(delay_between_frames)
    # if current_time - start_time > 30:
    if shoot is True:
            break

    cv2.waitKey(1)