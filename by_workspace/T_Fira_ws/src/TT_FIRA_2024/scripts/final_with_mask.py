import rospy
from op3_ros_utils import getWalkingParams, Robot
import cv2
import numpy as np
import time
import math
from scipy import optimize

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
    
# Initialize ROS node
rospy.init_node("fira_Archery_PF")

# Define robot
robot = Robot()

# Define class for states (if needed)
class States:
    INIT = -1
    READY = 0
    CLAC_ROTATION = 1
    SHOOT_STATE = 2
    END = 99

# cap = cv2.VideoCapture("Target_04.mp4")
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

start = False

# cv2.namedWindow("track_bar")
# cv2.resizeWindow("track_bar", 640,320)
# cv2.createTrackbar("hue min", "track_bar",   0 , 179, empty)
# cv2.createTrackbar("hue max", "track_bar", 179 , 179, empty)
# cv2.createTrackbar("sat min", "track_bar",   0 , 255, empty)
# cv2.createTrackbar("sat max", "track_bar", 255 , 255, empty)
# cv2.createTrackbar("val min", "track_bar",   0 , 255, empty)
# cv2.createTrackbar("val max", "track_bar", 255 , 255, empty)

while not rospy.is_shutdown():
    
    ret, img = cap.read()

    if ret:
        
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
        
        cv2.imshow("img",img)
        cv2.imshow("mask",mask_img)

    if not ret:
        break
        
    current_time = time.time()

    while start is not True:

        if robot.buttonCheck("start"):

            start = True

            robot.setGeneralControlModule("action_module")
            robot.playMotion(1, wait_for_end=True)
            time.sleep(2)

            # robot.setGeneralControlModule("action_module")
            # robot.setJointsControlModule(["l_gripper", "r_gripper"], ["none", "none"])
            # robot.setJointPos(["l_gripper"], [0])

            robot.setGeneralControlModule("action_module")
            robot.playMotion(95, wait_for_end=True)
            time.sleep(2)

            robot.setGeneralControlModule("action_module")
            robot.playMotion(96, wait_for_end=True)
            time.sleep(2)

            break

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

                    # if target_height < 250:
                    #     shoot = True
                    #     print("shoot")
                    #     print("high target : ", target_height)
                    #     robot.setGeneralControlModule("action_module")
                    #     robot.playMotion(97, wait_for_end=True)
                    #     time.sleep(2)
                    # elif target_height > 300:
                    #     shoot = True
                    #     print("shoot")
                    #     print("low target : ", target_height)
                    #     robot.setGeneralControlModule("action_module")
                    #     robot.playMotion(97, wait_for_end=True)
                    #     time.sleep(2)
                    # else:
                    #     print("shoot")
                    #     shoot = True
                    #     print("normal target : ", target_height)
                    #     robot.setGeneralControlModule("action_module")
                    #     robot.playMotion(97, wait_for_end=True)
                    #     time.sleep(2)

                    shoot = True
                    print("shoot")
                    print("high target : ", target_height)
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

    cv2.waitKey(1)