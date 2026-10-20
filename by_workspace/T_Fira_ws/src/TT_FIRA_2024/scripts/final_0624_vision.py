#! /usr/bin/env python
import rospy
import cv2
import numpy as np
import time
import math
from scipy import optimize
from std_msgs.msg import Int32

pub = rospy.Publisher("Tina_Archery_Vision", Int32, queue_size=1)
rospy.init_node("fira_Archery_vision")

class States:
    INIT = -1
    READY = 0
    WALK_TO_THE_MIDDLE = 1
    CORRECT_ANGLE = 3
    INIT_SHOOTING_POSITION = 4
    AIM_AND_SHOOT = 5
    END = 99

cap = cv2.VideoCapture(0)

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

delay_time = 1.6

detected_circle_positions = []

start_time = time.time()
current_time = start_time
prev_time = start_time

prev_angle = None
angular_velocity = None
angular_velocity_history = []
data = 0

rospy.sleep(1)

currState = States.WALK_TO_THE_MIDDLE
position_correct = False
position_correct_2 = False

while not rospy.is_shutdown():
    
    data = 0
    ret, frame = cap.read()

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

    img = cv2.GaussianBlur(frame, (3,3) , 5)
    img_blur_2 = cv2.GaussianBlur(img, (5,5) , 5)
    hsv_img = cv2.cvtColor(img_blur_2, cv2.COLOR_BGR2HSV)
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
                    data = 6
                    pub.publish(data)

                    currState = States.END
                    rospy.sleep(1)

            prev_time = current_time
            prev_angle = angle

    cv2.imshow('Video2', img)
    cv2.imshow('Video3', mask_img)
    cv2.waitKey(1)

    if currState == States.WALK_TO_THE_MIDDLE:
        print("[WALK_TO_THE_MIDDLE]")
        print("smallest_y_value",smallest_y_value)

        if smallest_y_value < 260:
            print("walk forward")
            data = 1
        elif smallest_y_value > 270:
            print("walk backward")
            data = 2
        else:
            print("walk position correct")
            position_correct = True

        if position_correct == True:
            currState = States.CORRECT_ANGLE
        
        pub.publish(data)

    elif currState == States.CORRECT_ANGLE:
        print("[CORRECT_ANGLE]")
        print("slope",slope)
        
        if slope < -0.008:
            print("turn left")
            data = 3
            print(data)
        elif slope > 0.008:
            print("turn right")
            data = 4
            print(data)
        else:
            print("angle position correct")
            position_correct_2 = True

        if position_correct_2 == True:
            currState = States.INIT_SHOOTING_POSITION
            
        pub.publish(data)

    elif currState == States.INIT_SHOOTING_POSITION:
        print("[INIT_SHOOTING_POSITION]")
        data = 5
        pub.publish(data)
        currState = States.AIM_AND_SHOOT

    elif currState == States.END:
        print("[END]")
        break

    # cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
