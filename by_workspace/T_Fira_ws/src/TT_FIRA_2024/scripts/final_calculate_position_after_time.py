import cv2
import numpy as np
import time
from scipy import optimize

def calc_R(xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x_coords-xc)**2 + (y_coords-yc)**2)

def f_2(c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    xc, yc = c  # Unpack the tuple into xc and yc
    Ri = calc_R(xc, yc)  # Call calc_R with individual coordinates
    return Ri - np.mean(Ri)

cap = cv2.VideoCapture("Target_04.mp4")

delay_between_frames = 20
delay_time = 1

detected_circle_positions = []

start_time = time.time()
current_time = start_time
prev_time = start_time

prev_angle = None
angular_velocity = None
angular_velocity_history = []

draw_angle = []

while True:
    ret, img = cap.read()
    if not ret:
        break
    
    current_time = time.time()

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

    if len(detected_circle_positions) >= 3:
        x_coords, y_coords = zip(*detected_circle_positions)

        center_estimate = (np.mean(x_coords), np.mean(y_coords)) 
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

                angle_change = avg_angular_velocity * float(delay_time)
                new_angle = prev_angle + angle_change
                new_x = int(center_x + radius * np.cos(new_angle))
                new_y = int(center_y + radius * np.sin(new_angle))

                cv2.circle(img, (new_x, new_y), r, (0, 0, 255), 3)

            prev_time = current_time
            prev_angle = angle

    cv2.imshow("show image", img)
    
    key = cv2.waitKey(delay_between_frames)

    cv2.waitKey(1)
