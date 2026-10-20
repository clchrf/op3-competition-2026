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

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r'/home/robotis/brian_ws/src/TT_FIRA_2024/scripts/Target_04.mp4')

delay_between_frames = 10
# delay_time = input("delay time : ")
delay_time = 2.3

tickrate = 60
rate = rospy.Rate(tickrate)

# position_correct = None
position_correct = True

xzcurrState = States.INIT

rospy.sleep(1)

z_gyro_offset_for_caculate = z_gyro_offset

z_gyro = 0.0

robot.setGeneralControlModule("action_module")
robot.playMotion(200 , wait_for_end=True)
time.sleep(2)

robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
robot.setJointPos(["head_pan", "head_tilt"], [1.4, 0])
rospy.sleep(1.0)

detected_circle_positions = []

start_time = time.time()
current_time = start_time
prev_time = start_time

prev_angle = None
angular_velocity = None
angular_velocity_history = []

aim = False
start = False
shoot = []
stable_count = 0
over_time =  False

# position checking
def move_head_to_center(center_x, img_shape, robot):

    w = img_shape[1]
    print(w)
    center_screen = w // 2
    dx = center_x - center_screen
    # Assume 1 pixel = 0.002 radians, adjust as needed for your camera
    k = 0.01
    current_angle_V = np.degrees(float(robot.joint_pos["head_tilt"]))
    current_angle_H = np.degrees(float(robot.joint_pos["head_pan"]))
    target_angle = current_angle_H - dx * k
    print("target angle:%.2f, dx:%.2f. " % (target_angle, dx))
     
    # ------------head angle test------------
    target_angle = 1.4 
    # ------------head angle test------------

    # Clamp the head angle to safe limits
    # target_angle = math.radians(target_angle)
    
    robot.setJointPos(["head_pan"], [target_angle])
    return target_angle

def move_body_to_center(head_angle, robot):
    # Tune gain for your robot
    gain = 20
    # Clamp angle to safe range
    head_angle = max(-1.5, min(1.5, head_angle))
    # Decide movement direction and velocity
    if (abs(head_angle) - 1.4) > math.radians(3):
        if head_angle < 1.4:
            direction = "forward"
            x_velocity = 8
            theta = 10
        else:
            direction = "backward"
            x_velocity = -20
            theta = 3
        # x_velocity = np.clip(x_velocity, -20, 20)
        print("Body adjustment: Move %s (head_angle=%.2f deg, x_velocity=%.2f)" % (direction, math.degrees(head_angle), x_velocity))
        robot.walkVelocities(x=x_velocity, y=-20, th=theta, z_move_amplitude=0.045, balance=True, z_offset=0)
    else:
        print("Head angle is centered. No body movement.")
        robot.walkVelocities(x=0, y=0, th=0, z_move_amplitude=0.035, balance=True, z_offset=0)

while not rospy.is_shutdown():
# while True:
    # robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch",
    #                 "l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],
    #                 ["walking_module"])
    # robot.walkStart()
    ret, img = cap.read()

    if not ret:
        break

    # ------------video looping test------------
    # if not ret:
    #     cap = cv2.VideoCapture(r'/home/robotis/brian_ws/src/TT_FIRA_2024/scripts/Target_04.mp4')
    #     continue
    # ------------video looping test------------
    if start == True:
        current_time = time.time()

    while start is not True:
        if robot.buttonCheck("start"):
            start = True
            start_time = time.time()
            break

    if start == True & robot.buttonCheck("start"):
        start = False
    
    img_blur = cv2.GaussianBlur(img, (3, 3), 5)
    img = cv2.GaussianBlur(img_blur, (5, 5), 5)
    hsv_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([50, 255, 255])

    blue_lower = np.array([100, 150, 100])
    blue_upper = np.array([130, 255, 255])

    mask_red = cv2.inRange(hsv_img, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    mask_blue = cv2.inRange(hsv_img, blue_lower, blue_upper)

    combined_mask = cv2.add(mask_red, mask_yellow)
    combined_mask = cv2.add(combined_mask, mask_blue)

    _, contours_red, _ = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_yellow, _ = cv2.findContours(mask_yellow.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_blue, _ = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours_red, -1, (0, 0, 255), 2)
    cv2.drawContours(img, contours_yellow, -1, (0, 255, 255), 2)
    cv2.drawContours(img, contours_blue, -1, (255, 0, 0), 2)

    img_blurred = cv2.GaussianBlur(combined_mask, (7, 7), 0)
    img_blurred = cv2.medianBlur(img_blurred, 5)
    img_blurred = cv2.blur(img_blurred, (5, 5))

    def calculate_center(mask):
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None

    center_red = calculate_center(mask_red)
    center_yellow = calculate_center(mask_yellow)
    center_blue = calculate_center(mask_blue)

    img_blurred = cv2.GaussianBlur(combined_mask, (7, 7), 1)
    img_blurred = cv2.medianBlur(img_blurred, 5)
    img_blurred = cv2.blur(img_blurred, (5, 5))

    detected_circles = cv2.HoughCircles(
        img_blurred,
        cv2.HOUGH_GRADIENT,
        1,
        1000,
        param1=50,
        param2=20,
        minRadius=1,
        maxRadius=50
    )

    circle_mask = np.zeros_like(combined_mask)

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
        center_estimate = (np.mean(x_coords), np.mean(y_coords))
        center, ier = optimize.leastsq(f_2, center_estimate)
        xc, yc = center
        Ri_2 = calc_R(*center)
        R_2 = Ri_2.mean()
        radius = np.mean(R_2)
        center_x, center_y = xc, yc
        

        if current_time - start_time > 8 & aim == False:
            cv2.circle(img, (int(center_x), int(center_y)), int(radius), (255, 255, 255), 3)
            cv2.circle(img, (int(center_x), int(center_y)), 3, (0, 255, 0), 3)

            if stable_count < 3:
                head_angle = move_head_to_center(center_x, img.shape, robot)
                head_angle_deg = math.degrees(head_angle)
                wait_time = abs(math.tan(head_angle_deg - 80)) * 3
                print("Head pan angle: %.2f degrees" % (head_angle_deg))
                
                # ------------aiming---------------
                # print("counting")
                if head_angle_deg > 81:
                    print("Head angle > 81 deg: Move body left")
                    move_body_to_center(head_angle, robot)

                    # ------------test---------------
                    robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch",
                        "l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],
                        ["walking_module"])
                    robot.walkStart()
                    # ------------test---------------

                    stable_count = stable_count + 1
                    # cv2.waitKey(15)
                    rospy.sleep(wait_time)
                    robot.walkStop()
                    rospy.sleep(3)
                    print("Stability count: %s" % (stable_count))

                elif head_angle_deg < 79:
                    print("Head angle < 79 deg: Move body right")
                    move_body_to_center(head_angle, robot)

                    # ------------test---------------
                    robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch",
                        "l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],
                        ["walking_module"])
                    robot.walkStart()
                    # ------------test---------------

                    stable_count = stable_count + 1
                    # cv2.waitKey(15)
                    rospy.sleep(wait_time)
                    robot.walkStop()
                    rospy.sleep(3)
                    print("Stability count: %s" % (stable_count))
                else:
                    # Head is centered, count stability
                    rospy.sleep(3)
                    stable_count = stable_count + 1
                    print("Head centered. Stability count: %s" % (stable_count))
                    print("Proceed to arrow prediction and shooting.")
                # ------------aiming---------------

            elif aim == False:
                aim = True
                print("Head is stable and centered. Executing action 202.")
                
                # ------------test---------------
                robot.setGeneralControlModule("action_module")
                robot.playMotion(202, wait_for_end=True)
                rospy.sleep(1.0)
                robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
                robot.setJointPos(["head_pan", "head_tilt"], [1.4, 0])
                rospy.sleep(10)
                # ------------test---------------
            
    if current_time - start_time > 30:
        over_time = True    

    if len(detected_circle_positions) > 3 & aim == True:
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

                if (distance <= 5 and current_time - start_time > 15) or over_time is True:
                    cv2.circle(img, (new_x, new_y), r, (255, 255, 255), 3)
                    
                    if center_red and center_yellow and center_blue:
                        distance_red_yellow = calculate_distance(center_red, center_yellow)
                        
                        if distance_red_yellow < 5 or over_time is True:
                            print("All centers overlapped. Shoot")
                            shoot = True
                            # ------------test---------------
                            robot.setGeneralControlModule("action_module")
                            robot.playMotion(203, wait_for_end=True)
                            # ------------test---------------

                    currState = States.END
                    rospy.sleep(1)

            prev_time = current_time
            prev_angle = angle

    cv2.imshow("img", img) 
    cv2.imshow("mask", combined_mask)

    # key = cv2.waitKey(delay_between_frames)
    
    # ------------test---------------
    # if current_time - start_time > 30:
    if shoot is True:
        break
    # ------------test---------------

    cv2.waitKey(1)