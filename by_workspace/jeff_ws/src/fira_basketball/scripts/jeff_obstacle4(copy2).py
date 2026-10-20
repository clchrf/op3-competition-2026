#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import rospy
from op3_ros_utils import *
from std_msgs.msg import Float32 , Int8
from sensor_msgs.msg import Image
import time
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
import math
from cv_bridge import CvBridge, CvBridgeError
from fira_basketball.msg import ThreeDouble


DEBUG_MODE = True
DEBUG_MODE = False # Show the detected image True
DEBUG_MARKER = False
DL_MODE = False 


argsIMU = [68,True]
camera_ang = 60
camera_h = 44.0
hfov = 80.54
vfov = 52.1
# hfov = 53.0
# vfov = 38.0#41.2794
hfov_rad = np.radians(hfov)
vfov_rad = np.radians(vfov)

pass_time = time.time()

oimg = []

IMU_RECORD = True
z_gyro = 0.0
pitch = 0.0
roll = 0.0
z_gyro_offset = 0.0
z_gyro_offset_for_caculate = 0.0
flag = 0

#for door
slop, err, _len = 0.0 ,0.0 ,0.0
pan_target  = 58  #53  ##35 -18
tilt_target = -4 #-4.5

def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0

def read_IMU(data):#imu_get_yaw_by_integral 
    if IMU_RECORD:
        global z_gyro
        global pitch
        global roll
        global z_gyro_offset
        global argsIMU
        angular_velocity = data.angular_velocity
        orientation = data.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, _ = euler_from_quaternion(orientation_list)

        pitch = np.degrees(pitch)
        roll  = np.degrees(roll)

        # === 原本 IMU 積分寫法 ===
        z_gyro = z_gyro - angular_velocity.z / 2.0 + z_gyro_offset_for_caculate
        z_gyro_offset = angular_velocity.z / 2.0

        z_gyro_offset = angular_velocity.z / 2.0
        try:
            argsIMU[0] = pitch - np.degrees(robot.joint_pos["head_tilt"])#for caculating angle about camera and floor
            # print("pitch",pitch)
            # print("roll",roll)
        except:
            pass
        # print("???:",z_gyro)
        # try:
            # robot.setJointPos(["head_pan"], [np.radians(0.0)])
            # robot.setJointPos(["head_tilt"], [np.radians(pitch - camera_ang)])
            # print(-camera_ang + pitch)
            # prtin
        # except:
        #     pass
    

directionx = 0.0
directiony = 0.0
find_the_path = 0.0

# === Discrete obstacle avoidance parameters ===
closest_obs_z = 1e6
closest_z_buf = deque(maxlen=3)

heading_target = 0.0
avoid_side = 1
last_avoid_side = 1
pass_clear_time = None
pass_start_time = None

STOP_DIST = 45.0      # 距離門檻調小：避免離藍色障礙物還很遠就停下
AVOID_ANGLE = 32.0
MIN_PASS_T = 1.0
PASS_X = 0            # 通過時前進量放小，避免斜著撞進去
PASS_Y = 12
MAX_PASS_T = 5.0
HOLD_TIME = 2.0
hold_start_time = None

TURN_SIGN = -1        # directiony 與轉身方向的關係；若轉錯邊就改回 +1
last_turn_dir = 1

TURN_TH_CLAMP = 2.2
TURN_TH_KP = 0.08
TURN_REACH_TOL = 4.0
TURN_SETTLE_TIME = 0.6
PASS_TH_CLAMP = 2.0
ROTATE_BACK_KP = 0.04
ROTATE_BACK_CLAMP = 0.8
ROTATE_BACK_TIMEOUT = 1.8
rotate_back_start_time = None



# === Heading correction while walking / side-stepping ===
PATH_Y_GAIN = 5.0
PATH_Y_CLAMP = 6.0
PATH_X_WHEN_SIDE = 1.5
HEADING_KP = 0.05
HEADING_TH_CLAMP = 1.0



# === A* no-path recovery ===
NO_PATH_TH = 0.5          # vision 的 0.3 代表 no path but try forward，也要當成沒路
BACKUP_TIME = 1.0         # 黃色平放障礙需要多退一點，讓 A* 重新看到路
BACKUP_X = -10            # 後退速度，仍保守但比 -12 明顯
RECHECK_WAIT = 0.7        # 後退後停一下，等 vision 重算
NO_PATH_RETRY_MAX = 1     # 黃色障礙容易造成 no path，多退幾次再掃頭
backup_start_time = None
no_path_retry_count = 0
last_backup_print = 0.0

# === Stop-and-scan before turning ===
FORWARD_LOCK_TIME = 0.0
FORWARD_LOCK_X = 0
FORWARD_LOCK_STOP_Z = 45
force_forward_until = 0.0

SCAN_ANGLES = [35, -35, 55, -55]
SCAN_WAIT = 2.0
SCAN_SAFE_Z = 20
scan_index = 0
SCAN_FORCE_SIDE_TIME = 4.0
SCAN_FORCE_SIDE_Y = 20
scan_force_side_sign = 0
scan_best_score = -999.0
scan_best_angle = 0
scan_best_dirx = 0.0
scan_best_diry = 0.0
scan_best_z = 1e6
scan_start_time = None

args_red = ((np.array([0, 50, 37]), np.array([10, 255, 255])),
         (np.array([170, 50, 126]), np.array([179, 255, 255]))) #red

def rotate_point(point, center, angle):

    angle_rad = np.radians(angle)
    
    translated_point = (point[0] - center[0], point[1] - center[1])
    
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rotated_point = np.dot(rotation_matrix, translated_point)
    
    final_point = (rotated_point[0] + center[0], rotated_point[1] + center[1])
    
    return final_point

def read_obstacle_vision_data(data):#imu_get_yaw_by_integral 
    global directionx
    global directiony
    global find_the_path
    directionx = data.first
    directiony = data.second
    find_the_path = data.third
    # print("directionx:",directionx,"directiony",directiony,"find_the_path",find_the_path)


obs_seen = False
last_seen_time = 0.0
SEEN_TIMEOUT = 1.5        # 視覺連續看不到障礙這麼久 → 視為前方淨空

def read_closest_z(data):
    global closest_obs_z, obs_seen, last_seen_time
    v = float(data.data)
    if 8 < v < 10000:                 # 過小距離通常是幾何估測誤差，先濾掉
        closest_z_buf.append(v)
        closest_obs_z = float(np.median(list(closest_z_buf)))
        obs_seen = True
        last_seen_time = time.time()
    else:                             # 視覺回報看不到（1e6）或距離太小
        if time.time() - last_seen_time > SEEN_TIMEOUT:
            obs_seen = False
            closest_obs_z = 1e6
            closest_z_buf.clear()


def get_img(data):
    bridge = CvBridge()
    try:
        # Convert the ROS Image message to OpenCV format
        global oimg
        oimg = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

def rotate_image(image_path, angle):
    # image = cv2.imread(image_path)
    image = image_path.copy()

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def see_the_door():
    img_copy = oimg.copy()
    lower1 = args_red[0][0]
    upper1 = args_red[0][1]
    lower2 = args_red[1][0]
    upper2 = args_red[1][1]
    hsv_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_img, lower1, upper1)
    mask2 = cv2.inRange(hsv_img, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))

    result = cv2.bitwise_and(img_copy, img_copy, mask=mask)

    _,contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = 0
    largest_area = 0
    for contour in contours:
        if (cv2.contourArea(contour) > largest_area):
            largest_area = cv2.contourArea(contour)
            largest_cnt = contour
    # area = largest_area
    if largest_area > 20000:
        return True

    return False

def get_door_information():
    img_w = oimg.shape[1]
    img_h = oimg.shape[0]
    img_copy = oimg.copy()
    lower1 = args_red[0][0]
    upper1 = args_red[0][1]
    lower2 = args_red[1][0]
    upper2 = args_red[1][1]
    hsv_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv_img, lower1, upper1)
    mask2 = cv2.inRange(hsv_img, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))

    result = cv2.bitwise_and(img_copy, img_copy, mask=mask)

    _,contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = 0
    largest_area = 0
    for contour in contours:
        if (cv2.contourArea(contour) > largest_area):
            largest_area = cv2.contourArea(contour)
            largest_cnt = contour
    # area = largest_area
    if largest_area < 5000:
        return False,0,(0,0)
    
    cnt =  cv2.approxPolyDP(largest_cnt, 7,True)
    xl,yl,xr,yr = 0,0,0,0
    max_left_down_val  = -1000000
    max_right_down_val = -1000000
    camera_ang_door = pitch-np.degrees(robot.joint_pos["head_tilt"])
    for point in cnt:
        x,y = point[0][0],point[0][1]
        left_down_val =  y - (x-img_w/2)
        right_down_val = y + (x-img_w/2)
        if left_down_val > max_left_down_val:
            xl,yl = x,y
            max_left_down_val = left_down_val
        if right_down_val > max_right_down_val:
            xr,yr = x,y
            max_right_down_val = right_down_val
        # result = cv2.circle(result, (point[0][0],point[0][1]), 10, (0,255,0), 3)
    try:
        ztheta = np.degrees(np.arctan((yl-(img_h/2))*np.tan(np.radians(vfov/2))/(img_h/2))) + camera_ang_door
        z = ((camera_h-30) / np.sin((ztheta) / 180 * np.pi)) * np.sin((90 - ztheta) / 180 * np.pi)
        
        rotate_ang = np.radians(abs(camera_ang_door-ztheta))
        rotate_z = z/np.cos(np.radians(ztheta))
        hfov_rotate = 2*np.arcsin( np.tan(hfov_rad/2) /       np.sqrt( np.tan(rotate_ang)**2     +   (1/(np.cos(hfov_rad/2)))**2 ) )
        hfov_rotate = np.degrees(hfov_rotate)

        xtheta = np.degrees(np.arctan((xl-(img_w/2))*np.tan(np.radians(hfov_rotate/2))/(img_w/2)))
        x = rotate_z * np.tan(np.radians(xtheta))

        yl_t = int(z)
        xl_t = int(x)


        ztheta = np.degrees(np.arctan((yr-(img_h/2))*np.tan(np.radians(vfov/2))/(img_h/2))) + camera_ang_door
        z = ((camera_h-30) / np.sin((ztheta) / 180 * np.pi)) * np.sin((90 - ztheta) / 180 * np.pi)
        
        rotate_ang = np.radians(abs(camera_ang_door-ztheta))
        rotate_z = z/np.cos(np.radians(ztheta))
        hfov_rotate = 2*np.arcsin( np.tan(hfov_rad/2) /       np.sqrt( np.tan(rotate_ang)**2     +   (1/(np.cos(hfov_rad/2)))**2 ) )
        hfov_rotate = np.degrees(hfov_rotate)

        xtheta = np.degrees(np.arctan((xr-(img_w/2))*np.tan(np.radians(hfov_rotate/2))/(img_w/2)))
        x = rotate_z * np.tan(np.radians(xtheta))

        yr_t = int(z)
        xr_t = int(x)

        print(xl,yl,xr,yr)

        print(xl_t,yl_t,xr_t,yr_t)

        print("--------------------")

        slope_t = float(yr_t - yl_t) / float(xr_t - xl_t)
        slope = float(yr - yl) / float(xr - xl)
        # slope = 10 
        slope = -math.degrees(math.atan(slope))
        slope_t = math.degrees(math.atan(slope_t))
        # print("slope:",slope,"slope_t",slope_t,"pan ang:",np.degrees(robot.joint_pos["head_pan"]))
        # slope -= roll
    except Exception as e:
        print("slope error:",e)
        slope_t = 0
        slope = 0
    xm = (xr + xl) / 2
    ym = (yr + yl) / 2
    result = cv2.circle(result, (xl,yl), 10, (0,255,0), 3)
    result = cv2.circle(result, (xr,yr), 10, (255,0,0), 3)
    result = cv2.circle(result, (xm,ym), 10, (0,0,255), 3)
    _len = np.sqrt((xr - xl) ** 2 + (yr - yl) ** 2)
    # print(xl,yl,mask.shape[1],mask.shape[0],float(xl)/mask.shape[1],float(yl)/mask.shape[0])
    
    Px = 0.5 - float(yl)/mask.shape[0]
    Py = 0.5 - float(xl)/mask.shape[1]
    # print(Px,Py)

    # pan_angle_old = np.degrees(robot.joint_pos["head_pan"])
    # tilt_angle_old = np.degrees(robot.joint_pos["head_tilt"])
    # print(pan_angle_old,tilt_angle_old)
    # kp = 10
    # pan_angle = pan_angle_old + Py*kp
    # pan_angle = np.clip(pan_angle, -70, 70)
    # tilt_angle = tilt_angle_old + Px*kp
    # tilt_angle = np.clip(tilt_angle, -70, 70)
    # # print(Py*150,Dy*kd,Py*150 +  Dy*kd)
    # # last_P = Py
    # robot.setJointPos(["head_pan","head_tilt"], [np.radians(pan_angle),np.radians(tilt_angle)])
    # robot.setJointPos(["head_pan"], [np.radians(pan_angle)])
    # print("slope: ",slope)
    # print("mask.shape: ",mask.shape)
    # print(tilt_angle,pan_angle)


    if DEBUG_MODE:
        pass
        # result = cv2.bitwise_and(img_copy, img_copy, mask=mask)
        # cv2.line(result, (mx1, my1), (mx2, my2), (0,255,0), 2)
        # if lines is not None:
        #     for line in lines:
        #         x1, y1, x2, y2 = line[0]
        #         color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        #         cv2.line(result, (x1, y1), (x2, y2), color, 2)
        # result = rotate_image(result, roll)
        # cv2.imshow("red", result)

        # cv2.imshow("lines", lines)
        cv2.waitKey(1)
    return True, slope_t, (Px, Py)

def init():
    robot.walkStop()
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan","head_tilt"], [0,np.radians(pitch - camera_ang)])
    robot.setJointsControlModule(["l_sho_pitch","l_sho_roll","l_el"],["none","none","none"])
    robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el"],[0.2,1.5,0])
    robot.setJointsControlModule(["r_sho_pitch","r_sho_roll","r_el"],["none","none","none"])
    robot.setJointPos(["r_sho_pitch","r_sho_roll","r_el"],[-0.2,-1.5,0])
    # robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["r_hip_yaw","l_hip_yaw","r_hip_roll","l_hip_roll","r_hip_pitch","l_hip_pitch","r_knee","l_knee","r_ank_pitch","l_ank_pitch","r_ank_roll","l_ank_roll"],["walking_module"])
    
    robot.walkStop()
    

class States:
    INIT = -1
    READY = 0 # Waits for start button
    OBSTACLE = 1
    OBSTACLE_NO_PATH = 2
    DOOR_LOCATE = 3
    DOOR_CROSS = 4
    OBSTACLE_ROTATE_OUT = 5
    OBSTACLE_PASS = 6
    OBSTACLE_ROTATE_BACK = 7
    OBSTACLE_HOLD = 8
    OBSTACLE_BACKUP_RECALC = 9
    OBSTACLE_SCAN_PATH = 10
    END = 99

#HSV 
lower = np.array([[10, 90, 180], [90, 75, 0]])
upper = np.array([[30, 255, 255], [115, 255, 255]])
cl = ["y", "b"]


mapx = 90
mapz = 100

rospy.Subscriber("/cv_camera/image_raw", Image, get_img, queue_size=1)
rospy.Subscriber("/robotis/open_cr/imu", Imu, read_IMU, queue_size=1)
rospy.Subscriber('obstacle_vision_data', ThreeDouble, read_obstacle_vision_data)
rospy.Subscriber('obstacle_closest_z', Float32, read_closest_z, queue_size=1)
pub = rospy.Publisher('obstacle_head_angle', Int8, queue_size=1)


rospy.init_node("benson_obstacle")
robot = Robot()

_head_pan_cmd = 0.0
HEAD_PAN_SLEW = 4.0        # 每個 cycle 頭最多轉幾度；調小 = 更慢更穩

def set_head_pan_smooth(target_deg):
    global _head_pan_cmd
    target_deg = float(np.clip(target_deg, -60, 60))
    delta = float(np.clip(target_deg - _head_pan_cmd, -HEAD_PAN_SLEW, HEAD_PAN_SLEW))
    _head_pan_cmd += delta
    robot.setJointPos(["head_pan"], [np.radians(_head_pan_cmd)])
    return abs(target_deg - _head_pan_cmd) < 2.0     # 回傳是否已到位


def set_head_forward():
    return set_head_pan_smooth(0.0)

def turn_th_cmd(err_deg):
    return np.clip(err_deg * TURN_TH_KP, -TURN_TH_CLAMP, TURN_TH_CLAMP)

def brace_arms():
    try:
        robot.setJointsControlModule(["l_sho_pitch","l_sho_roll","l_el"], ["none","none","none"])
        robot.setJointPos(["l_sho_pitch","l_sho_roll","l_el"], [0.2, 1.5, 0])
        robot.setJointsControlModule(["r_sho_pitch","r_sho_roll","r_el"], ["none","none","none"])
        robot.setJointPos(["r_sho_pitch","r_sho_roll","r_el"], [-0.2, -1.5, 0])
    except Exception:
        pass


rospy.sleep(2) # Make sure every publisher has registered to their topic,               # avoiding lost messages
z_gyro_offset_for_caculate = z_gyro_offset
z_gyro = 0.0

tickrate = 60
rate = rospy.Rate(tickrate)

time_for_Pd = time.time()


currState = States.INIT
while not rospy.is_shutdown():
    # get_door_information()


    
    if robot.buttonCheck("mode"):
        currState = States.INIT

    try:
        pub.publish(int(round(argsIMU[0])))
    except Exception:
        pass

    if DEBUG_MODE:
        # robot.setJointPos(["head_tilt"], [np.radians(pitch - camera_ang)])
        # print(np.degrees(robot.joint_pos["head_tilt"]),pitch,-np.degrees(robot.joint_pos["head_tilt"])+pitch)
        # print(directionx,find_the_path)
        pass
    if currState == States.INIT:
        print("[INIT]")
        pub.publish(camera_ang)
        
        
        init()
        # robot.walkStart()
        rospy.sleep(3)
        
        # Transition
        tick_count = 0
        direction = False
        currState = States.READY
        # currState = States.DOOR_LOCATE
        # currState = States.DOOR_CROSS

        z_gyro_offset_for_caculate = z_gyro_offset
        z_gyro = 0.0
        print("[INIT_FINISH]")
    elif currState == States.READY:
        # print("[READY]")
        
        if robot.buttonCheck("start"):
        
            tick_count = 0
            robot.walkStart()
            z_gyro = 0.0
            currState = States.OBSTACLE
            # currState = States.DOOR_CROSS
            # currState = States.DOOR_LOCATE
            # currState = States.DOOR_CROSS
            print("start")

    elif currState == States.OBSTACLE:
        # === 1) 進場：朝原本前進方向走，頭持續看原方向 ===
        heading_target = 0.0
        set_head_forward()


        # 重要：黃色障礙物平放在地上時，A* 常會先變成 no path。
        # 所以現在先看 no path；只要沒路，就優先後退重算，不要直接轉彎。
        if obs_seen and closest_obs_z < 80 and find_the_path < NO_PATH_TH:
            robot.walkStop()
            rospy.sleep(0.3)
            backup_start_time = time.time()

            if no_path_retry_count <= 0:
                no_path_retry_count = 1

            print("[NO_PATH_BACKUP_START] z:", closest_obs_z,
                  "find_path:", find_the_path,
                  "retry:", no_path_retry_count)
            

            currState = States.OBSTACLE_BACKUP_RECALC
            continue

        # 有路但距離太近，才進入停 2 秒 + 轉身避障。
        if obs_seen and closest_obs_z < STOP_DIST:
            robot.walkStop()

            if abs(directiony) > 0.2:
                turn_dir = 1 if directiony > 0 else -1
                last_turn_dir = turn_dir
            else:
                turn_dir = last_turn_dir

            heading_target = TURN_SIGN * turn_dir * AVOID_ANGLE
            avoid_side = int(np.sign(heading_target)) or 1
            hold_start_time = time.time()

            print("[AVOID_STOP] z:", closest_obs_z,
                  "directiony:", directiony,
                  "turn_dir:", turn_dir,
                  "heading_target:", heading_target)
            currState = States.OBSTACLE_HOLD
            continue

        # 側移完成後，只有在沒有 no path、沒有太近障礙物時，才允許短暫往前
        if time.time() < force_forward_until:
            robot.walkStart()
            robot.walkVelocities(
                x=FORWARD_LOCK_X,
                y=0,
                th=np.clip(z_gyro * 0.05, -1.0, 1.0),
                z_move_amplitude=0.045,
                balance=True,
                z_offset=0
            )
            continue

        no_path_retry_count = 0

        # 距離越近走越慢，留反應時間
        speed_scale = float(np.clip((closest_obs_z - STOP_DIST) / 60.0, 0.15, 1.0))
        step_lenx = 6 * speed_scale

        # === A* path following ===
        # directiony 有明顯左右方向時，不要只直走，要小前進 + 側移
        if abs(directiony) > 0.2:
            x_cmd = PATH_X_WHEN_SIDE * speed_scale
            y_cmd = np.clip(directiony * PATH_Y_GAIN,
                            -PATH_Y_CLAMP,
                            PATH_Y_CLAMP)
        else:
            x_cmd = max(0.0, directionx) * step_lenx
            y_cmd = 0.0

        robot.walkVelocities(
            x=x_cmd,
            y=y_cmd,
            th=np.clip(z_gyro * HEADING_KP, -HEADING_TH_CLAMP, HEADING_TH_CLAMP),
            z_move_amplitude=0.045,
            balance=True,
            z_offset=0
        )

        if see_the_door():
            robot.walkStop()
            currState = States.DOOR_LOCATE
            get_door_information()

    elif currState == States.OBSTACLE_BACKUP_RECALC:
        # === A*) A* 算不出路：先後退幾秒，停一下讓視覺重算，再決定是否繼續 ===
        set_head_forward()

        if backup_start_time is None:
            backup_start_time = time.time()

        dt = time.time() - backup_start_time

        if dt < BACKUP_TIME:
            if obs_seen and closest_obs_z < 15:
                robot.walkStart()

                if time.time() - last_backup_print > 0.5:
                    print("[MICRO_BACKING] dt:", round(dt, 2),
                          "x:", BACKUP_X,
                          "z:", closest_obs_z,
                          "find_path:", find_the_path)
                    last_backup_print = time.time()

                robot.walkVelocities(
                    x=BACKUP_X,
                    y=0,
                    th=np.clip(z_gyro * 0.3, -2, 2),
                    z_move_amplitude=0.045,
                    balance=True,
                    z_offset=0
                )
            else:
                robot.walkStop()

                if time.time() - last_backup_print > 0.5:
                    print("[WAIT_RECALC] dt:", round(dt, 2),
                          "z:", closest_obs_z,
                          "find_path:", find_the_path)
                    last_backup_print = time.time()

        else:
            robot.walkStop()
            rospy.sleep(RECHECK_WAIT)

            print("[NO_PATH_RECHECK] z:", closest_obs_z,
                  "find_path:", find_the_path,
                  "retry:", no_path_retry_count)

            # 如果 A* 重新算出路，恢復正常流程。
            # 若距離仍太近，回 OBSTACLE 後會進 AVOID_STOP；若距離夠遠就會正常前進。
            if find_the_path >= NO_PATH_TH:
                no_path_retry_count = 0

                if obs_seen and closest_obs_z < 80:
                    print("[NO_PATH_CLOSE_FORCE_SCAN] z:", closest_obs_z,
                        "find_path:", find_the_path)
                    currState = States.OBSTACLE_NO_PATH
                else:
                    currState = States.OBSTACLE

            # 還是沒路，就再後退一次重算。
            # 注意：這裡不能因 closest_obs_z < STOP_DIST 就直接轉彎，
            # 因為黃色平放障礙常常會在很近時讓 find_path=0。
            elif no_path_retry_count < NO_PATH_RETRY_MAX:
                no_path_retry_count += 1
                backup_start_time = time.time()
                print("[NO_PATH_BACKUP_RETRY] z:", closest_obs_z,
                      "find_path:", find_the_path,
                      "retry:", no_path_retry_count)

            # 連續後退仍沒路，才進原本掃頭找路流程
            else:
                no_path_retry_count = 0

                print("[NO_PATH_SCAN_HEAD] still no path, enter OBSTACLE_NO_PATH")
                currState = States.OBSTACLE_NO_PATH


    elif currState == States.OBSTACLE_NO_PATH:
        _flag = True
        ang = 0.0
        robot.walkStop()

        if scan_force_side_sign != 0:
            side_sign = scan_force_side_sign
            scan_force_side_sign = 0

            print("[FORCE_SIDE_STEP] side:", side_sign)

            pass_time = time.time()
            while (time.time() - pass_time) < SCAN_FORCE_SIDE_TIME:
                robot.walkStart()
                robot.walkVelocities(
                    x=0,
                    y=side_sign * SCAN_FORCE_SIDE_Y,
                    th=np.clip(z_gyro*0.05, -2, 2),
                    z_move_amplitude=0.045,
                    balance=True,
                    z_offset=0
                )

                if obs_seen and closest_obs_z < 8:
                    robot.walkStop()
                    print("[FORCE_SIDE_EMERGENCY_STOP] z:", closest_obs_z)
                    break

            robot.walkStop()
            rospy.sleep(0.2)
            
            heading_target = 0.0
            rotate_back_start_time = time.time()
            print("[FORCE_SIDE_DONE_ROTATE_BACK] z_gyro:", z_gyro)

            currState = States.OBSTACLE_ROTATE_BACK
            continue

        order = [30,40,60,80]
        for i in order:
            robot.setJointPos(["head_pan"], [np.radians(i)])
            rospy.sleep(2)
            if find_the_path > 0.5:
                _flag = False
                ang = i
                break
            robot.setJointPos(["head_pan"], [np.radians(-i)])
            rospy.sleep(2)
            if find_the_path > 0.5:
                _flag = False
                ang = -i
                break
        # for i in order:
        #     if _flag:
        #         robot.setJointPos(["head_pan"], [np.radians(-i)])
        #         rospy.sleep(2)
        #         if find_the_path > 0.5:
        #             _flag = False
        #             ang = -i
        #             break

        # while not robot.buttonCheck("start"):
        #     robot.walkStop()

        if not _flag:
            robot.setJointPos(["head_pan"], [np.radians(0.0)])
            step_lenx = 0
            step_leny = 18
            rotate_XY = rotate_point((directionx,directiony), (0,0), ang)
            directionx_rotate = rotate_XY[0]
            directiony_rotate = rotate_XY[1]
            print("directionx_rotate:",directionx_rotate,"directiony_rotate:",directiony_rotate)
            # y= directiony_rotate * step_leny
            pass_time = time.time()
            while(time.time() - pass_time) < 6:
                robot.walkStart()

                if abs(directiony_rotate) < 0.1:
                    side_sign = last_turn_dir
                else:
                    side_sign = 1 if directiony_rotate > 0 else -1

                robot.walkVelocities(
                    x=0,
                    y=side_sign * step_leny,
                    th=np.clip(z_gyro*0.05, -2, 2),
                    z_move_amplitude=0.045,
                    balance=True,
                    z_offset=0
                )

                if (time.time() - pass_time) > 3.5 and find_the_path > 0.5 and closest_obs_z > 75:
                    break
            # robot.walkStop()
        else:
            robot.setJointPos(["head_pan"], [np.radians(0.0)])
            print("still no path, side step instead of going back")
            pass_time = time.time()

            while(time.time() - pass_time) < 8:
                robot.walkStart()
                robot.walkVelocities(
                    x=0,
                    y=last_turn_dir * 18,
                    th=np.clip(z_gyro*0.05, -2, 2),
                    z_move_amplitude=0.045,
                    balance=True,
                    z_offset=0
                )

                if (time.time() - pass_time) > 2 and find_the_path > 0.5:
                    break

            if find_the_path <= 0.5:
                last_turn_dir = -last_turn_dir

        # while not robot.buttonCheck("start"):
        #     robot.walkStop()
        if not _flag:
            robot.walkStop()
            rospy.sleep(0.2)
            heading_target = 0.0
            rotate_back_start_time = time.time()
            print("[SIDE_DONE_ROTATE_BACK] z_gyro:", z_gyro)
            currState = States.OBSTACLE_ROTATE_BACK
        else:
            currState = States.OBSTACLE


    elif currState == States.OBSTACLE_HOLD:
        # === 1.5) 停住等待 2 秒；頭仍補償看原本前方 ===
        robot.walkStop()
        set_head_forward()

        if hold_start_time is None:
            hold_start_time = time.time()

        if time.time() - hold_start_time >= HOLD_TIME:
            scan_index = 0
            scan_best_score = -999.0
            scan_best_angle = 0
            scan_best_dirx = directionx
            scan_best_diry = directiony
            scan_best_z = closest_obs_z
            scan_start_time = time.time()

            print("[AVOID_SCAN_START] hold done, scan left/right. z:", closest_obs_z,
                  "target:", heading_target,
                  "side:", avoid_side)

            currState = States.OBSTACLE_SCAN_PATH


    elif currState == States.OBSTACLE_SCAN_PATH:
        robot.walkStop()
        if scan_index >= len(SCAN_ANGLES):
            set_head_forward()

            if scan_best_score < 5.0 or scan_best_dirx < 0.2 or scan_best_z < SCAN_SAFE_Z or abs(scan_best_angle) < 20:
                robot.walkStop()
                backup_start_time = time.time()
                print("[SCAN_NO_SAFE_PATH] score:", scan_best_score,
                    "dirx:", scan_best_dirx,
                    "diry:", scan_best_diry,
                    "best_z:", scan_best_z,
                    "now_z:", closest_obs_z)
                currState = States.OBSTACLE_BACKUP_RECALC
                continue

            if abs(scan_best_angle) >= 20:
                turn_dir = 1 if scan_best_angle > 0 else -1
                last_turn_dir = turn_dir
            elif abs(scan_best_diry) > 0.2:
                turn_dir = 1 if scan_best_diry > 0 else -1
                last_turn_dir = turn_dir
            else:
                turn_dir = last_turn_dir

            print("[SCAN_DECIDE] best_angle:", scan_best_angle,
                "best_dirx:", scan_best_dirx,
                "best_diry:", scan_best_diry,
                "turn_dir:", turn_dir)

            scan_force_side_sign = 1 if scan_best_angle > 0 else -1

            print("[SCAN_TO_FORCE_SIDE] best_angle:", scan_best_angle,
                "side:", scan_force_side_sign)

            currState = States.OBSTACLE_NO_PATH
            continue

        reached = set_head_pan_smooth(SCAN_ANGLES[scan_index])   # 平滑轉到掃描角度
        if not reached:
            scan_start_time = None                # 還在轉，尚未開始計時
            continue
        if scan_start_time is None:
            scan_start_time = time.time()         # 剛到位，開始等視覺穩定
        if time.time() - scan_start_time >= SCAN_WAIT:
            scan_angle_now = SCAN_ANGLES[scan_index]
            scan_z_unknown = closest_obs_z > 9000

            if abs(scan_angle_now) < 20:
                score = -100.0
            elif (not scan_z_unknown) and closest_obs_z < SCAN_SAFE_Z:
                score = -100.0
            else:
                score = find_the_path * 10.0 + max(directionx, 0.0) - 0.02 * abs(scan_angle_now)

            print("[SCAN_CHECK] angle:", SCAN_ANGLES[scan_index],"dirx:", directionx,"diry:", directiony,"find:", find_the_path,"score:", score,"z:", closest_obs_z)
            
            if score > scan_best_score:
                scan_best_score = score
                scan_best_angle = SCAN_ANGLES[scan_index]
                scan_best_dirx  = directionx
                scan_best_diry  = directiony
                scan_best_z     = closest_obs_z
            scan_index += 1
            scan_start_time = None

    
    elif currState == States.OBSTACLE_ROTATE_OUT:
        # === 2) 原地慢速旋轉到避障方向 ===
        brace_arms()
        set_head_forward()
        robot.walkStart()

        turn_err = z_gyro - heading_target

        robot.walkVelocities(
            x=0,
            y=0,
            th=turn_th_cmd(turn_err),
            z_move_amplitude=0.045,
            balance=True,
            z_offset=0
        )

        if abs(turn_err) < TURN_REACH_TOL:
            robot.walkStop()
            rospy.sleep(TURN_SETTLE_TIME)
            robot.walkStart()

            pass_clear_time = None
            pass_start_time = time.time()
            print("[AVOID_PASS] heading reached:", z_gyro, "target:", heading_target,
                "side:", avoid_side, "turn_err:", turn_err)
            currState = States.OBSTACLE_PASS

    elif currState == States.OBSTACLE_PASS:
        if find_the_path < NO_PATH_TH:
            robot.walkStop()
            rospy.sleep(0.2)
            print("[PASS_ABORT_NO_PATH_TO_SIDE] z:", closest_obs_z,
                "find_path:", find_the_path)
            currState = States.OBSTACLE_NO_PATH
            continue


        if obs_seen and closest_obs_z < 35:
            robot.walkStop()
            rospy.sleep(0.2)
            print("[PASS_ABORT_TOO_CLOSE] z:", closest_obs_z)
            currState = States.OBSTACLE_NO_PATH
            continue


        # === 3) 斜向前進通過障礙；橫移與轉身同側；用「看不到障礙」判定通過 ===
        set_head_forward()
        pass_y = -np.sign(heading_target) * PASS_Y

        # 障礙物還近：只側移，不往前
        if obs_seen and closest_obs_z < 65:
            pass_x_cmd = 0
            pass_y_cmd = pass_y
        else:
            pass_x_cmd = PASS_X
            pass_y_cmd = pass_y * 0.7

        robot.walkVelocities(
            x=pass_x_cmd,
            y=pass_y_cmd,
            th=np.clip((z_gyro - heading_target) * TURN_TH_KP, -PASS_TH_CLAMP, PASS_TH_CLAMP),
            z_move_amplitude=0.045,
            balance=True,
            z_offset=0
        )

        pass_timeout = (pass_start_time is not None and
                        time.time() - pass_start_time > MAX_PASS_T)

        if not obs_seen:                       # 視覺已看不到障礙 → 通過了
            if pass_clear_time is None:
                pass_clear_time = time.time()
            elif time.time() - pass_clear_time > MIN_PASS_T:
                robot.walkStop(); rospy.sleep(0.2)
                heading_target = 0.0
                print("[AVOID_BACK] cleared (obstacle out of view)")
                currState = States.OBSTACLE_ROTATE_BACK
        elif pass_timeout:
            robot.walkStop(); rospy.sleep(0.2)
            last_turn_dir = -last_turn_dir      # 這一側繞不過，下次換邊
            heading_target = 0.0
            print("[AVOID_BACK_TIMEOUT] flip side next time")
            currState = States.OBSTACLE_ROTATE_BACK
        else:
            pass_clear_time = None

    elif currState == States.OBSTACLE_ROTATE_BACK:
        # === 4) 慢速原地轉回原本前進方向 ===
        brace_arms()
        set_head_forward()
        robot.walkStart()

        if rotate_back_start_time is None:
            rotate_back_start_time = time.time()

        back_elapsed = time.time() - rotate_back_start_time

        th_back = np.clip(-z_gyro * ROTATE_BACK_KP,
                        -ROTATE_BACK_CLAMP,
                        ROTATE_BACK_CLAMP)

        robot.walkVelocities(
            x=0,
            y=0,
            th=th_back,
            z_move_amplitude=0.045,
            balance=True,
            z_offset=0
        )

        print("[ROTATE_BACK] z_gyro:", z_gyro,
            "th:", th_back,
            "t:", round(back_elapsed, 2))

        if abs(z_gyro) < TURN_REACH_TOL or back_elapsed > ROTATE_BACK_TIMEOUT:
            robot.walkStop()
            rospy.sleep(TURN_SETTLE_TIME)

            # 舊 IMU 會漂，轉回流程結束後直接重新定義目前方向為新的前進方向
            z_gyro_offset_for_caculate = z_gyro_offset
            z_gyro = 0.0
            rotate_back_start_time = None

            robot.walkStart()
            print("[AVOID_DONE] rotate back finished/reset heading")
            currState = States.OBSTACLE


        
    elif currState == States.DOOR_LOCATE:
        if flag < 180:
            if flag == 0:
                robot.setJointPos(["head_pan"], [np.radians(0)])
                rospy.sleep(1)
            # robot.setJointPos(["head_pan","head_tilt"], [np.radians(0), np.radians(-40)])
            # robot.setJointPos(["head_pan"], [np.radians(0)])
            robot.walkStop()
            ret, _slop, _err = get_door_information()
            flag = flag + 1
            if ret:
                robot.walkStop()
                slop, err = _slop, _err
                target_slop = slop + np.degrees(robot.joint_pos["head_pan"])
                print("target_slop:",target_slop)

            pass_time = time.time()
            Px , Py  = _err[0] , _err[1]
            pan_angle_old = np.degrees(robot.joint_pos["head_pan"])
            tilt_angle_old = np.degrees(robot.joint_pos["head_tilt"])
            # print(tilt_angle_old,pan_angle_old,z_gyro + target_slop)
            kp = 20
            pan_angle = pan_angle_old + Py*kp
            pan_angle = np.clip(pan_angle, -70, 70)
            tilt_angle = tilt_angle_old + Px*kp
            tilt_angle = np.clip(tilt_angle, -70, 70)
            # print(Py*150,Dy*kd,Py*150 +  Dy*kd)
            # last_P = Py
            x_step = tilt_angle - tilt_target
            y_step = pan_angle - pan_target
            robot.setJointPos(["head_pan","head_tilt"], [np.radians(pan_angle),np.radians(tilt_angle)])
            # if flag == 300:
                # robot.setJointPos(["head_pan","head_tilt"], [np.radians(0), np.radians(pitch - 15)])
            # robot.setJointPos(["head_pan","head_tilt"], [np.radians(0), np.radians(pitch - 35)])
            # print("z to :",target_slop)

                # z_gyro -= slop
            continue



        # while not robot.buttonCheck("start"):
        #     robot.walkStop()
        ret, _slop, _err = get_door_information()
        if ret:
            slop, err = _slop, _err

        Px , Py  = _err[0] , _err[1]
        pan_angle_old = np.degrees(robot.joint_pos["head_pan"])
        tilt_angle_old = np.degrees(robot.joint_pos["head_tilt"])
        # print(tilt_angle_old,pan_angle_old,z_gyro + target_slop)
        kp = 20
        pan_angle = pan_angle_old + Py*kp
        pan_angle = np.clip(pan_angle, -70, 70)
        tilt_angle = tilt_angle_old + Px*kp
        tilt_angle = np.clip(tilt_angle, -70, 70)
        # print(Py*150,Dy*kd,Py*150 +  Dy*kd)
        # last_P = Py
        x_step = tilt_angle - tilt_target
        y_step = pan_angle - pan_target
        # print(x_step,y_step,z_gyro + target_slop)
        robot.setJointPos(["head_pan","head_tilt"], [np.radians(pan_angle),np.radians(tilt_angle)])

        if (time.time() - pass_time) > 2:
            robot.walkStart()
        robot.walkVelocities(x=np.clip(x_step, -15, 5), y=np.clip(y_step, -15, 15), th=np.clip(z_gyro + target_slop, -5, 5), z_move_amplitude=0.045, balance=True, z_offset=0)# y+:left,-:right the+:left, -:right
        

        if(abs(x_step) < 5 and abs(y_step) < 3 and abs(z_gyro + target_slop) < 3) or (time.time() - pass_time) > 25:
            flag = 0
            # robot.walkVelocities(x=-5, y=0, th=np.clip(z_gyro, -5, 5), z_move_amplitude=0.045, balance=True, z_offset=0)# y+:left,-:right the+:left, -:right
            robot.walkStop()
            currState = States.DOOR_CROSS
            robot.walkStop()
            # while not robot.buttonCheck("start"):
            #     robot.walkStop()
        # robot.walkVelocities(x=np.clip(-5, -15, 5), y=np.clip(y_step, -15, 15), th=np.clip(0, -5, 5), z_move_amplitude=0.045, balance=True, z_offset=0)# y+:left,-:right the+:left, -:right

    elif currState == States.DOOR_CROSS:
        # while not robot.buttonCheck("start"):
        #     robot.walkStop()
        robot.setGeneralControlModule("action_module")
        IMU_RECORD = False
        robot.playMotion(30, wait_for_end=True)
        for i in range(2):
            robot.playMotion(31, wait_for_end=True)
        IMU_RECORD = True
        robot.playMotion(32, wait_for_end=True)
        rospy.sleep(1)
        IMU_RECORD = False
        currState = States.OBSTACLE
        init()
        robot.setJointPos(["head_tilt"], [np.radians(pitch - camera_ang)])
        rospy.sleep(5)
        z_gyro_offset_for_caculate = z_gyro_offset
        z_gyro = 0.0
        IMU_RECORD = True
        # while not robot.buttonCheck("start"):
        #     robot.setJointPos(["head_pan"], [np.radians(0.0)])
        #     robot.walkStop()
        rospy.sleep(0.2)
        robot.walkStart()




robot.walkStop()
