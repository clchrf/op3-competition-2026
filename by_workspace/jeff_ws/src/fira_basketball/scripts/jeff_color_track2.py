#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from op3_utils import Robot_weightlift
from op3_ros_utils import Robot

# ==========================================
# [核心調校區] 請依照機器人行為調整這裡
# ==========================================

# 1. [前進模式] 解決直走會橫移
# 往左滑 -> 設 1.0 / 往右滑 -> 設 -1.0
WALK_Y_BIAS = 1.0 

# 2. [前進模式] 解決直走會轉彎
# 路徑左彎 -> 設 0.05 / 路徑右彎 -> 設 -0.05
WALK_THETA_BIAS = 0.1

# 3. [旋轉模式] 解決原地轉會橫移
# 原地轉時身體往左飄 -> 設 1.0 ~ 2.0 (因為轉速變快，這個值可能要加大)
# 原地轉時身體往右飄 -> 設 -1.0 ~ -2.0
TURN_Y_BIAS = 2.0

# ==========================================
# 方向設定
# ==========================================
DIR_HEAD_PAN  = -1.0 
DIR_BODY_TURN = 1.0 

# ==========================================
# 參數調整區 (解限速版)
# ==========================================
DEBUG_MODE = True
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 360
TICKRATE = 30 

# ----- 紅色偵測 -----
MIN_AREA_RED = 200
H1_LO, S_LO, V_LO = 0,   140, 60
H1_HI, S_HI, V_HI = 10,  255, 255
H2_LO, S2_LO, V2_LO = 170, 140, 60
H2_HI, S2_HI, V2_HI = 180, 255, 255

# ----- 停止邏輯 -----
STOP_AREA_ON  = 20000 
STOP_AREA_OFF = 14000 

# ----- PD 控制器 (反應調快) -----
# [修正] 提高 P 值 (0.5 -> 1.2)，讓它對角度更敏感
KP_BODY = 1.2   
KD_BODY = 0.10  

# [修正] 提高最大轉速 (0.2 -> 0.6)，讓轉彎幅度變大
TH_MAX = 0.8    

# ----- 遲滯閾值 -----
PAN_GO_ENTER = 0.25  
PAN_GO_EXIT  = 0.10  

# ----- 前進速度 (速度全開) -----
# [修正] 提高最大速度 (5.0 -> 12.0)
MAX_X = 8.0       
MIN_X = 0.0       

# ----- 平滑與加速度 (加速變快) -----
ALPHA_CMD = 0.2    
# [修正] 提高加速度限制 (0.05 -> 0.2)，起步更靈敏
X_STEP_MAX = 0.2  
TH_STEP_MAX = 0.1 

# ------------------------------------------

cx_f, cy_f, area_f = 0.0, 0.0, 0.0
is_walking_active = False 
is_stopped_logic = False
is_aligning_mode = False 
robot = None
robot_weightlift = None
prev_pan_err = 0.0

def cleanup_node():
    global robot, robot_weightlift
    rospy.logwarn("!!! EMERGENCY STOP !!!")
    if robot:
        try: robot.walkStop()
        except: pass
    rospy.logwarn("!!! ROBOT STOPPED !!!")

def find_red_target(bgr_img):
    blurred = cv2.GaussianBlur(bgr_img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (H1_LO, S_LO, V_LO), (H1_HI, S_HI, V_HI))
    mask2 = cv2.inRange(hsv, (H2_LO, S2_LO, V2_LO), (H2_HI, S2_HI, V2_HI))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    if not contours: return False, None, 0.0, None, mask
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < MIN_AREA_RED: return False, None, area, None, mask
    x, y, w, h = cv2.boundingRect(c)
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return True, (cx, cy), area, (x, y, w, h), mask

def main():
    global cx_f, cy_f, area_f, is_walking_active, is_stopped_logic
    global is_aligning_mode
    global robot, robot_weightlift
    global prev_pan_err

    rospy.init_node("op3_red_follow_fast")
    rospy.on_shutdown(cleanup_node)

    robot = Robot()
    try:
        robot_weightlift = Robot_weightlift('fira_basketball')
        if not hasattr(robot_weightlift, 'joint_pos'):
            robot_weightlift.joint_pos = {}
    except Exception as e:
        rospy.logerr("Init error: " + str(e))
        return

    rospy.sleep(1.0)
    try: robot.walkStop()
    except: pass

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened(): return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    robot.setGeneralControlModule("action_module")
    robot.playMotion(1, wait_for_end=True)
    robot_weightlift.walk_set_param_pub.publish(robot_weightlift.walking_params[0])
    robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [0.0, -0.5])

    rospy.loginfo("Press START button...")
    while not rospy.is_shutdown():
        if robot.buttonCheck("start"): break
        rospy.sleep(0.1)

    rospy.loginfo("START WALKING ENGINE...")
    robot.walkStart()
    is_walking_active = True
    
    rate = rospy.Rate(TICKRATE)
    x_cmd_f, y_cmd_f, th_cmd_f = 0.0, 0.0, 0.0
    x_prev, th_prev = 0.0, 0.0
    was_found = False

    while not rospy.is_shutdown():
        if robot.buttonCheck("mode"):
            rospy.loginfo("Mode pressed - Stopping")
            break

        ret, frame = cap.read()
        if not ret: continue

        image = cv2.resize(frame, (FRAME_W, FRAME_H))
        H, W = image.shape[:2]
        img_cx, img_cy = W // 2, H // 2

        found, cxy, area, bbox, mask = find_red_target(image)

        x_cmd = 0.0
        y_cmd = 0.0
        th_cmd = 0.0
        target_pan = 0.0

        if found:
            cx, cy = cxy
            x, y, w, h = bbox

            if not was_found:
                cx_f, cy_f, area_f = float(cx), float(cy), float(area)
                prev_pan_err = 0.0 
            else:
                cx_f = 0.7 * cx_f + 0.3 * float(cx)
                cy_f = 0.7 * cy_f + 0.3 * float(cy)
                area_f = 0.7 * area_f + 0.3 * float(area)
            was_found = True

            err_x = (cx_f - img_cx) / float(img_cx)
            err_y = (cy_f - img_cy) / float(img_cy)

            curr_pan = robot.joint_pos.get("head_pan", 0.0)
            curr_tilt = robot.joint_pos.get("head_tilt", -0.5)

            target_pan = curr_pan + (0.35 * DIR_HEAD_PAN * err_x) 
            target_tilt = curr_tilt + (-0.25 * err_y)
            target_pan = max(-1.4, min(1.4, target_pan))
            target_tilt = max(-1.1, min(0.2, target_tilt))

            robot.setJointPos(["head_pan", "head_tilt"], [target_pan, target_tilt])

            if area >= STOP_AREA_ON:
                is_stopped_logic = True
            elif area_f <= STOP_AREA_OFF:
                is_stopped_logic = False

            if is_stopped_logic and is_walking_active:
                robot.walkStop()
                is_walking_active = False
                x_cmd_f, y_cmd_f, th_cmd_f = 0.0, 0.0, 0.0
                prev_pan_err = 0.0
            elif (not is_stopped_logic) and (not is_walking_active):
                robot.walkStart()
                is_walking_active = True
                prev_pan_err = 0.0
                rospy.sleep(0.5)

            if is_walking_active:
                pan_abs = abs(target_pan)

                # 計算 PD 轉向
                p_term = KP_BODY * target_pan
                d_err = target_pan - prev_pan_err
                d_term = KD_BODY * d_err
                raw_turn = (p_term + d_term) * DIR_BODY_TURN
                prev_pan_err = target_pan
                
                # 限制最大轉速 (現在設為 0.6)
                if raw_turn > TH_MAX: raw_turn = TH_MAX
                if raw_turn < -TH_MAX: raw_turn = -TH_MAX
                
                # 狀態切換
                if is_aligning_mode:
                    if pan_abs < PAN_GO_EXIT:
                        is_aligning_mode = False
                else:
                    if pan_abs > PAN_GO_ENTER:
                        is_aligning_mode = True

                if is_aligning_mode:
                    # [黃色模式] 原地旋轉
                    x_cmd = 0.0
                    y_cmd = TURN_Y_BIAS 
                    th_cmd = raw_turn 
                else:
                    # [綠色模式] 直線前進
                    x_cmd = MAX_X
                    y_cmd = WALK_Y_BIAS
                    th_cmd = WALK_THETA_BIAS

            if DEBUG_MODE:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(image, "A:%.0f" % area, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            was_found = False
            prev_pan_err = 0.0
            if is_walking_active:
                x_cmd, y_cmd, th_cmd = 0.0, 0.0, 0.0

        if is_walking_active:
            x_cmd_f = (1 - ALPHA_CMD) * x_cmd_f + ALPHA_CMD * x_cmd
            y_cmd_f = y_cmd 
            th_cmd_f = (1 - ALPHA_CMD) * th_cmd_f + ALPHA_CMD * th_cmd

            dx = x_cmd_f - x_prev
            if dx > X_STEP_MAX: x_cmd_f = x_prev + X_STEP_MAX
            elif dx < -X_STEP_MAX: x_cmd_f = x_prev - X_STEP_MAX
            x_prev = x_cmd_f

            try:
                robot_weightlift.walkVelocities(x=x_cmd_f, y=y_cmd_f, th=th_cmd_f)
            except: pass

        if DEBUG_MODE:
            if is_stopped_logic:
                status = "STOP"
                col = (0, 0, 255)
            elif is_aligning_mode:
                status = "SPOT TURN"
                col = (0, 255, 255) 
            else:
                status = "FORWARD"
                col = (0, 255, 0)   

            cv2.putText(image, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(image, "Pan:%.2f Th:%.2f X:%.1f" % (target_pan, th_cmd_f, x_cmd_f), 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow("View", image)
            cv2.waitKey(1)
            
        rate.sleep()

if __name__ == "__main__":
    main()