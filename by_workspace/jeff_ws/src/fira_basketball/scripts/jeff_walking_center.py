#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from op3_utils import Robot_weightlift
from op3_ros_utils import Robot

# ==========================================
# [體態參數調校] 這是解決 "打滑/走不動" 的關鍵
# ==========================================

# 1. 重心前後調整 (解決 "壓一下才能走")
# 正數 = 身體前傾 (重心壓在前腳掌) / 負數 = 身體後仰
# 建議設 0.01 ~ 0.03 (單位: 公尺)
BODY_OFFSET_X = 0.0 

# 2. 身體高度調整 (解決 "重心不穩")
# 正數 = 站高 / 負數 = 蹲低
# 蹲低一點可以增加穩定性，建議 -0.01 ~ -0.03
BODY_OFFSET_Z = -0.02

# 3. 重心左右調整 (解決 "死命往左偏")
# 往左滑 -> 試試設 -0.01 (重心移右)
# 往右滑 -> 試試設  0.01 (重心移左)
BODY_OFFSET_Y = -0.01

# ==========================================
# 參數調整區
# ==========================================
DEBUG_MODE = True
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 360
TICKRATE = 30 

# ----- 方向與速度 -----
DIR_HEAD_PAN  = -1.0 
DIR_BODY_TURN = 1.0 
WALK_Y_BIAS = 0.0
FIX_THETA   = -0.05 

# ----- 走路參數 -----
MAX_X = 6.0       
MIN_X = 0.0       
KP_BODY = 1.0   
TH_MAX = 0.5    
PAN_GO_ENTER = 0.25  
PAN_GO_EXIT  = 0.10  
ALPHA_CMD = 0.2    

# [補回遺失的參數]
X_STEP_MAX = 0.1   # 前進加速度限制
TH_STEP_MAX = 0.05 # 轉向加速度限制

# ------------------------------------------

cx_f, cy_f, area_f = 0.0, 0.0, 0.0
is_walking_active = False 
is_stopped_logic = False
is_aligning_mode = False 
robot = None
robot_weightlift = None

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
    mask1 = cv2.inRange(hsv, (0, 140, 60), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 140, 60), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    if not contours: return False, None, 0.0, None, mask
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 200: return False, None, area, None, mask
    x, y, w, h = cv2.boundingRect(c)
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return True, (cx, cy), area, (x, y, w, h), mask

def main():
    global cx_f, cy_f, area_f, is_walking_active, is_stopped_logic
    global is_aligning_mode
    global robot, robot_weightlift

    rospy.init_node("op3_balance_tuner_v2")
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
    
    # ==========================================
    # 套用體態參數
    # ==========================================
    rospy.loginfo("Applying Body Offsets...")
    
    current_params = robot_weightlift.walking_params[0]
    
    # 修改參數 (疊加)
    current_params.init_x_offset += BODY_OFFSET_X
    current_params.init_y_offset += BODY_OFFSET_Y
    current_params.init_z_offset += BODY_OFFSET_Z
    
    robot_weightlift.walk_set_param_pub.publish(current_params)
    rospy.sleep(1.0)
    rospy.loginfo("Offsets Applied: X=%.3f, Y=%.3f, Z=%.3f" % (BODY_OFFSET_X, BODY_OFFSET_Y, BODY_OFFSET_Z))

    try: robot.walkStop()
    except: pass

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    robot.setGeneralControlModule("action_module")
    robot.playMotion(1, wait_for_end=True)
    
    robot.setGeneralControlModule("walking_module")
    robot_weightlift.walk_set_param_pub.publish(current_params)
    
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_pan", "head_tilt"], [0.0, -0.5])

    rospy.loginfo("Press START button...")
    while not rospy.is_shutdown():
        if robot.buttonCheck("start"): break
        rospy.sleep(0.1)

    rospy.loginfo("START WALKING...")
    robot.walkStart()
    is_walking_active = True
    
    rate = rospy.Rate(TICKRATE)
    x_cmd_f, y_cmd_f, th_cmd_f = 0.0, 0.0, 0.0
    x_prev, th_prev = 0.0, 0.0  # 補上 th_prev
    was_found = False

    while not rospy.is_shutdown():
        if robot.buttonCheck("mode"):
            rospy.loginfo("Mode pressed - Stopping")
            break

        ret, frame = cap.read()
        if not ret: continue
        image = cv2.resize(frame, (FRAME_W, FRAME_H))
        
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
            else:
                cx_f = 0.7 * cx_f + 0.3 * float(cx)
                cy_f = 0.7 * cy_f + 0.3 * float(cy)
                area_f = 0.7 * area_f + 0.3 * float(area)
            was_found = True

            err_x = (cx_f - (FRAME_W/2)) / (FRAME_W/2)
            err_y = (cy_f - (FRAME_H/2)) / (FRAME_H/2)

            curr_pan = robot.joint_pos.get("head_pan", 0.0)
            curr_tilt = robot.joint_pos.get("head_tilt", -0.5)
            target_pan = curr_pan + (0.35 * DIR_HEAD_PAN * err_x) 
            target_tilt = curr_tilt + (-0.25 * err_y)
            target_pan = max(-1.4, min(1.4, target_pan))
            target_tilt = max(-1.1, min(0.2, target_tilt))
            robot.setJointPos(["head_pan", "head_tilt"], [target_pan, target_tilt])

            # 邏輯區
            if area >= 20000: is_stopped_logic = True
            elif area_f <= 14000: is_stopped_logic = False

            if is_stopped_logic:
                if is_walking_active:
                    robot.walkStop()
                    is_walking_active = False
                    x_cmd, y_cmd, th_cmd = 0.0, 0.0, 0.0
            else:
                if not is_walking_active:
                    robot.walkStart()
                    is_walking_active = True
                    rospy.sleep(0.5)

            if is_walking_active:
                pan_abs = abs(target_pan)
                
                # 狀態機
                if is_aligning_mode:
                    if pan_abs < PAN_GO_EXIT: is_aligning_mode = False
                else:
                    if pan_abs > PAN_GO_ENTER: is_aligning_mode = True

                if is_aligning_mode:
                    x_cmd = 0.0
                    y_cmd = 0.0
                    th_cmd = KP_BODY * target_pan * DIR_BODY_TURN
                    if th_cmd > TH_MAX: th_cmd = TH_MAX
                    if th_cmd < -TH_MAX: th_cmd = -TH_MAX
                else:
                    x_cmd = MAX_X
                    y_cmd = WALK_Y_BIAS
                    th_cmd = FIX_THETA 

            if DEBUG_MODE:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 2)
                cv2.putText(image, "A:%.0f" % area, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        else:
            was_found = False
            if is_walking_active: x_cmd, y_cmd, th_cmd = 0.0, 0.0, 0.0

        if is_walking_active:
            x_cmd_f = (1 - ALPHA_CMD) * x_cmd_f + ALPHA_CMD * x_cmd
            y_cmd_f = y_cmd
            th_cmd_f = (1 - ALPHA_CMD) * th_cmd_f + ALPHA_CMD * th_cmd
            
            # X軸 平滑
            dx = x_cmd_f - x_prev
            if dx > X_STEP_MAX: x_cmd_f = x_prev + X_STEP_MAX
            elif dx < -X_STEP_MAX: x_cmd_f = x_prev - X_STEP_MAX
            x_prev = x_cmd_f

            # Th軸 平滑 (這次也加上去，避免瞬間轉向太快)
            dth = th_cmd_f - th_prev
            if dth > TH_STEP_MAX: th_cmd_f = th_prev + TH_STEP_MAX
            elif dth < -TH_STEP_MAX: th_cmd_f = th_prev - TH_STEP_MAX
            th_prev = th_cmd_f

            try:
                # 這裡使用 balance=True，因為我們已經修了物理重心
                robot_weightlift.walkVelocities(x=x_cmd_f, y=y_cmd_f, th=th_cmd_f, balance=True)
            except: pass

        if DEBUG_MODE:
            status = "STOP" if is_stopped_logic else "ALIGN" if is_aligning_mode else "FWD"
            cv2.putText(image, "Mode: %s" % status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(image, "Offset X:%.3f Z:%.3f" % (BODY_OFFSET_X, BODY_OFFSET_Z), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
            cv2.imshow("View", image)
            cv2.waitKey(1)
        
        rate.sleep()

if __name__ == "__main__":
    main()