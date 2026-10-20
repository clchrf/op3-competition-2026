#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from op3_utils import Robot_weightlift
from op3_ros_utils import Robot

# ==========================================
# 參數調整區
# ==========================================
DEBUG_MODE = True
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 360
TICKRATE = 30 

# ----- [走路偏差修正] -----
# 如果往右偏，設 0.05；如果往左偏，設 -0.05
WALK_BIAS = 0.0 

# ----- 紅色偵測 -----
MIN_AREA_RED = 200
H1_LO, S_LO, V_LO = 0,   140, 60
H1_HI, S_HI, V_HI = 10,  255, 255
H2_LO, S2_LO, V2_LO = 170, 140, 60
H2_HI, S2_HI, V2_HI = 180, 255, 255

# ----- 停止邏輯 -----
STOP_AREA_ON  = 20000  # 大於此值 -> 觸發停止
STOP_AREA_OFF = 14000  # 小於此值 -> 重新啟動

# ----- 身體跟隨 -----
K_BODY = 0.8      
TH_MAX = 0.6
PAN_DEADZONE = 0.15 

# ----- 前進速度 -----
MAX_X = 5.0       
MIN_X = 1.0
PAN_ALIGN = 0.30  
PAN_HARD  = 0.70  

# ----- 平滑與加速度 -----
ALPHA_CMD = 0.1    
X_STEP_MAX = 0.05  
TH_STEP_MAX = 0.05

# ------------------------------------------

cx_f, cy_f, area_f = 0.0, 0.0, 0.0
# [重要] 引擎狀態旗標
is_walking_active = False 
is_stopped_logic = False

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

    mask1 = cv2.inRange(hsv, (H1_LO, S_LO, V_LO), (H1_HI, S_HI, V_HI))
    mask2 = cv2.inRange(hsv, (H2_LO, S2_LO, V2_LO), (H2_HI, S2_HI, V2_HI))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    if not contours:
        return False, None, 0.0, None, mask

    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < MIN_AREA_RED:
        return False, None, area, None, mask

    x, y, w, h = cv2.boundingRect(c)
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return True, (cx, cy), area, (x, y, w, h), mask

def main():
    global cx_f, cy_f, area_f, is_walking_active, is_stopped_logic
    global robot, robot_weightlift

    rospy.init_node("op3_red_follow_visual")
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

    # 姿勢初始化
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
    x_cmd_f, th_cmd_f = 0.0, 0.0
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
        th_cmd = 0.0
        target_pan = 0.0

        if found:
            cx, cy = cxy
            x, y, w, h = bbox

            if not was_found:
                cx_f, cy_f, area_f = float(cx), float(cy), float(area)
            else:
                cx_f = 0.8 * cx_f + 0.2 * float(cx)
                cy_f = 0.8 * cy_f + 0.2 * float(cy)
                area_f = 0.8 * area_f + 0.2 * float(area)
            was_found = True

            err_x = (cx_f - img_cx) / float(img_cx)
            err_y = (cy_f - img_cy) / float(img_cy)

            curr_pan = robot.joint_pos.get("head_pan", 0.0)
            curr_tilt = robot.joint_pos.get("head_tilt", -0.5)

            target_pan = curr_pan + (-0.25 * err_x) 
            target_tilt = curr_tilt + (-0.20 * err_y)
            target_pan = max(-1.4, min(1.4, target_pan))
            target_tilt = max(-1.1, min(0.2, target_tilt))

            robot.setJointPos(["head_pan", "head_tilt"], [target_pan, target_tilt])

            # 停止邏輯判斷
            if area >= STOP_AREA_ON:
                is_stopped_logic = True
            elif area_f <= STOP_AREA_OFF:
                is_stopped_logic = False

            # 引擎控制
            if is_stopped_logic and is_walking_active:
                robot.walkStop()
                is_walking_active = False
                x_cmd_f, th_cmd_f = 0.0, 0.0
                x_prev, th_prev = 0.0, 0.0
            elif (not is_stopped_logic) and (not is_walking_active):
                robot.walkStart()
                is_walking_active = True
                rospy.sleep(0.5)

            # 計算指令
            if is_walking_active:
                if abs(target_pan) < PAN_DEADZONE:
                    th_cmd = 0.0
                else:
                    th_cmd = K_BODY * target_pan
                th_cmd += WALK_BIAS 

                pan_abs = abs(target_pan)
                if pan_abs > PAN_HARD:
                    x_cmd = 0.0
                elif pan_abs < PAN_ALIGN:
                    x_cmd = MAX_X
                else:
                    x_cmd = MIN_X

            # [已修復] 視覺顯示：畫出紅框和球心
            if DEBUG_MODE:
                
                # 畫出紅框
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # 畫出球心
                cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)
                # 在框框旁邊顯示面積
                cv2.putText(image, "A:%.0f" % area, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        else:
            was_found = False
            if is_walking_active:
                x_cmd = 0.0
                th_cmd = 0.0

        # 平滑與發送
        if is_walking_active:
            x_cmd_f = (1 - ALPHA_CMD) * x_cmd_f + ALPHA_CMD * x_cmd
            dx = x_cmd_f - x_prev
            if dx > X_STEP_MAX: x_cmd_f = x_prev + X_STEP_MAX
            elif dx < -X_STEP_MAX: x_cmd_f = x_prev - X_STEP_MAX
            x_prev = x_cmd_f

            th_cmd_f = (1 - ALPHA_CMD) * th_cmd_f + ALPHA_CMD * th_cmd
            dth = th_cmd_f - th_prev
            if dth > TH_STEP_MAX: th_cmd_f = th_prev + TH_STEP_MAX
            elif dth < -TH_STEP_MAX: th_cmd_f = th_prev - TH_STEP_MAX
            th_prev = th_cmd_f

            try:
                robot_weightlift.walkVelocities(x=x_cmd_f, y=0.0, th=th_cmd_f)
            except: pass

        # [已修復] 狀態顯示文字 (左上角)
        if DEBUG_MODE:
            if is_stopped_logic:
                # 停止時顯示紅色
                cv2.putText(image, "STATUS: HARD STOP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # 行走時顯示綠色
                cv2.putText(image, "STATUS: WALKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 顯示面積閾值資訊
            cv2.putText(image, "Stop Thres: %d" % STOP_AREA_ON, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

            cv2.imshow("View", image)
            cv2.waitKey(1)
            
        rate.sleep()

if __name__ == "__main__":
    main()