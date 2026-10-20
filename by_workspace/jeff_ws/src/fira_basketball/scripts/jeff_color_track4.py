#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from op3_utils import Robot_weightlift
from op3_ros_utils import Robot
from sensor_msgs.msg import Imu

# ==========================================
# [參數設定區] 黃線追蹤 + 霸氣橫移 + IMU 直行
# ==========================================

# 1. 頭部平滑係數
HEAD_SMOOTH = 0.1

# 2. 橫移控制參數 (螃蟹步)
CRAB_SPEED = 18.0  # 橫移固定速度 (若力量不夠大走不動，可往上調到 20 或 25)
Z_MOVE_AMP = 0.045 # 側步抬腳高度 (必加參數，否則腳會拖地)

# 3. IMU 鎖定航向參數
KP_IMU = 0.5       # IMU 抵抗旋轉的力道 (維持面朝前方的抵抗力)
IMU_TH_LIMIT = 5.0 # 限制 IMU 修正的最大轉速，避免爆衝

# 4. 切換門檻限制
TH_MAX = 5.0       # 總轉速硬限制
PAN_GO_ENTER = 0.15 # 頭部偏離超過此值，進入「橫移對準模式」
PAN_GO_EXIT  = 0.05 # 頭部回到中心小於此值，進入「直走模式」

# 5. 重心與其他步態參數
BODY_OFFSET_X = 0.0     
BODY_OFFSET_Z = -0.015  # 壓低重心防摔
BODY_OFFSET_Y = -0.01   
MAX_X = 6.0        # 前進最大速度
ALPHA_CMD = 0.2    
X_STEP_MAX = 0.1   
TH_STEP_MAX = 0.05 
DIR_HEAD_PAN  = -1.0 
WALK_THETA_BIAS = 0.0 
WALK_Y_BIAS     = 0.0

# ------------------------------------------
DEBUG_MODE = True
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 360
TICKRATE = 30 

cx_f, cy_f, area_f = 0.0, 0.0, 0.0
is_walking_active = False 
is_stopped_logic = False
is_aligning_mode = False 
robot = None
robot_weightlift = None

# 紀錄變數
prev_head_pan = 0.0

# IMU 變數紀錄
z_gyro = 0.0
target_yaw = 0.0
last_imu_time = None

def cleanup_node():
    global robot, robot_weightlift
    rospy.logwarn("!!! EMERGENCY STOP !!!")
    if robot:
        try: robot.walkStop()
        except: pass
    rospy.logwarn("!!! ROBOT STOPPED !!!")

def imu_callback(msg):
    global z_gyro, last_imu_time
    current_time = rospy.Time.now()
    if last_imu_time is None:
        last_imu_time = current_time
        return
    dt = (current_time - last_imu_time).to_sec()
    last_imu_time = current_time
    angular_z = msg.angular_velocity.z
    z_gyro += angular_z * dt * 50  # 積分計算 Yaw

# ==========================================
# [新增] 黃線視覺辨識函式
# ==========================================
def find_yellow_line(bgr_img):
    blurred = cv2.GaussianBlur(bgr_img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # 黃色的 HSV 範圍 (若現場光線太暗，可將 S_LO 或 V_LO 的 100 調低到 80)
    Y_H_LO, Y_S_LO, Y_V_LO = 20, 100, 100
    Y_H_HI, Y_S_HI, Y_V_HI = 40, 255, 255
    
    mask = cv2.inRange(hsv, (Y_H_LO, Y_S_LO, Y_V_LO), (Y_H_HI, Y_S_HI, Y_V_HI))
    
    # 線條辨識建議使用方形 Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    
    if not contours: return False, None, 0.0, None, mask
    
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    
    # 黃線比較細長，面積門檻設低一點
    if area < 300: return False, None, area, None, mask
    
    x, y, w, h = cv2.boundingRect(c)
    cx = int(x + w / 2)
    cy = int(y + h / 2)
    return True, (cx, cy), area, (x, y, w, h), mask

def main():
    global cx_f, cy_f, area_f, is_walking_active, is_stopped_logic
    global is_aligning_mode
    global robot, robot_weightlift
    global prev_head_pan
    global z_gyro, target_yaw 

    rospy.init_node("op3_yellow_line_crab")
    rospy.on_shutdown(cleanup_node)
    rospy.Subscriber("/robotis/open_cr/imu", Imu, imu_callback, queue_size=1)

    robot = Robot()
    try:
        robot_weightlift = Robot_weightlift('fira_basketball')
        if not hasattr(robot_weightlift, 'joint_pos'):
            robot_weightlift.joint_pos = {}
    except Exception as e:
        rospy.logerr("Init error: " + str(e))
        return

    rospy.sleep(1.0)
    
    rospy.loginfo("Applying Body Offsets...")
    current_params = robot_weightlift.walking_params[0]
    current_params.init_x_offset += BODY_OFFSET_X
    current_params.init_y_offset += BODY_OFFSET_Y
    current_params.init_z_offset += BODY_OFFSET_Z
    robot_weightlift.walk_set_param_pub.publish(current_params)
    rospy.sleep(1.0)

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
    prev_head_pan = 0.0
    
    # 【關鍵修改】把頭壓低 (-1.0)，讓它看地板的黃線
    robot.setJointPos(["head_pan", "head_tilt"], [prev_head_pan, -1.0])

    rospy.loginfo("Press START button to begin...")
    while not rospy.is_shutdown():
        if robot.buttonCheck("start"): 
            target_yaw = z_gyro # 起步鎖定初始航向
            break
        rospy.sleep(0.1)

    rospy.loginfo("START WALKING...")
    robot.walkStart()
    is_walking_active = True
    
    rate = rospy.Rate(TICKRATE)
    x_cmd_f, y_cmd_f, th_cmd_f = 0.0, 0.0, 0.0
    x_prev, th_prev = 0.0, 0.0
    was_found = False

    while not rospy.is_shutdown():
        if robot.buttonCheck("mode"):
            break

        ret, frame = cap.read()
        if not ret: continue
        image = cv2.resize(frame, (FRAME_W, FRAME_H))
        
        # 使用黃線辨識
        found, cxy, area, bbox, mask = find_yellow_line(image)

        x_cmd = 0.0
        y_cmd = 0.0
        th_cmd = 0.0
        yaw_error = 0.0  # 【Bug 修復】預設給予 0.0，避免 UnboundLocalError
        
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

            # --- 頭部追蹤黃線 (平滑化) ---
            curr_pan = prev_head_pan 
            curr_tilt = robot.joint_pos.get("head_tilt", -1.0) # 預設參考值也要改低
            
            raw_target_pan = curr_pan + (0.35 * DIR_HEAD_PAN * err_x) 
            target_tilt = curr_tilt + (-0.25 * err_y)
            
            smooth_target_pan = prev_head_pan + (raw_target_pan - prev_head_pan) * HEAD_SMOOTH
            smooth_target_pan = max(-1.4, min(1.4, smooth_target_pan))
            target_tilt = max(-1.1, min(0.2, target_tilt))
            
            robot.setJointPos(["head_pan", "head_tilt"], [smooth_target_pan, target_tilt])
            prev_head_pan = smooth_target_pan 
            
            pan_abs = abs(smooth_target_pan)

            # --- 停止邏輯 (看見一大片黃色代表到底了) ---
            if area >= 20000: is_stopped_logic = True
            elif area_f <= 14000: is_stopped_logic = False

            if is_stopped_logic:
                if is_walking_active:
                    robot.walkStop()
                    is_walking_active = False
                    x_cmd, y_cmd, th_cmd = 0.0, 0.0, 0.0
                target_yaw = z_gyro # 停止時不斷更新航向，確保不偏移
            else:
                if not is_walking_active:
                    robot.walkStart()
                    is_walking_active = True
                    target_yaw = z_gyro
                    rospy.sleep(0.5)

            if is_walking_active:
                error = smooth_target_pan

                # --- 狀態機切換邏輯 ---
                if is_aligning_mode:
                    if pan_abs < PAN_GO_EXIT: 
                        is_aligning_mode = False
                        target_yaw = z_gyro # 結束橫移，鎖定現在航向準備直走
                else:
                    if pan_abs > PAN_GO_ENTER: 
                        is_aligning_mode = True
                        target_yaw = z_gyro # 準備橫移，鎖定航向防止亂轉

                # ==========================================
                # 執行步態：全程靠 IMU 鎖死旋轉，僅靠 X 與 Y 移動
                # ==========================================
                
                yaw_error = target_yaw - z_gyro
                imu_th_cmd = np.clip(yaw_error * KP_IMU, -IMU_TH_LIMIT, IMU_TH_LIMIT)

                if is_aligning_mode:
                    # [橫移螃蟹步模式]
                    x_cmd = 0.0  
                    
                    # 【方向修正】如果實測方向還是反的，把正負號對調即可
                    if error > 0:
                        y_cmd = -CRAB_SPEED   # 線偏左，往左橫移
                    else:
                        y_cmd = CRAB_SPEED  # 線偏右，往右橫移
                    
                    # 旋轉完全交給 IMU
                    th_cmd = imu_th_cmd
                    
                else:
                    # [直走模式] 黃線在正前方
                    x_cmd = MAX_X
                    y_cmd = WALK_Y_BIAS
                    
                    # 直走時加入硬體 Bias + IMU 修正
                    th_cmd = WALK_THETA_BIAS + imu_th_cmd

                if th_cmd > TH_MAX: th_cmd = TH_MAX
                if th_cmd < -TH_MAX: th_cmd = -TH_MAX

            if DEBUG_MODE:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 2)
                status_msg = "SIDE-STEP" if is_aligning_mode else "FORWARD"
                col = (0, 255, 255) if is_aligning_mode else (0, 255, 0)
                cv2.putText(image, "Mode: %s" % status_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                
                if is_aligning_mode:
                    cv2.putText(image, "Y_CMD: %.2f" % y_cmd, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1)
                else:
                    cv2.putText(image, "IMU err: %.2f" % yaw_error, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
                
        else:
            was_found = False
            if is_walking_active: 
                x_cmd, y_cmd, th_cmd = 0.0, 0.0, 0.0
                target_yaw = z_gyro

        if is_walking_active:
            x_cmd_f = (1 - ALPHA_CMD) * x_cmd_f + ALPHA_CMD * x_cmd
            y_cmd_f = y_cmd  # 橫移不經過濾波，讓機器人反應更乾脆
            th_cmd_f = (1 - ALPHA_CMD) * th_cmd_f + ALPHA_CMD * th_cmd
            
            dx = x_cmd_f - x_prev
            if dx > X_STEP_MAX: x_cmd_f = x_prev + X_STEP_MAX
            elif dx < -X_STEP_MAX: x_cmd_f = x_prev - X_STEP_MAX
            x_prev = x_cmd_f

            dth = th_cmd_f - th_prev
            if dth > TH_STEP_MAX: th_cmd_f = th_prev + TH_STEP_MAX
            elif dth < -TH_STEP_MAX: th_cmd_f = th_prev - TH_STEP_MAX
            th_prev = th_cmd_f

            # 【重點】呼叫並加入 z_move_amplitude 抬高腳步
            try:
                robot.walkVelocities(x=x_cmd_f, 
                                     y=y_cmd_f, 
                                     th=th_cmd_f, 
                                     z_move_amplitude=Z_MOVE_AMP, 
                                     balance=True, 
                                     z_offset=0)
            except: 
                pass

        if DEBUG_MODE:
            # 顯示辨識遮罩(Mask)幫助調整顏色參數 (可解除註解)
            # cv2.imshow("Mask", mask)
            cv2.imshow("View", image)
            cv2.waitKey(1)
        
        rate.sleep()

if __name__ == "__main__":
    main()