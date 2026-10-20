#! /usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from op3_ros_utils import Robot
import cv2
import numpy as np

# 填入你剛才校準好的數值
Y_LOWER = np.array([20, 100, 100]) 
Y_UPPER = np.array([40, 255, 255])

rospy.init_node("fira_line_stop")
robot = Robot()
cap = cv2.VideoCapture(0)

robot.setGeneralControlModule("walking_module")
is_running = False

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    # 我們將畫面分為「遠方（導引）」和「近處（判斷停止）」
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, Y_LOWER, Y_UPPER)

    # 尋找最大的黃色色塊
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        M = cv2.moments(c)
        
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # --- 停止邏輯 ---
            # 如果色塊面積太大（代表線就在腳下）或者重心位置非常靠下
            if area > (h * w * 0.2) or cY > (h * 0.85):
                print("到達黃線！停止移動。")
                robot.walkStop()
                is_running = False
            
            # --- 循線邏輯 ---
            elif is_running:
                error = (cX - (w / 2)) / (w / 2)
                turn_vel = error * -15.0 # P 控制
                robot.walkVelocities(x=8, y=0, th=np.clip(turn_vel, -10, 10), balance=True)

    # 按鈕啟動
    if robot.buttonCheck("start") and not is_running:
        print("開始循線...")
        robot.walkStart()
        is_running = True

    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

robot.walkStop()