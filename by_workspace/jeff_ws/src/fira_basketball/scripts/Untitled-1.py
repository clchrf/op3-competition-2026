#! /usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from op3_ros_utils import Robot

# ==========================================
# 參數設定區
# ==========================================
DEBUG_MODE = True
MIN_AREA = 800  # 判定為紅線的最小像素面積 (可視情況微調)

# ==========================================
# 狀態機定義
# ==========================================
class States:
    INIT = -1
    READY = 0          
    SEARCH_RED = 1     
    SIDEWALK_RIGHT = 2 
    END = 99

# ==========================================
# 初始化節點與系統
# ==========================================
rospy.init_node("red_line_sidewalk")
robot = Robot()
bridge = CvBridge()

# 全域變數，用來儲存影像辨識結果
found_red = False
debug_frame = None

print("等待節點註冊...")
rospy.sleep(3) 

def init():
    robot.setGeneralControlModule("walking_module")
    robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
    robot.setJointPos(["head_tilt"], [-0.3])
    robot.setJointPos(["head_pan"], [0.0])
    rospy.sleep(1.0)
    print("[系統] 初始化完成！")

# ==========================================
# 📷 內建視覺系統 (直接取代原本的 VisionSystem)
# ==========================================
def image_callback(msg):
    global found_red, debug_frame
    
    try:
        # 將 ROS 影像格式轉換為 OpenCV 格式
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        return

    # 1. 轉換為 HSV 色彩空間
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. 嚴格的紅色範圍 (包含 HSV 的兩端)
    # 提高 Saturation(飽和度) 到 150，過濾掉偏白/粉紅的雜訊
    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([5, 255, 255])  # 色相縮窄到 5
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([175, 150, 100])
    upper_red2 = np.array([180, 255, 255]) # 色相縮窄到 175-180
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 合併兩個紅色遮罩
    mask = mask1 + mask2

    # 3. 尋找輪廓 (計算紅色區塊面積)
    # 3. 尋找輪廓 (計算紅色區塊面積)
    # 針對 OpenCV 3.x 的寫法，接收 3 個回傳值
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    current_found = False
    if len(contours) > 0:
        # 找出最大的紅色區塊
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        
        if area > MIN_AREA:
            current_found = True
            # 在 Debug 畫面上畫出綠色外框方便觀察
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            
    # 更新全域變數供主迴圈使用
    found_red = current_found
    
    # 儲存處理好的影像供顯示
    if DEBUG_MODE:
        debug_frame = frame

# 訂閱攝影機影像
rospy.Subscriber("/cv_camera/image_raw", Image, image_callback, queue_size=1)

# ==========================================
# 主迴圈
# ==========================================
tickrate = 60
rate = rospy.Rate(tickrate)

currState = States.INIT
start_time = 0.0  

while not rospy.is_shutdown():
    
    # 【防呆機制】按下 Mode 按鈕強制重置
    if robot.buttonCheck("mode"):
        print("\n[系統] 收到 Mode 指令，強制重置！")
        robot.walkStop()
        currState = States.INIT
    
    # 【Debug 影像顯示】
    if DEBUG_MODE and debug_frame is not None:
        cv2.imshow("Strict Red Detection", debug_frame)
        cv2.waitKey(1)

    # ---------------- 狀態機邏輯 ----------------
    if currState == States.INIT:
        print("[INIT] 準備階段")
        init()
        currState = States.READY

    elif currState == States.READY:
        if robot.buttonCheck("start"):
            print("[READY] 收到 Start 指令，開始尋找嚴格定義的紅線！")
            robot.walkStart()
            currState = States.SEARCH_RED
            
    elif currState == States.SEARCH_RED:
        if found_red:
            print("[SEARCH_RED] 🎯 偵測到紅線！頭部右轉，啟動右側移 5 秒")
            start_time = rospy.Time.now().to_sec()
            
            # 頭部往右轉
            robot.setJointPos(["head_pan"], [-1.0]) 
            currState = States.SIDEWALK_RIGHT
        else:
            # 原地踏步尋找
            robot.walkVelocities(x=0.0, y=0.0, th=0, balance=True)

    elif currState == States.SIDEWALK_RIGHT:
        elapsed_time = rospy.Time.now().to_sec() - start_time
        
        if elapsed_time >= 5.0:
            print("[SIDEWALK_RIGHT] ⏱️ 5 秒時間到！煞車")
            robot.walkStop()
            rospy.sleep(1.0) 
            print("[END] 任務完成，停機待命。")
            currState = States.END 
        else:
            # 往右側移
            robot.walkVelocities(x=0.0, y=-20.0, th=0, balance=True)

    elif currState == States.END:
        pass

    rate.sleep()