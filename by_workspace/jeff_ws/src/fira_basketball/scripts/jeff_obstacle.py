#! /usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
from op3_ros_utils import Robot

def empty_callback(x):
    pass

class States:
    INIT = 0
    READY = 1
    WALK_STRAIGHT = 2
    AVOID_OBSTACLE = 3

class BlueObstacleAvoider:
    def __init__(self):
        rospy.init_node("op3_blue_avoider_tune")
        self.robot = Robot()
        self.bridge = CvBridge()
        self.rate = rospy.Rate(30)
        
        # IMU 參數
        self.z_gyro = 0.0
        self.target_yaw = 0.0
        self.last_imu_time = rospy.Time.now()
        
        # 視覺變數
        self.obstacle_cx = 0.0
        self.obstacle_area = 0.0
        self.area_threshold = 2000
        self.safe_threshold = 1000
        self.img_width = 640
        self.state = States.INIT

        # === 建立調色盤視窗 ===
        cv2.namedWindow("Trackbars")
        cv2.resizeWindow("Trackbars", 400, 300)
        # 預設給一個很寬的藍色範圍
        cv2.createTrackbar("H_MIN", "Trackbars", 90, 180, empty_callback)
        cv2.createTrackbar("H_MAX", "Trackbars", 130, 180, empty_callback)
        cv2.createTrackbar("S_MIN", "Trackbars", 50, 255, empty_callback)
        cv2.createTrackbar("S_MAX", "Trackbars", 255, 255, empty_callback)
        cv2.createTrackbar("V_MIN", "Trackbars", 50, 255, empty_callback)
        cv2.createTrackbar("V_MAX", "Trackbars", 255, 255, empty_callback)
        
        rospy.Subscriber("/robotis/open_cr/imu", Imu, self.imu_callback, queue_size=1)
        # 確保你的相機 Topic 是這個，如果沒畫面可能要改成 /usb_cam/image_raw
        rospy.Subscriber("/cv_camera/image_raw", Image, self.image_callback, queue_size=1)
        rospy.sleep(2)

    def imu_callback(self, msg):
        current_time = rospy.Time.now()
        dt = (current_time - self.last_imu_time).to_sec()
        self.last_imu_time = current_time
        self.z_gyro += msg.angular_velocity.z * dt * 50 

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.img_width = cv_image.shape[1]
        except CvBridgeError as e:
            return

        # 從滑桿即時讀取 HSV 數值
        h_min = cv2.getTrackbarPos("H_MIN", "Trackbars")
        h_max = cv2.getTrackbarPos("H_MAX", "Trackbars")
        s_min = cv2.getTrackbarPos("S_MIN", "Trackbars")
        s_max = cv2.getTrackbarPos("S_MAX", "Trackbars")
        v_min = cv2.getTrackbarPos("V_MIN", "Trackbars")
        v_max = cv2.getTrackbarPos("V_MAX", "Trackbars")

        blue_lower = np.array([h_min, s_min, v_min])
        blue_upper = np.array([h_max, s_max, v_max])

        hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, blue_lower, blue_upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))

        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            self.obstacle_area = cv2.contourArea(largest_contour)
            if self.obstacle_area > 500:
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    self.obstacle_cx = int(M["m10"] / M["m00"])
                    # 在原圖上畫出紅色中心點跟綠色框框，方便確認有沒有抓對
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    cv2.rectangle(cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(cv_image, (self.obstacle_cx, int(M["m01"]/M["m00"])), 5, (0, 0, 255), -1)
        else:
            self.obstacle_area = 0.0

        # === 顯示畫面 ===
        cv2.imshow("Raw Image", cv_image) # 顯示機器人看到的彩色原圖
        cv2.imshow("Blue Mask", mask)     # 顯示過濾後的黑白遮罩
        cv2.waitKey(1)

    def run(self):
        while not rospy.is_shutdown():
            if self.robot.buttonCheck("mode"):
                self.state = States.INIT
                self.robot.walkStop()

            if self.state == States.INIT:
                self.robot.setGeneralControlModule("head_control_module")
                self.robot.setJointsControlModule(["head_pan", "head_tilt"], ["none", "none"])
                self.robot.setJointPos(["head_tilt", "head_pan"], [np.radians(-30), 0])
                self.robot.setGeneralControlModule("walking_module")
                self.robot.walkStop()
                rospy.loginfo("[系統] 正在調色中... 調整好後請按下 START。")
                self.state = States.READY

            elif self.state == States.READY:
                if self.robot.buttonCheck("start"):
                    self.target_yaw = self.z_gyro
                    self.robot.walkStart()
                    self.state = States.WALK_STRAIGHT
                    rospy.loginfo("[系統] 開始直行！")

            elif self.state == States.WALK_STRAIGHT:
                if self.obstacle_area > self.area_threshold:
                    self.state = States.AVOID_OBSTACLE
                    rospy.logwarn("[警報] 偵測到藍色障礙物！面積: %d -> 開始閃避" % self.obstacle_area)
                else:
                    yaw_error = self.target_yaw - self.z_gyro
                    th_correction = np.clip(yaw_error * 0.5, -10, 10) 
                    self.robot.walkVelocities(x=15, y=0, th=th_correction, z_move_amplitude=0.045, balance=True, z_offset=0)
                    rospy.loginfo_throttle(0.5, "[直行中] 藍色面積: %d | 航向誤差: %.2f" % (self.obstacle_area, yaw_error))

            elif self.state == States.AVOID_OBSTACLE:
                if self.obstacle_area < self.safe_threshold:
                    self.target_yaw = self.z_gyro
                    self.state = States.WALK_STRAIGHT
                    rospy.loginfo("[系統] 障礙物已遠離。恢復直行。")
                else:
                    center_offset = self.obstacle_cx - (self.img_width / 2)
                    if center_offset < 0:
                        self.robot.walkVelocities(x=5, y=-15, th=-5, z_move_amplitude=0.045, balance=True, z_offset=0)
                        rospy.loginfo_throttle(0.5, "[閃避中] 障礙在左 -> 向右平移")
                    else:
                        self.robot.walkVelocities(x=5, y=15, th=5, z_move_amplitude=0.045, balance=True, z_offset=0)
                        rospy.loginfo_throttle(0.5, "[閃避中] 障礙在右 -> 向左平移")

            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = BlueObstacleAvoider()
        node.run()
    except rospy.ROSInterruptException:
        pass