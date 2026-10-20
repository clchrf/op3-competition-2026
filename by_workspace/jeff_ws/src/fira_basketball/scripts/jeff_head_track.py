#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
import rospy

from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from robotis_controller_msgs.msg import JointCtrlModule


# =========================================================
# 你只需要改這一區（參數區）
# =========================================================
IMAGE_TOPIC = "/cv_camera/image_raw"

DEBUG_MODE = True

# 紅色 HSV（紅色 hue 會跨 0/179，所以要兩段）
RED1_LO = np.array([0,   120, 70])
RED1_HI = np.array([10,  255, 255])
RED2_LO = np.array([170, 120, 70])
RED2_HI = np.array([179, 255, 255])

MIN_AREA = 250          # 太小不追（避免雜訊）
DEADBAND = 0.03         # 死區（避免抖）

# P 控制增益（先保守，避免抖很大）
KP_YAW   = 0.9          # 左右
KP_PITCH = 0.7          # 上下

# 單次更新最大步進（rad），避免一下轉太多
MAX_STEP_YAW   = 0.08
MAX_STEP_PITCH = 0.06

# 頭部角度限制（rad）— 依你機器可動範圍可再調
YAW_MIN,   YAW_MAX   = -1.6, 1.6
PITCH_MIN, PITCH_MAX = -0.8, 0.6

# 找不到目標時要不要「掃描找球」
ENABLE_SCAN_WHEN_LOST = True
SCAN_SPEED = 0.02       # 每次掃描 yaw 增量（rad）
SCAN_YAW_RANGE = 1.2    # 掃描範圍（±range）
# =========================================================


class OP3HeadTrackRed:
    def __init__(self):
        self.bridge = CvBridge()
        self.last_img = None

        # 目前頭部目標角度
        self.yaw = 0.0
        self.pitch = 0.0

        # 掃描用
        self.scan_dir = +1

        # Publishers
        self.pub_module = rospy.Publisher(
            "/robotis/set_joint_ctrl_modules", JointCtrlModule, queue_size=1
        )
        self.pub_head = rospy.Publisher(
            "/robotis/head_control/set_joint_states", JointState, queue_size=1
        )

        # Subscriber
        rospy.Subscriber(IMAGE_TOPIC, Image, self.cb_img, queue_size=1)

        # 保險：等一下 publisher/subscriber 建立
        rospy.sleep(0.5)
        self.ensure_head_module()

        rospy.loginfo("[OP3HeadTrackRed] Ready. Subscribing: %s", IMAGE_TOPIC)

    def cb_img(self, msg):
        try:
            self.last_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridgeError: %s", str(e))
            self.last_img = None

    def ensure_head_module(self):
        # 等 controller 相關 topic 起來（如果沒起來也不致命）
        try:
            rospy.wait_for_message("/robotis/present_joint_states", JointState, timeout=3.0)
        except rospy.ROSException:
            rospy.logwarn("[OP3HeadTrackRed] /robotis/present_joint_states not ready yet, continue anyway.")

        msg = JointCtrlModule()
        msg.joint_name = ["head_pan", "head_tilt"]
        msg.module_name = ["head_control_module", "head_control_module"]

        # 多發幾次更保險（避免啟動初期丟包）
        for _ in range(5):
            self.pub_module.publish(msg)
            rospy.sleep(0.1)

        rospy.loginfo("[OP3HeadTrackRed] Switched head_pan/head_tilt to head_control_module")

    def detect_red_centroid(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        m1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
        m2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
        mask = cv2.bitwise_or(m1, m2)

        # morphology 去雜訊
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, mask, 0

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            return None, mask, area

        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            return None, mask, area

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy), mask, area

    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def publish_head(self):
        msg = JointState()
        msg.name = ["head_pan", "head_tilt"]
        msg.position = [float(self.yaw), float(self.pitch)]
        self.pub_head.publish(msg)

    def track_step(self, err_x, err_y):
        # 死區避免抖
        if abs(err_x) < DEADBAND:
            err_x = 0.0
        if abs(err_y) < DEADBAND:
            err_y = 0.0

        # 控制方向：
        # 常見情況：目標在右邊(err_x>0) → 頭往右轉通常需要 yaw 減少（所以有負號）
        # 如果你發現方向相反，把 yaw_delta 或 pitch_delta 的負號改掉即可。
        yaw_delta = -KP_YAW * err_x
        pitch_delta = -KP_PITCH * err_y

        yaw_delta = self.clamp(yaw_delta, -MAX_STEP_YAW, MAX_STEP_YAW)
        pitch_delta = self.clamp(pitch_delta, -MAX_STEP_PITCH, MAX_STEP_PITCH)

        self.yaw = self.clamp(self.yaw + yaw_delta, YAW_MIN, YAW_MAX)
        self.pitch = self.clamp(self.pitch + pitch_delta, PITCH_MIN, PITCH_MAX)

    def scan_step(self):
        # 找不到目標：左右掃描
        if not ENABLE_SCAN_WHEN_LOST:
            return

        if self.yaw > +SCAN_YAW_RANGE:
            self.scan_dir = -1
        elif self.yaw < -SCAN_YAW_RANGE:
            self.scan_dir = +1

        self.yaw = self.clamp(self.yaw + self.scan_dir * SCAN_SPEED, YAW_MIN, YAW_MAX)
        # pitch 可以固定一個角度或慢慢回中
        self.pitch = self.clamp(self.pitch * 0.98, PITCH_MIN, PITCH_MAX)

    def spin(self):
        rate = rospy.Rate(60)

        while not rospy.is_shutdown():
            if self.last_img is None:
                rate.sleep()
                continue

            img = self.last_img.copy()
            h, w = img.shape[:2]
            center = (w // 2, h // 2)

            centroid, mask, area = self.detect_red_centroid(img)

            if centroid is not None:
                cx, cy = centroid
                err_x = (cx - center[0]) / float(center[0])  # -1 ~ +1
                err_y = (cy - center[1]) / float(center[1])

                self.track_step(err_x, err_y)
                self.publish_head()

                if DEBUG_MODE:
                    cv2.circle(img, center, 5, (255, 255, 255), -1)
                    cv2.circle(img, centroid, 6, (0, 255, 0), -1)
                    cv2.line(img, center, centroid, (255, 0, 0), 2)

                    text = "FOUND area={:.0f} err=({:+.2f},{:+.2f}) yaw={:+.2f} pitch={:+.2f}".format(
                        area, err_x, err_y, self.yaw, self.pitch
                    )

                    cv2.putText(
                        img,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
            else:
                # 找不到就掃描
                self.scan_step()
                self.publish_head()

                if DEBUG_MODE:
                    cv2.circle(img, center, 5, (255, 255, 255), -1)

                    text = "LOST scanning... yaw={:+.2f} pitch={:+.2f}".format(
                        self.yaw, self.pitch
                    )

                    cv2.putText(
                        img,
                        text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                    )

            if DEBUG_MODE:
                cv2.imshow("camera", img)
                cv2.imshow("red_mask", mask)
                cv2.waitKey(1)

            rate.sleep()



def main():
    rospy.init_node("op3_head_track_red", anonymous=False)
    node = OP3HeadTrackRed()
    rospy.sleep(0.5)
    node.spin()


if __name__ == "__main__":
    main()
