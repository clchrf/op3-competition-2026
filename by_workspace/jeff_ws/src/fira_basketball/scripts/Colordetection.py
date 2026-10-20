#! /usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np


def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow("Trackbars")

# 黃色初步範圍：H(20-40), S(100-255), V(100-255)
cv2.createTrackbar("L-H", "Trackbars", 20, 180, nothing)
cv2.createTrackbar("L-S", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 40, 180, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

print("調整拉桿直到 Mask 畫面正確顯示黃線，然後記下這些數值。")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")
    
    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])
    
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", res)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("最終數值: lower = [{}, {}, {}], upper = [{}, {}, {}]".format(l_h,l_s,l_v,u_h,u_s,u_v))
        break

cap.release()
cv2.destroyAllWindows()