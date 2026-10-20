#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

# 创建输出目录
output_dir = "cropped_contours"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开相机
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开相机")
    exit()

cut_bount = (np.array([0, 0, 0]), np.array([100, 255, 50]))  # 绿色范围

def cut_and_scale_pic(img, color_bount, scale_percent=100):
    # 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义 HSV 范围
    lower_black = color_bount[0]  # 最低范围
    upper_black = color_bount[1]  # 最高范围

    # 创建遮罩
    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3)))

    cv2.imshow('Black Mask Cut', mask)

    # 找出轮廓
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制轮廓及框出矩形
    cropped = None
    have_cropped = False
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if 25000 > cv2.contourArea(contour) > 5000:  # 设置面积阈值
            have_cropped = True
            padding = 10  # 边界扩展
            x_start = max(x - padding, 0)
            y_start = max(y - padding, 0)
            x_end = min(x + w + padding, img.shape[1])
            y_end = min(y + h + padding, img.shape[0])

            # 裁剪图像
            cropped = img.copy()
            cropped = cropped[y_start:y_end, x_start:x_end]

            width = int(cropped.shape[1] * scale_percent / 100)
            height = int(cropped.shape[0] * scale_percent / 100)
            resized_cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
            break
    if not have_cropped:
        return False, img

    return True, resized_cropped

def find_ans(img, bount, det_yellow=False, dec_middle=True, approxPolyDP_val=10):
    ans = "unknown"

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (600, 300))
        cv2.imshow('gray', gray)

        center_x = gray.shape[1] // 2  # 图像中心

        # 创建遮罩
        _, mask = cv2.threshold(gray, bount, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3)))
        cv2.imshow('Black Mask Ans', mask)

        # 找出轮廓
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            contour = max(_, contours, key=cv2.contourArea)
            contour = cv2.approxPolyDP(contour, approxPolyDP_val, True)

            sorted_points = sorted(contour[:, 0, :], key=lambda point: point[1], reverse=True)
            bottom_points = sorted_points[:5]

            top_point = sorted_points[4]
            sec_top_point = sorted_points[3]
            cv2.circle(gray, tuple(top_point), 10, (255), 1)
            cv2.circle(gray, tuple(sec_top_point), 10, (255), 1)

            if abs(top_point[1] - sec_top_point[1]) > 20:
                bottom_points = bottom_points[:4]

            for point in bottom_points:
                cv2.circle(gray, tuple(point), 5, (255), -1)

            sorted_points = sorted(bottom_points, key=lambda point: abs(point[0] - center_x))
            middle_point = sorted_points[0]
            cv2.circle(gray, tuple(middle_point), 15, (255), 1)
            other_points = sorted_points[1:]

            left = 0
            right = 0
            dir = 1
            if (middle_point[0] > center_x) and dec_middle:
                dir = -1
            for point in other_points:
                if point[0] < center_x:
                    left += dir
                else:
                    right += dir
            if left > right:
                ans = "red"
            elif right == left:
                if det_yellow:
                    ans = "yellow"
            else:
                ans = "blue"
        cv2.imshow('img with contour', gray)
        return ans
    except Exception as e:
        print(e)
        return ans

frame_count = 0
while True:
    ret, img = cap.read()
    if not ret:
        print("无法读取相机画面")
        break

    ret, cut_pic = cut_and_scale_pic(img, cut_bount)
    if ret:
        ans = find_ans(cut_pic, bount=90, det_yellow=True)
        cv2.putText(img, ans, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Camera Feed', img)
    if ret:
        cv2.imshow('cut_pic', cut_pic)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
