#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import cv2
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cam", type=int, default=0)
parser.add_argument("--out", type=str, default="images")
parser.add_argument("--prefix", type=str, default="right")
parser.add_argument("--interval", type=float, default=0.5)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
args = parser.parse_args()

if not os.path.exists(args.out):
    os.makedirs(args.out)

cap = cv2.VideoCapture(args.cam)

# 設定解析度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

# 嘗試使用 MJPG，避免部分 OP3 相機格式讓 OpenCV 讀不到
try:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
except Exception:
    pass

if not cap.isOpened():
    print("Cannot open camera /dev/video{}".format(args.cam))
    exit(1)

print("Camera opened.")
print("Saving images to:", args.out)
print("Press Ctrl+C to stop.")

count = 0
last_save = 0

# 先丟掉前幾張，讓曝光穩定
for i in range(10):
    cap.read()
    time.sleep(0.05)

try:
    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to read frame.")
            time.sleep(0.1)
            continue

        now = time.time()

        if now - last_save >= args.interval:
            filename = "{}_{:06d}.jpg".format(args.prefix, count)
            path = os.path.join(args.out, filename)

            cv2.imwrite(path, frame)

            print("Saved:", path)
            count += 1
            last_save = now

except KeyboardInterrupt:
    print("\nStopped by user.")

cap.release()
print("Camera released.")