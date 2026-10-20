#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
從相機抓畫面存成訓練用圖片（Python 2.7 / 3 都能跑，可直接放機器人那台 NUC 抓那顆鏡頭）。

操作：
  空白鍵 = 存目前這張
  a      = 切換自動連拍（每 AUTO_EVERY 幀存一張）
  q/ESC  = 離開

存出來是正常彩色圖（不是 HSV），檔名自動編號，放到 RAW_DIR。
記得涵蓋不同距離、角度、光線，球從遠到很近、籃框正面/側面都要拍。
"""
from __future__ import print_function
import os
import cv2

CAM_INDEX = 0
RAW_DIR = "dataset/raw"        # 抓到的原始圖先放這，標註後再分到 train/val
PREFIX = "robot"               # 檔名前綴，分批拍可改成 robot_day1 之類避免覆蓋
AUTO_EVERY = 10                # 自動連拍時每幾幀存一張


def next_index(d, prefix):
    os.makedirs(d) if not os.path.isdir(d) else None
    n = 0
    for f in os.listdir(d):
        if f.startswith(prefix) and f.endswith(".jpg"):
            try:
                n = max(n, int(f[len(prefix) + 1:-4]) + 1)
            except ValueError:
                pass
    return n


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # 對齊機器人端假設的 640x480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("相機開不起來，檢查 CAM_INDEX")
        return

    idx = next_index(RAW_DIR, PREFIX)
    auto = False
    frame_no = 0
    print("空白鍵=存圖  a=自動連拍開關  q/ESC=離開")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1

        view = frame.copy()
        cv2.putText(view, "saved:%d  auto:%s" % (idx, auto), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("capture (space=save, a=auto, q=quit)", view)

        save = False
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('a'):
            auto = not auto
        elif key == ord(' '):
            save = True
        if auto and frame_no % AUTO_EVERY == 0:
            save = True

        if save:
            path = os.path.join(RAW_DIR, "%s_%05d.jpg" % (PREFIX, idx))
            cv2.imwrite(path, frame)      # 存原始 BGR 彩色圖
            print("saved", path)
            idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
