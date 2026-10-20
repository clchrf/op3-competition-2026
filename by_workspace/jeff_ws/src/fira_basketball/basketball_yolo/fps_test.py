#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在機器人那台 NUC 上量 YOLO 推論速度（在 conda 的 yolo 環境裡跑）。

用法：
  python fps_test.py                                   # 量 yolo11n.pt @416
  python fps_test.py --imgsz 320                        # 換輸入尺寸
  python fps_test.py --model yolo11n_openvino_model     # 量匯出後的 OpenVINO 版

判讀：
  - 純 PyTorch CPU 通常最慢；OpenVINO（尤其 INT8）會快很多。
  - 大概抓：<5 FPS 表示一定要 OpenVINO/INT8 + 偵測獨立執行緒；
    10~15+ FPS 表示已經可用。
"""
import time
import argparse
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()

    m = YOLO(a.model)
    img = np.zeros((480, 640, 3), dtype=np.uint8)   # 假畫面，只量推論速度

    for _ in range(5):                               # 暖機
        m.predict(img, imgsz=a.imgsz, verbose=False)

    t = time.time()
    for _ in range(a.n):
        m.predict(img, imgsz=a.imgsz, verbose=False)
    dt = (time.time() - t) / a.n

    print("model=%s imgsz=%d  ->  %.1f ms/frame, %.1f FPS"
          % (a.model, a.imgsz, dt * 1000.0, 1.0 / dt))


if __name__ == "__main__":
    main()
