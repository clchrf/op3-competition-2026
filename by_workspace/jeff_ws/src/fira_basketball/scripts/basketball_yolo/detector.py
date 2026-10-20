#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OP3 偵測程序（在 conda 的 yolo 環境跑，Python 3.10）。

職責：接管相機 -> OpenVINO 推論 -> 把最新的 ball / hoop 偵測用 UDP 丟給
Python 2.7 控制節點。控制節點本身不再開相機。

啟動（在 conda yolo 環境，先避開 ROS 的 cv2）：
    unset PYTHONPATH
    python detector.py --model best_openvino_model --imgsz 320 --show
正式跑時拿掉 --show（省 CPU）。
"""
import sys
# 把 ROS 的 python2 路徑從 sys.path 拿掉，避免 import 到給 Py2 編的 cv2.so
sys.path = [p for p in sys.path if "/opt/ros" not in p]

import time
import json
import socket
import argparse
import cv2
from ultralytics import YOLO

# ---- 預設設定 ----
MODEL = "best_openvino_model"   # 你訓練後匯出的 OpenVINO 資料夾
IMGSZ = 320
CONF = 0.35
CAM_INDEX = 0
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
CLASS_BALL = 0                  # data.yaml: 球=0
CLASS_HOOP = 1                  # data.yaml: 紅色圓柱籃框=1


def best_box(result, cls_id):
    """回傳該類別信心最高的框 (cx, cy, w, h, conf)，沒有就 None"""
    best = None
    for b in result.boxes:
        if int(b.cls[0]) != cls_id:
            continue
        conf = float(b.conf[0])
        if best is None or conf > best[4]:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            best = [(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1, conf]
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--cam", type=int, default=CAM_INDEX)
    ap.add_argument("--port", type=int, default=UDP_PORT)
    ap.add_argument("--ball", type=int, default=CLASS_BALL)
    ap.add_argument("--hoop", type=int, default=CLASS_HOOP)
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    model = YOLO(a.model, task="detect")
    cap = cv2.VideoCapture(a.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("相機開不起來，檢查 --cam")
        return

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (UDP_IP, a.port)
    print("detector running -> udp %s:%d  model=%s imgsz=%d"
          % (UDP_IP, a.port, a.model, a.imgsz))

    n, t0 = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        res = model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        ball = best_box(res, a.ball)
        hoop = best_box(res, a.hoop)

        msg = {"t": time.time(), "ball": ball, "hoop": hoop}
        sock.sendto(json.dumps(msg).encode("utf-8"), addr)

        # 簡單印個實際 FPS（每 30 幀）
        n += 1
        if n % 30 == 0:
            fps = 30.0 / (time.time() - t0)
            t0 = time.time()
            print("fps %.1f  ball=%s hoop=%s"
                  % (fps, "Y" if ball else "-", "Y" if hoop else "-"))

        if a.show:
            for tag, box, color in (("ball", ball, (0, 255, 0)),
                                    ("hoop", hoop, (0, 0, 255))):
                if box:
                    cx, cy, w, h, cf = box
                    p1 = (int(cx - w / 2), int(cy - h / 2))
                    p2 = (int(cx + w / 2), int(cy + h / 2))
                    cv2.rectangle(frame, p1, p2, color, 2)
                    cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
                    cv2.putText(frame, "%s %.2f" % (tag, cf), p1,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow("detector (ESC quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
