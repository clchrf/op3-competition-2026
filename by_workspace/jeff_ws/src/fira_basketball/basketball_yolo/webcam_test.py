#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本機 webcam 測試（吃 CPU 即可，現在就能跑）。

兩種用法：
1) 還沒訓練：用 COCO 預訓練的 yolo11n.pt，橘色籃球會被當成 "sports ball" 偵測到，
   先驗證整條偵測管線通不通。
2) 訓練完：把 WEIGHTS 換成 runs/basketball/weights/best.pt，就會偵測 ball / hoop。

重點：這支印出來的 (cx, cy) 就是之後要餵進機器人對準邏輯的「框中心」，
介面跟現在 find_ball / find_board 算出的 bcx, bcy 完全一致。
"""
import cv2
from ultralytics import YOLO

WEIGHTS = "yolo11n.pt"          # 訓練後改成 "runs/basketball/weights/best.pt"
IMG_SIZE = 416                  # 推論輸入尺寸，越小越快（之後在機器人上也用這個值）
CONF = 0.4                      # 信心門檻
CAM_INDEX = 0

# 用 stock 權重時，只關心「球」這一類（COCO 的 sports ball = 32）；
# 換成自己訓練的權重後，把這行設成 None 就會顯示所有類別。
ONLY_CLASSES = {32}            # 自訓練權重請改成 None

def main():
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 對齊機器人端假設的 640x480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("相機開不起來，檢查 CAM_INDEX")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        res = model.predict(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)[0]
        for b in res.boxes:
            cls = int(b.cls[0])
            if ONLY_CLASSES is not None and cls not in ONLY_CLASSES:
                continue
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2     # <-- 這就是之後要送出的座標
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            label = "%s %.2f (%d,%d)" % (model.names[cls], conf, cx, cy)
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("yolo test (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:   # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
