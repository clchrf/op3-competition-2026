#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detector.py  ── 偵測獨立程序(Python 3)v2
  YOLO11n-OBB + OpenVINO + imgsz 320

v2 改動(配合多階段停線需求):
  - bar 維持只送信心最高的那一個
  - line 改成「送出畫面上所有偵測到的白線」(list),
    控制節點自己依階段挑想要的那條(通常取 cY 最大=最近的)。
  - 多附送 fw/fh,Py2 端可正確算 offset。

封包格式(JSON, UDP):
{
  "t": 時戳, "fw": 640, "fh": 480,
  "bar":   {"found": true, "cx":..., "cy":..., "angle":..., "conf":...},
  "lines": [   # 已按 cy 由大到小排(近→遠)
    {"cx":..., "cy":..., "angle":..., "conf":...},
    ...
  ]
}

啟動順序:先跑這支,再跑控制節點。
"""

import socket, json, time
import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
#  設定區
# ============================================================
MODEL_PATH      = "best_openvino_model"   # OP3 上的 OpenVINO IR 資料夾
CAMERA_INDEX    = 0
IMGSZ           = 320
CONF_THRES      = 0.45

DET_TARGET_HOST = "127.0.0.1"
DET_TARGET_PORT = 5005

SHOW_DEBUG      = False
PRINT_FPS       = True

CLASS_BAR  = "bar_target"
CLASS_LINE = "white_line"

# ============================================================
#  初始化
# ============================================================
sock  = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
model = YOLO(MODEL_PATH, task="obb")
cap   = cv2.VideoCapture(CAMERA_INDEX)


def obb_center_angle(pts):
    """pts: 4x2 角點 → (cx, cy, angle_deg),angle 折到 -45~+45,水平=0。"""
    cx, cy = pts.mean(axis=0)
    best_len, best_edge = -1.0, None
    for i in range(4):
        p1, p2 = pts[i], pts[(i + 1) % 4]
        d = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if d > best_len:
            best_len, best_edge = d, (p1, p2)
    (x1, y1), (x2, y2) = best_edge
    ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    while ang > 45:
        ang -= 90
    while ang < -45:
        ang += 90
    return float(cx), float(cy), ang


def parse(res):
    """bar 取信心最高一個;line 全部送出。"""
    bar   = {"found": False}
    lines = []
    obb = res.obb
    if obb is None or obb.cls is None:
        return bar, lines
    n = len(obb.cls)
    for i in range(n):
        conf = float(obb.conf[i])
        if conf < CONF_THRES:
            continue
        name = model.names[int(obb.cls[i])]
        pts  = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
        cx, cy, ang = obb_center_angle(pts)
        rec = {"found": True, "cx": int(round(cx)), "cy": int(round(cy)),
               "angle": round(ang, 2), "conf": round(conf, 3),
               "pts": pts.astype(int).tolist()}
        if name == CLASS_BAR and (not bar["found"] or conf > bar.get("conf", 0)):
            bar = rec
        elif name == CLASS_LINE:
            lines.append(rec)
    # 按 cY 由大到小排(近的在前),控制節點預設取 lines[0] 就是最近的線
    lines.sort(key=lambda r: -r["cy"])
    return bar, lines


def draw(frame, rec, color, tag):
    if rec.get("found"):
        pts = np.array(rec["pts"], dtype=np.int32)
        cv2.polylines(frame, [pts], True, color, 2)
        cv2.circle(frame, (rec["cx"], rec["cy"]), 5, color, -1)
        cv2.putText(frame, "{} {:.0f}deg cY={}".format(tag, rec["angle"], rec["cy"]),
                    (rec["cx"] + 6, rec["cy"]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# ============================================================
#  主迴圈
# ============================================================
def main():
    t_last, fps = time.time(), 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        res = model(frame, imgsz=IMGSZ, verbose=False)[0]
        bar, lines = parse(res)

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        msg = {
            "t": time.time(), "fw": fw, "fh": fh,
            "bar":   {k: bar[k] for k in bar if k != "pts"},
            "lines": [{k: l[k] for k in l if k != "pts"} for l in lines],
        }
        sock.sendto(json.dumps(msg).encode(), (DET_TARGET_HOST, DET_TARGET_PORT))

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-6))
        t_last = now
        if PRINT_FPS:
            print("[DET] fps={:.1f} bar={} lines={} (cY: {})".format(
                fps,
                (bar.get("cx"), bar.get("cy"), bar.get("angle")) if bar["found"] else None,
                len(lines),
                [l["cy"] for l in lines]))

        if SHOW_DEBUG:
            draw(frame, bar, (0, 0, 255), "BAR")
            for idx, l in enumerate(lines):
                color = (255, 0, 0) if idx == 0 else (200, 200, 0)  # 最近的線藍色,其他青色
                draw(frame, l, color, "LINE#{}".format(idx))
            cv2.imshow("detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()