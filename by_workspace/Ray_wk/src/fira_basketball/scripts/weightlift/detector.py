#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import time
import json
import socket
import threading
import numpy as np
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from openvino import Core

# ============================================================
# 設定區
# ============================================================

MODEL_PATH = "best_openvino_model/best.xml"
CAM_ID = 0

IMG_SIZE = 320
CONF_THRES = 0.25
IOU_THRES = 0.45

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

# 開啟本機瀏覽器畫面串流，不需要網際網路
SHOW_WEB_STREAM = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8080

CLASS_NAMES = {
    0: "bar_target",
    1: "white_line",
}

latest_jpeg = None
latest_lock = threading.Lock()


# ============================================================
# Web stream：用瀏覽器看畫面
# ============================================================

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        global latest_jpeg

        if self.path == "/" or self.path == "/index.html":
            html = """
            <html>
            <head>
                <title>OP3 YOLO Detector</title>
                <style>
                    body {
                        background: #111;
                        color: white;
                        font-family: Arial, sans-serif;
                        text-align: center;
                    }
                    img {
                        width: 960px;
                        max-width: 95%;
                        border: 2px solid #555;
                    }
                    .hint {
                        color: #ccc;
                        font-size: 14px;
                    }
                </style>
            </head>
            <body>
                <h2>OP3 YOLO / OpenVINO Detector</h2>
                <div class="hint">Yellow target = bar_target, white line = white_line</div>
                <br>
                <img src="/stream.mjpg">
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    with latest_lock:
                        frame = latest_jpeg

                    if frame is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")

                    time.sleep(0.05)
            except Exception:
                return

        self.send_response(404)
        self.end_headers()


def start_web_stream():
    server = HTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    t = threading.Thread(target=server.serve_forever)
    t.daemon = True
    t.start()
    print("Web stream started.")
    print("OP3 local browser: http://127.0.0.1:{}/".format(STREAM_PORT))


# ============================================================
# 影像前處理 / 後處理
# ============================================================

def letterbox(img, new_size=320):
    h, w = img.shape[:2]
    scale = min(float(new_size) / w, float(new_size) / h)

    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)

    pad_x = (new_size - nw) // 2
    pad_y = (new_size - nh) // 2

    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    return canvas, scale, pad_x, pad_y


def preprocess(frame):
    img, scale, pad_x, pad_y = letterbox(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, scale, pad_x, pad_y


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return float(inter) / float(union)


def nms(dets, iou_thres=0.45):
    if not dets:
        return []

    dets = sorted(dets, key=lambda x: x["conf"], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)

        remain = []
        for d in dets:
            if d["class_id"] != best["class_id"]:
                remain.append(d)
                continue

            iou = box_iou(
                [best["x1"], best["y1"], best["x2"], best["y2"]],
                [d["x1"], d["y1"], d["x2"], d["y2"]]
            )

            if iou < iou_thres:
                remain.append(d)

        dets = remain

    return keep






def normalize_angle_deg(a):
    """
    把角度正規化到 -45 ~ +45。
    目標是讓水平槓鈴接近 0 度。
    """
    while a > 45.0:
        a -= 90.0
    while a < -45.0:
        a += 90.0
    return a


def get_long_axis_angle_deg(det):
    """
    YOLO OBB 的 raw angle 不一定代表槓鈴長邊角度。
    這裡用 w/h 判斷長邊方向：
    - 如果 w >= h，raw angle 就比較接近長邊
    - 如果 h > w，長邊方向要 +90 度
    最後再折回 -45 ~ +45。
    """
    angle_deg = float(det["angle"]) * 180.0 / np.pi

    w = float(det["w"])
    h = float(det["h"])

    if h > w:
        angle_deg += 90.0

    angle_deg = normalize_angle_deg(angle_deg)

    return angle_deg


def postprocess(output, scale, pad_x, pad_y, frame_w, frame_h):
    pred = np.array(output)
    pred = np.squeeze(pred)

    # OpenVINO YOLO OBB output: [7, 2100] or [2100, 7]
    if pred.ndim != 2:
        return []

    if pred.shape[0] == 7:
        pred = pred.T

    detections = []

    for row in pred:
        if len(row) < 7:
            continue

        cx, cy, w, h = row[0], row[1], row[2], row[3]
        class_scores = row[4:6]
        angle = row[6]

        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])

        if conf < CONF_THRES:
            continue

        # 從 letterbox 座標轉回原圖座標
        cx0 = (float(cx) - pad_x) / scale
        cy0 = (float(cy) - pad_y) / scale
        w0 = float(w) / scale
        h0 = float(h) / scale

        x1 = int(cx0 - w0 / 2)
        y1 = int(cy0 - h0 / 2)
        x2 = int(cx0 + w0 / 2)
        y2 = int(cy0 + h0 / 2)

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "conf": conf,
            "cx": float(cx0),
            "cy": float(cy0),
            "w": float(w0),
            "h": float(h0),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "angle": float(angle),  # raw angle, usually rad
        })

    detections = nms(detections, IOU_THRES)

    return detections


# ============================================================
# 畫框與輸出訊息
# ============================================================

def draw(frame, detections, fps=0.0):
    for det in detections:
        cls = det["class_id"]
        name = det["class_name"]
        conf = det["conf"]

        cx = int(det["cx"])
        cy = int(det["cy"])
        w = max(1, float(det["w"]))
        h = max(1, float(det["h"]))

        angle_deg = get_long_axis_angle_deg(det)

        if cls == 0:
            color = (0, 255, 255)   # yellow
        else:
            color = (255, 255, 255) # white

        # OBB rotated box
        rect = ((float(cx), float(cy)), (w, h), angle_deg)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, color, 2)

        # center point
        cv2.circle(frame, (cx, cy), 5, color, -1)

        label = "{} {:.2f} cx={} cy={} ang={:.1f}".format(
            name, conf, cx, cy, angle_deg
        )

        text_y = max(20, cy - 15)
        cv2.putText(
            frame,
            label,
            (max(0, cx - 120), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.putText(
        frame,
        "FPS: {:.1f}".format(fps),
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


def r1(x):
    return round(float(x), 1)

def r2(x):
    return round(float(x), 2)


def make_message(detections, frame_w=640, frame_h=480):
    msg = {
        "t": r2(time.time()),
        "fw": int(frame_w),
        "fh": int(frame_h),
        "bar": {
            "found": False,
            "cx": 0,
            "cy": 0,
            "angle": 0.0,
            "conf": 0.0,
        },
        "lines": [],
    }

    # bar_target：取信心最高的一個
    bars = [d for d in detections if d["class_id"] == 0]
    if bars:
        b = sorted(bars, key=lambda x: x["conf"], reverse=True)[0]

        raw_angle = float(b["angle"]) * 180.0 / np.pi
        send_angle = get_long_axis_angle_deg(b)

        print("[BAR_ANGLE] raw={:.1f} send={:.1f} w={:.1f} h={:.1f}".format(
            raw_angle,
            send_angle,
            float(b["w"]),
            float(b["h"])
        ))

        msg["bar"] = {
            "found": True,
            "cx": int(b["cx"]),
            "cy": int(b["cy"]),
            "angle": r1(send_angle),
            "conf": r2(b["conf"]),
        }

    # white_line：可多條，按 cy 由大到小，近 → 遠
    lines = [d for d in detections if d["class_id"] == 1]
    lines = sorted(lines, key=lambda x: x["cy"], reverse=True)

    for l in lines:
        msg["lines"].append({
            "cx": int(l["cx"]),
            "cy": int(l["cy"]),
            "angle": r1(get_long_axis_angle_deg(l)),
            "conf": r2(l["conf"]),
        })

    return msg


# ============================================================
# main
# ============================================================

def main():
    global latest_jpeg

    model_path = Path(MODEL_PATH)
    print("model exists:", model_path.exists())

    core = Core()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, "CPU")

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    print("OpenVINO model loaded")
    print("input shape:", input_layer.shape)
    print("output shape:", output_layer.shape)

    if SHOW_WEB_STREAM:
        start_web_stream()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(CAM_ID)
    print("camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("ERROR: camera not opened")
        return

    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: failed to read frame")
                time.sleep(0.05)
                continue

            frame_h, frame_w = frame.shape[:2]

            blob, scale, pad_x, pad_y = preprocess(frame)

            result = compiled([blob])[output_layer]
            detections = postprocess(result, scale, pad_x, pad_y, frame_w, frame_h)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw(frame, detections, fps)

            msg = make_message(detections, frame_w, frame_h)
            msg_text = json.dumps(msg, ensure_ascii=False)

            # 終端機顯示數值
            bar = msg["bar"]
            if bar["found"]:
                print("[DET] cx={} cy={} ang={:.1f} conf={:.2f} lines={}".format(
                    bar["cx"],
                    bar["cy"],
                    bar["angle"],
                    bar["conf"],
                    len(msg["lines"])
                ))
            else:
                print("[DET] bar LOST lines={}".format(len(msg["lines"])))

            # UDP 傳給 op3_weightlifting_v4.py
            sock.sendto(msg_text.encode("utf-8"), (UDP_IP, UDP_PORT))

            # 網頁串流畫面
            if SHOW_WEB_STREAM:
                ok, jpg = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                )
                if ok:
                    with latest_lock:
                        latest_jpeg = jpg.tobytes()

    except KeyboardInterrupt:
        print("detector stopped by Ctrl+C")

    finally:
        cap.release()
        sock.close()


if __name__ == "__main__":
    main()