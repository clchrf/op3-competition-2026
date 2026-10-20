#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#start the camera and detect
#env -u PYTHONPATH /home/robotis/anaconda3/envs/yolo_ov/bin/python /home/robotis/joshua_ws/src/marathon/scripts/arrow_detector.py

#OP3 local browser: http://127.0.0.1:8080/



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
# Settings
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = "/home/robotis/Ray_wk/src/fira_basketball/scripts/marathon/models/arrow_openvino_model/best.xml"
CAM_ID = 0

IMG_SIZE = 320
CONF_THRES = 0.40
DECISION_CONF_THRES = 0.60
IOU_THRES = 0.45

UDP_IP = "127.0.0.1"
UDP_PORT = 5006

SHOW_WEB_STREAM = True
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8080

CLASS_NAMES = {
    0: "forward_arrow",
    1: "left_arrow",
    2: "right_arrow",
}

# ============================================================
# AprilTag settings
# Official marathon markers detected with OpenCV AprilTag 36h11:
#   forward = id 1, right = id 2, left = id 3
# ============================================================

USE_APRILTAG = True
TAG_DICT_NAME = "DICT_APRILTAG_36h11"
TAG_ID_TO_NAME = {
    1: "right_arrow",
    2: "forward_arrow",
    3: "left_arrow",
}

latest_jpeg = None
latest_lock = threading.Lock()


# ============================================================
# Web stream
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
                <title>OP3 Marathon Arrow Detector</title>
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
                <h2>OP3 Marathon Arrow Detector</h2>
                <div class="hint">Blue = forward, green = left, red = right</div>
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
# Preprocess / postprocess
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


def postprocess(output, scale, pad_x, pad_y, frame_w, frame_h):
    pred = np.array(output)
    pred = np.squeeze(pred)

    # YOLO11 Detect output for 3 classes: [7, 2100] or [2100, 7]
    # 7 = cx, cy, w, h, forward_score, left_score, right_score
    if pred.ndim != 2:
        print("WARNING: unexpected output ndim:", pred.ndim)
        return []

    expected_dim = 4 + len(CLASS_NAMES)

    if pred.shape[0] == expected_dim:
        pred = pred.T
    elif pred.shape[1] != expected_dim:
        print("WARNING: unexpected output shape:", pred.shape)
        return []

    detections = []

    for row in pred:
        if len(row) < expected_dim:
            continue

        cx, cy, w, h = row[0], row[1], row[2], row[3]
        class_scores = row[4:4 + len(CLASS_NAMES)]

        class_id = int(np.argmax(class_scores))
        conf = float(class_scores[class_id])

        if conf < CONF_THRES:
            continue

        cx0 = (float(cx) - pad_x) / scale
        cy0 = (float(cy) - pad_y) / scale
        w0 = float(w) / scale
        h0 = float(h) / scale

        x1 = int(round(cx0 - w0 / 2.0))
        y1 = int(round(cy0 - h0 / 2.0))
        x2 = int(round(cx0 + w0 / 2.0))
        y2 = int(round(cy0 + h0 / 2.0))

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, str(class_id)),
            "conf": conf,
            "cx": float(cx0),
            "cy": float(cy0),
            "w": float(w0),
            "h": float(h0),
            "area": float(w0 * h0),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        })

    return nms(detections, IOU_THRES)


# ============================================================
# Drawing and UDP message
# ============================================================

def get_color(class_id):
    if class_id == 0:
        return (255, 120, 0)
    if class_id == 1:
        return (0, 255, 0)
    return (0, 0, 255)


def select_primary_arrow(detections):
    if not detections:
        return None

    # Prefer a large nearby arrow while still considering confidence.
    return max(detections, key=lambda d: float(d["area"]) * float(d["conf"]))


def init_tag_detector():
    if not USE_APRILTAG:
        print("AprilTag disabled")
        return None

    if not hasattr(cv2, "aruco"):
        print("ERROR: cv2.aruco not found")
        return None

    if not hasattr(cv2.aruco, TAG_DICT_NAME):
        print("ERROR: {} not found in cv2.aruco".format(TAG_DICT_NAME))
        print([x for x in dir(cv2.aruco) if "APRILTAG" in x])
        return None

    tag_dict_id = getattr(cv2.aruco, TAG_DICT_NAME)
    dictionary = cv2.aruco.getPredefinedDictionary(tag_dict_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
    else:
        params = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        print("AprilTag detector loaded with ArucoDetector, dict={}".format(TAG_DICT_NAME))
        return ("new", detector)

    print("AprilTag detector loaded with old detectMarkers API, dict={}".format(TAG_DICT_NAME))
    return ("old", dictionary, params)


def detect_tag(frame, tag_detector):
    if tag_detector is None:
        return None, []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        if tag_detector[0] == "new":
            corners, ids, rejected = tag_detector[1].detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, tag_detector[1], parameters=tag_detector[2])
    except Exception as e:
        print("AprilTag detect error:", e)
        return None, []

    if ids is None or len(ids) == 0:
        return None, []

    best_i = -1
    best_area = -1.0

    for i, c in enumerate(corners):
        pts = np.array(c[0], dtype=np.float32)
        area = float(cv2.contourArea(pts))
        if area > best_area:
            best_area = area
            best_i = i

    if best_i < 0:
        return None, []

    pts = np.array(corners[best_i][0], dtype=np.float32)
    tag_id = int(ids[best_i][0])
    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))
    name = TAG_ID_TO_NAME.get(tag_id, "unknown")

    info = {
        "found": True,
        "id": tag_id,
        "name": name,
        "cx": cx,
        "cy": cy,
        "area": int(best_area),
    }

    return info, corners


def draw_tags(frame, tag_info, corners):
    for c in corners:
        pts = np.array(c[0], dtype=np.int32)

        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            cv2.line(frame, p1, p2, (0, 255, 255), 2)

    if tag_info is not None:
        cx = int(tag_info["cx"])
        cy = int(tag_info["cy"])
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
        cv2.putText(frame, "TAG id={} {}".format(tag_info["id"], tag_info["name"]),
                    (max(0, cx - 60), max(20, cy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def draw(frame, detections, primary=None, fps=0.0):
    for det in detections:
        color = get_color(det["class_id"])
        thickness = 3 if det is primary else 2

        cv2.rectangle(frame, (det["x1"], det["y1"]), (det["x2"], det["y2"]), color, thickness)
        cv2.circle(frame, (int(det["cx"]), int(det["cy"])), 4, color, -1)

        label = "{} {:.2f} cx={} cy={}".format(det["class_name"], det["conf"], int(det["cx"]), int(det["cy"]))
        text_y = max(20, det["y1"] - 8)
        cv2.putText(frame, label, (det["x1"], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if primary is not None:
        status = "PRIMARY: {} {:.2f}".format(primary["class_name"], primary["conf"])
    else:
        status = "PRIMARY: NONE"

    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(frame, "FPS: {:.1f}".format(fps), (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def r2(x):
    return round(float(x), 2)


def make_message(detections, tag_info=None, frame_w=640, frame_h=480):
    primary = select_primary_arrow(detections)

    msg = {
        "t": r2(time.time()),
        "fw": int(frame_w),
        "fh": int(frame_h),
        "arrow": {
            "found": False,
            "accepted": False,
            "class_id": -1,
            "name": "none",
            "conf": 0.0,
            "cx": 0,
            "cy": 0,
            "w": 0,
            "h": 0,
        },
        "detections": [],
        "tag": {
            "found": False,
            "id": -1,
            "name": "none",
            "cx": 0,
            "cy": 0,
            "area": 0,
        },
    }

    for det in detections:
        msg["detections"].append({
            "class_id": int(det["class_id"]),
            "name": det["class_name"],
            "conf": r2(det["conf"]),
            "cx": int(det["cx"]),
            "cy": int(det["cy"]),
            "w": int(det["w"]),
            "h": int(det["h"]),
        })

    if primary is not None:
        msg["arrow"] = {
            "found": True,
            "accepted": bool(primary["conf"] >= DECISION_CONF_THRES),
            "class_id": int(primary["class_id"]),
            "name": primary["class_name"],
            "conf": r2(primary["conf"]),
            "cx": int(primary["cx"]),
            "cy": int(primary["cy"]),
            "w": int(primary["w"]),
            "h": int(primary["h"]),
        }

    if tag_info is not None:
        msg["tag"] = {
            "found": True,
            "id": int(tag_info["id"]),
            "name": tag_info["name"],
            "cx": int(tag_info["cx"]),
            "cy": int(tag_info["cy"]),
            "area": int(tag_info["area"]),
        }

    return msg, primary


# ============================================================
# Main
# ============================================================

def main():
    global latest_jpeg

    model_path = Path(MODEL_PATH)
    print("model path:", model_path)
    print("model exists:", model_path.exists())

    if not model_path.exists():
        print("ERROR: model not found")
        return

    core = Core()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, "CPU")

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    print("OpenVINO model loaded")
    print("input shape:", input_layer.shape)
    print("output shape:", output_layer.shape)
    print("classes:", CLASS_NAMES)
    print("UDP target: {}:{}".format(UDP_IP, UDP_PORT))

    if SHOW_WEB_STREAM:
        start_web_stream()

    tag_detector = init_tag_detector()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    cap = cv2.VideoCapture(CAM_ID)
    print("camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("ERROR: camera not opened")
        sock.close()
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
            tag_info, tag_corners = detect_tag(frame, tag_detector)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            msg, primary = make_message(detections, tag_info, frame_w, frame_h)
            draw(frame, detections, primary, fps)
            draw_tags(frame, tag_info, tag_corners)

            msg_text = json.dumps(msg, ensure_ascii=False)

            arrow = msg["arrow"]
            if arrow["found"]:
                print("[ARROW] name={} cx={} cy={} conf={:.2f} accepted={} total={}".format(
                    arrow["name"],
                    arrow["cx"],
                    arrow["cy"],
                    arrow["conf"],
                    arrow["accepted"],
                    len(msg["detections"])
                ))
            else:
                print("[ARROW] LOST total=0")

            tag = msg["tag"]
            if tag["found"]:
                print("[TAG] id={} name={} cx={} cy={} area={}".format(
                    tag["id"], tag["name"], tag["cx"], tag["cy"], tag["area"]
                ))
            else:
                print("[TAG] LOST")

            sock.sendto(msg_text.encode("utf-8"), (UDP_IP, UDP_PORT))

            if SHOW_WEB_STREAM:
                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
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
