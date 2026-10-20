import cv2
import numpy as np
import argparse
import os
import time
import threading
import socket
import json

from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    from openvino import Core
except Exception:
    from openvino.runtime import Core


latest_jpeg = None
latest_text = "waiting..."
frame_lock = threading.Lock()


def letterbox(img, new_shape=320):
    h, w = img.shape[:2]
    r = min(float(new_shape) / h, float(new_shape) / w)

    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img, (nw, nh))

    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)

    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2

    canvas[dh:dh + nh, dw:dw + nw] = resized

    return canvas, r, dw, dh


def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]

    return keep


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            html = """
            <html>
            <head>
                <title>OP3 Basketball YOLO</title>
            </head>
            <body style="background:#111;color:white;text-align:center;font-family:Arial;">
                <h2>OP3 Basketball YOLO Stream</h2>
                <img src="/video" style="max-width:95%; border:2px solid white;">
                <p>ball / hoop detection stream</p>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/video":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while True:
                with frame_lock:
                    frame = latest_jpeg

                if frame is None:
                    time.sleep(0.05)
                    continue

                try:
                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
                except Exception:
                    break
        else:
            self.send_response(404)
            self.end_headers()


def detector_loop(args):
    global latest_jpeg, latest_text

    model_dir = args.model
    xml_files = [f for f in os.listdir(model_dir) if f.endswith(".xml")]

    if not xml_files:
        print("找不到 .xml 模型檔，請檢查:", model_dir)
        return

    model_path = os.path.join(model_dir, xml_files[0])
    print("Using model:", model_path)

    # Open camera first, OP3 camera is more stable this way
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)

    if not cap.isOpened():
        print("相機開啟失敗，cam =", args.cam)
        return

    ret, test_frame = cap.read()
    if not ret:
        print("相機已開啟，但讀不到畫面")
        cap.release()
        return

    print("Camera OK, frame shape =", test_frame.shape)

    core = Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, "CPU")

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    # UDP send to Basketball_yolo.py / yolo_receiver.py
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_addr = ("127.0.0.1", args.udp_port)
    print("UDP send to 127.0.0.1:%d" % args.udp_port)

    names = {
        0: "ball",
        1: "hoop"
    }

    frame_count = 0

    # FPS counter
    fps = 0.0
    fps_count = 0
    fps_t0 = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("讀不到相機畫面")
            time.sleep(0.1)
            continue

        original = frame.copy()
        ih, iw = original.shape[:2]

        img, r, dw, dh = letterbox(frame, args.imgsz)

        # YOLO usually uses RGB input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inp = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        pred = compiled([inp])[output_layer]
        pred = np.squeeze(pred)

        if pred.shape[0] == 6:
            pred = pred.T

        boxes = []
        scores = []
        class_ids = []

        for p in pred:
            x, y, w, h = p[0], p[1], p[2], p[3]
            cls_scores = p[4:]

            cls_id = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_id])

            if score < args.conf:
                continue

            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r

            x1 = max(0, min(iw - 1, x1))
            y1 = max(0, min(ih - 1, y1))
            x2 = max(0, min(iw - 1, x2))
            y2 = max(0, min(ih - 1, y2))

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            class_ids.append(cls_id)

        keep = nms(boxes, scores)

        ball_found = False
        hoop_found = False

        best_ball = None
        best_hoop = None
        best_ball_score = -1.0
        best_hoop_score = -1.0

        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            cls_id = class_ids[i]
            label = names.get(cls_id, str(cls_id))

            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            score_f = float(score)

            if cls_id == 0 and score_f > best_ball_score:
                best_ball = [cx, cy, bw, bh, score_f]
                best_ball_score = score_f

            if cls_id == 1 and score_f > best_hoop_score:
                best_hoop = [cx, cy, bw, bh, score_f]
                best_hoop_score = score_f

            if cls_id == 0:
                ball_found = True
            elif cls_id == 1:
                hoop_found = True

            cv2.rectangle(original, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                original,
                "%s %.2f" % (label, score),
                (int(x1), max(20, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        msg = {
            "t": time.time(),
            "ball": best_ball,
            "hoop": best_hoop
        }

        udp_sock.sendto(json.dumps(msg).encode("utf-8"), udp_addr)

        # update FPS about once per second
        fps_count += 1
        now = time.time()
        dt = now - fps_t0

        if dt >= 1.0:
            fps = fps_count / dt
            fps_count = 0
            fps_t0 = now

        status = "ball=%s hoop=%s detections=%d FPS=%.1f" % (
            "Y" if ball_found else "N",
            "Y" if hoop_found else "N",
            len(keep),
            fps
        )

        def fmt_det_line(name, box):
            if box is None:
                return "[%s] None" % name

            return "[%s] cx=%d cy=%d w=%.1f h=%.1f conf=%.2f" % (
                name,
                int(box[0]),
                int(box[1]),
                box[2],
                box[3],
                box[4]
            )


        det_text = "[DET] ball=%s hoop=%s detections=%d FPS=%.1f" % (
            "Y" if ball_found else "N",
            "Y" if hoop_found else "N",
            len(keep),
            fps
        )

        ball_text = fmt_det_line("BALL", best_ball)
        hoop_text = fmt_det_line("HOOP", best_hoop)

        cv2.putText(
            original,
            det_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.putText(
            original,
            ball_text,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            original,
            hoop_text,
            (10, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        ok, jpg = cv2.imencode(".jpg", original)

        if ok:
            with frame_lock:
                latest_jpeg = jpg.tobytes()
                latest_text = status

        frame_count += 1
        if frame_count % args.print_every == 0:
            print(det_text)
            print(ball_text)
            print(hoop_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_openvino_model")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--print-every", type=int, default=5)
    args = parser.parse_args()

    t = threading.Thread(target=detector_loop, args=(args,))
    t.daemon = True
    t.start()

    print("Open browser:")
    print("http://OP3_IP:%d" % args.port)

    server = HTTPServer((args.host, args.port), StreamHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
