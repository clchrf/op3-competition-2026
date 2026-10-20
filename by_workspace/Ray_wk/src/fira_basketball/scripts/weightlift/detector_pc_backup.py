cat > detector.py <<'PY'
import cv2
import time
import numpy as np
from pathlib import Path
from openvino import Core

MODEL_PATH = "best_openvino_model/best.xml"
IMG_SIZE = 320
CONF_THRES = 0.25
CAM_ID = 0

# classes: 依照你的訓練設定
CLASS_NAMES = {
    0: "bar_target",
    1: "white_line",
}

def letterbox(img, new_size=320):
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((new_size, new_size, 3), 114, dtype=np.uint8)

    dx = (new_size - nw) // 2
    dy = (new_size - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized

    return canvas, scale, dx, dy

def preprocess(frame):
    img, scale, dx, dy = letterbox(frame, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img, scale, dx, dy

def draw_detection(frame, det, scale, dx, dy):
    # det: [cx, cy, w, h, class_id, conf, angle]
    cx, cy, bw, bh, cls_id, conf, angle = det

    # 還原到原圖座標
    cx = (cx - dx) / scale
    cy = (cy - dy) / scale
    bw = bw / scale
    bh = bh / scale

    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    x2 = int(cx + bw / 2)
    y2 = int(cy + bh / 2)

    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    name = CLASS_NAMES.get(int(cls_id), str(int(cls_id)))
    label = "{} {:.2f}".format(name, conf)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

def postprocess(output):
    """
    OpenVINO YOLO OBB output shape: [1, 7, 2100]
    7 = x, y, w, h, class0_score, class1_score, angle
    """
    pred = output[0]          # [7, 2100]
    pred = pred.T             # [2100, 7]

    detections = []

    for row in pred:
        cx, cy, bw, bh = row[0], row[1], row[2], row[3]
        cls_scores = row[4:6]
        angle = row[6]

        cls_id = int(np.argmax(cls_scores))
        conf = float(cls_scores[cls_id])

        if conf < CONF_THRES:
            continue

        detections.append([cx, cy, bw, bh, cls_id, conf, angle])

    # 先簡化：每個 class 只取信心最高的一個
    best_by_class = {}
    for det in detections:
        cls_id = int(det[4])
        if cls_id not in best_by_class or det[5] > best_by_class[cls_id][5]:
            best_by_class[cls_id] = det

    return list(best_by_class.values())

def main():
    model_path = Path(MODEL_PATH)
    print("model path:", model_path.resolve())
    print("model exists:", model_path.exists())

    core = Core()
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, "CPU")

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    print("OpenVINO model loaded")
    print("input shape:", input_layer.shape)
    print("output shape:", output_layer.shape)

    cap = cv2.VideoCapture(CAM_ID)
    print("camera opened:", cap.isOpened())

    if not cap.isOpened():
        print("ERROR: camera open failed. Try CAM_ID = 1 or check /dev/video*")
        return

    prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("camera read failed")
            time.sleep(0.1)
            continue

        inp, scale, dx, dy = preprocess(frame)
        result = compiled([inp])[output_layer]

        detections = postprocess(result)

        for det in detections:
            draw_detection(frame, det, scale, dx, dy)

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now

        cv2.putText(frame, "FPS: {:.1f}".format(fps), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        print("detections:", detections)

        cv2.imshow("OpenVINO YOLO OBB", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
PY