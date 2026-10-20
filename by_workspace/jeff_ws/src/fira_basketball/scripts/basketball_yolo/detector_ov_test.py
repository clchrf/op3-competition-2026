import cv2
import numpy as np
import argparse
import os

try:
    from openvino import Core
except Exception:
    from openvino.runtime import Core


def letterbox(img, new_shape=320):
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_openvino_model")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    model_dir = args.model
    xml_files = [f for f in os.listdir(model_dir) if f.endswith(".xml")]
    if not xml_files:
        print("找不到 .xml 模型檔，請檢查:", model_dir)
        return

    model_path = os.path.join(model_dir, xml_files[0])
    print("Using model:", model_path)

    core = Core()
    model = core.read_model(model_path)
    compiled = core.compile_model(model, "CPU")

    input_layer = compiled.input(0)
    output_layer = compiled.output(0)

    cap = cv2.VideoCapture(args.cam)

    if not cap.isOpened():
        print("相機開啟失敗，cam =", args.cam)
        return

    names = {0: "ball", 1: "hoop"}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("讀不到相機畫面")
            break

        original = frame.copy()
        ih, iw = original.shape[:2]

        img, r, dw, dh = letterbox(frame, args.imgsz)
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

        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            score = scores[i]
            cls_id = class_ids[i]
            label = names.get(cls_id, str(cls_id))

            if cls_id == 0:
                ball_found = True
            elif cls_id == 1:
                hoop_found = True

            if args.show:
                cv2.rectangle(original, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(original, "%s %.2f" % (label, score), (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print("ball=%s hoop=%s detections=%d" % (
            "Y" if ball_found else "N",
            "Y" if hoop_found else "N",
            len(keep)
        ))

        if args.show:
            cv2.imshow("detector_ov_test", original)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
