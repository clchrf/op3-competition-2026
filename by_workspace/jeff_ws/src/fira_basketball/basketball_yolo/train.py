#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
訓練腳本（Colab GPU 或本機 CPU 都能用）。

從 COCO 預訓練的 yolo11n.pt 做遷移學習，小資料集收斂很快。

Colab（免費 GPU）：
    python train.py --device 0 --imgsz 640 --batch 16 --epochs 100

本機 CPU（沒 GPU，較慢，把圖縮小、batch 調小）：
    python train.py --device cpu --imgsz 416 --batch 8 --epochs 80

訓練結果在 runs/basketball/weights/best.pt
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt",
                    help="起始權重；先用 yolo11n.pt 做遷移學習")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="訓練影像尺寸；CPU 訓練建議 416")
    ap.add_argument("--batch", type=int, default=16,
                    help="CPU 訓練建議 8 或更小")
    ap.add_argument("--device", default="0",
                    help="'0' 用 GPU，'cpu' 用 CPU")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=30,            # 30 個 epoch 沒進步就提早停
        project="runs",
        name="basketball",
        exist_ok=True,
    )
    # 訓練完順手在驗證集上評估一次
    metrics = model.val()
    print("mAP50-95:", metrics.box.map)
    print("mAP50   :", metrics.box.map50)


if __name__ == "__main__":
    main()
