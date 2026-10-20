#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
匯出腳本（部署到機器人前用）。

重要：最終匯出要對準機器人上「實際裝到」的推論執行階段版本。
Ubuntu 16.04 上裝得到的 OpenVINO / onnxruntime 比較舊，opset / IR 版本要相容，
所以這一步建議等確認機器人端版本後再做，並在機器人上實際載入驗證能不能讀。

先在電腦上產 ONNX 驗證流程：
    python export.py --weights runs/basketball/weights/best.pt --format onnx --imgsz 416

之後針對機器人（Intel CPU）匯 OpenVINO，可加 --int8 量化加速：
    python export.py --weights runs/basketball/weights/best.pt --format openvino --imgsz 416 --int8
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/basketball/weights/best.pt")
    ap.add_argument("--format", default="openvino", choices=["onnx", "openvino"])
    ap.add_argument("--imgsz", type=int, default=416,
                    help="要跟機器人端推論時用的尺寸一致")
    ap.add_argument("--opset", type=int, default=12,
                    help="ONNX opset；老執行階段相容性問題時調低（如 11/12）")
    ap.add_argument("--int8", action="store_true",
                    help="INT8 量化，CPU 上更快（需要少量校正影像，Ultralytics 會用 data.yaml 的 val 集）")
    ap.add_argument("--data", default="data.yaml")
    args = ap.parse_args()

    model = YOLO(args.weights)
    kwargs = dict(format=args.format, imgsz=args.imgsz)
    if args.format == "onnx":
        kwargs["opset"] = args.opset
    if args.int8:
        kwargs["int8"] = True
        kwargs["data"] = args.data
    path = model.export(**kwargs)
    print("匯出完成:", path)


if __name__ == "__main__":
    main()
