# Basketball YOLO（球 / 籃框偵測）— 電腦端訓練流程

沒有機器人、也沒有 GPU 也能完成這一整套。機器人到位後只需要做最後的匯出對版本、
FPS 實測、相機座標確認和 socket 整合。

## 資料夾結構

```
basketball_yolo/
├── data.yaml          # 類別與資料路徑設定（ball / hoop）
├── train.py           # 訓練（Colab GPU 或本機 CPU）
├── webcam_test.py     # 本機 webcam 測試（CPU 即可，現在就能跑）
├── export.py          # 匯出 ONNX / OpenVINO（部署前再做）
└── dataset/
    ├── images/train/  # 訓練影像 (.jpg/.png)
    ├── images/val/    # 驗證影像
    ├── labels/train/  # 對應的 YOLO 標註 (.txt)
    └── labels/val/    # 對應的 YOLO 標註
```

YOLO 標註格式：每張圖一個同名 `.txt`，每行一個物件
`<class_id> <x_center> <y_center> <width> <height>`，座標都是 0~1 正規化值。
class_id：球=0，籃框=1。

## 0. 安裝（在你的電腦上，不是機器人）

需要 Python 3.8 以上：

```bash
pip install ultralytics
```

## 1. 現在就能做：先驗證球的偵測（零訓練）

橘色籃球在 COCO 預訓練權重裡會被當成 "sports ball"，所以不用先訓練就能測通管線：

```bash
python webcam_test.py
```

對著球，框會框出來並印出中心 `(cx, cy)`。這個 `(cx, cy)` 就是之後要送進機器人
對準邏輯的座標，介面跟現在程式裡算的 `bcx, bcy` 一致。

## 2. 準備資料（主要的工）

球和籃框各拍/收集數百到一兩千張，涵蓋不同距離、角度、光線。三條路：

- **球**：可先靠 stock 權重，重點放在自己拍籃框。
- **籃框**：到 Roboflow Universe 搜 "basketball hoop" 找現成資料集起步。
- **貼近現場**：用一支跟機器人相近的 USB webcam、同樣設 640×480 自己拍，
  之後拿到機器人再補一批「實際相機 + 實際場地」的照片做 fine-tune，準確度最高。

標註工具任選：Roboflow（網頁，可標註+增強+直接匯出 YOLO 格式，最省事）、
CVAT、Label Studio、LabelImg。把匯出的 images / labels 放進上面的資料夾。

## 3. 訓練

**Colab 免費 GPU（建議）**：把 `basketball_yolo` 上傳到 Google Drive 或打包上傳，
新建 notebook，`!pip install ultralytics`，掛上資料夾後：

```bash
python train.py --device 0 --imgsz 640 --batch 16 --epochs 100
```

**本機 CPU（沒 GPU，較慢）**：

```bash
python train.py --device cpu --imgsz 416 --batch 8 --epochs 80
```

結果在 `runs/basketball/weights/best.pt`。把它下載回本機保管。

## 4. 驗證

把 `webcam_test.py` 裡的 `WEIGHTS` 改成 `runs/basketball/weights/best.pt`、
`ONLY_CLASSES` 改成 `None`，再跑一次，確認 ball / hoop 都框得準。

## 5. 匯出（等機器人到位、確認執行階段版本後再做）

```bash
# 先產 ONNX 驗證流程
python export.py --weights runs/basketball/weights/best.pt --format onnx --imgsz 416
# 之後對 Intel CPU 匯 OpenVINO（可加 --int8 加速）
python export.py --weights runs/basketball/weights/best.pt --format openvino --imgsz 416 --int8
```

注意：Ubuntu 16.04 上裝得到的 OpenVINO / onnxruntime 較舊，匯出的 opset / IR 版本
要相容，務必在機器人上實際載入驗證。這一步留到知道機器人端版本再定。

## 機器人到位後才做的事（先記著）

1. 在那台 NUC 上裝 Miniconda 開 Python 3 環境，跑偵測程序（接管相機 + YOLO 推論）。
2. 偵測程序用 UDP / ZeroMQ 把最新一筆 `(cx, cy, w, h, conf)` 丟給 Python 2 控制節點。
3. 控制節點把現在 `find_ball` / `find_board` 裡 HSV + findContours 那段，換成
   「從 socket 讀框中心」，下面的 EMA + 對準門檻原封不動。
4. NUC 上實測 FPS，必要時用 OpenVINO INT8 加速；確認相機解析度好換算座標。

模型選型先用 YOLO11n；之後想壓 CPU 速度，換成 YOLO26n 只是把權重字串改一行，
資料不用重來。
