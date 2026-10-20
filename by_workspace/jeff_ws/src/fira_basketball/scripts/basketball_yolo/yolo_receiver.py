#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
給 Python 2.7 控制節點用的 YOLO 偵測接收器。
非阻塞地從 UDP 讀 detector.py 發來的最新 ball / hoop 偵測。

用法：
    from yolo_receiver import YoloReceiver
    yolo = YoloReceiver(port=5005)
    ...
    det = yolo.get_latest()
    # det = {"ball": [cx, cy, w, h, conf] 或 None,
    #        "hoop": [cx, cy, w, h, conf] 或 None,
    #        "age":  距離最後一筆偵測的秒數}
"""
import socket
import json
import time


class YoloReceiver(object):
    def __init__(self, ip="127.0.0.1", port=5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)
        self.last = {"t": 0.0, "ball": None, "hoop": None}

    def _drain(self):
        # 把佇列裡的封包全讀掉，只保留最新一筆（避免吃到舊資料）
        while True:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.error:
                break
            try:
                self.last = json.loads(data.decode("utf-8"))
            except Exception:
                pass

    def get_latest(self):
        self._drain()
        age = time.time() - self.last.get("t", 0.0)
        return {"ball": self.last.get("ball"),
                "hoop": self.last.get("hoop"),
                "age": age}
