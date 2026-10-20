import cv2
import numpy as np
import time
import math

def empty(v):
    pass

cap = cv2.VideoCapture(0)

start_time = time.time()
current_time = start_time

position_correct = False
detect_line = False

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    height = frame.shape[0]
    crop_height = int(height * 0.25)
    crop = 100
    frame = frame[:crop_height, crop:-crop]

    img_blur = cv2.GaussianBlur(frame, (3,3) , 5)
    hsv_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    # h_min = cv2.getTrackbarPos("hue min", "track_bar")
    # h_max = cv2.getTrackbarPos("hue max", "track_bar")
    # s_min = cv2.getTrackbarPos("sat min", "track_bar")
    # s_max = cv2.getTrackbarPos("sat max", "track_bar")
    # v_min = cv2.getTrackbarPos("val min", "track_bar")
    # v_max = cv2.getTrackbarPos("val max", "track_bar")
    h_min, h_max, s_min, s_max, v_min, v_max = 90, 120, 100, 255, 50, 255
    # print(h_min, h_max, s_min, s_max, v_min, v_max )
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])

    mask_img = cv2.inRange(hsv_img, lower, upper)
    
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_blurred = cv2.GaussianBlur(gray, (7, 7), 0)
   
    img_blurred = cv2.GaussianBlur(mask_img, (7, 7), 0)
    img_blurred = cv2.medianBlur(img_blurred, 5)
    img_blurred = cv2.medianBlur(img_blurred, 5)
   
    edges = cv2.Canny(img_blurred, 50, 150)
   
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=30, maxLineGap=20)
    
    detect_line = False

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = float(y2 - y1) / float(x2 - x1 + 1e-6)
            if abs(slope) < 0.3:
                detect_line = True
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
       
    cv2.imshow('Video', frame)
    cv2.imshow("mask",mask_img)
    cv2.imshow("mask",edges)
    cv2.waitKey(1)