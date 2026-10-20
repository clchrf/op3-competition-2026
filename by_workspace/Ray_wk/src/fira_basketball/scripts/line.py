import numpy as np
import cv2

def detect_line(frame,hsv_l,hsv_h):
    height,width=frame.shape[:2]
    point_x=width//2
    point_y=height-1-130

    color = frame[point_y,point_x]
    hsv_color=cv2.cvtColor(np.uint8([[color]]),cv2.COLOR_BGR2HSV)

    detected = cv2.inRange(hsv_color,hsv_l,hsv_h).any()
    circle_color = (0,0,0)
    cv2.circle(frame,(point_x,point_y),5,circle_color,-1)
    # return frame,detected
    return frame,detected


cap =cv2.VideoCapture(0)
lower_r=np.array([0, 59, 137])
upper_r=np.array([ 180, 255, 181])

while True:
    ret,frame = cap.read()
    if not ret:
        print("?")
        break

    frame,detect = detect_line(frame,lower_r,upper_r)
    cv2.imshow("frame",frame)
    if detect:
        print("YYYYYY")
    
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cap.release()
cv2.destroyALLWindows()