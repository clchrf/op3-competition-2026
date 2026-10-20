import cv2
import time
import numpy as np
cap = cv2.VideoCapture(0)
while(True):

    ret, frame = cap.read()
    lower_hsv = np.array([121, 60, 143])
    upper_hsv = np.array([179, 255, 255])
    mask = cv2.inRange(frame, lower_hsv, upper_hsv)
    if ret:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Convert frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask with the given HSV range
        mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
        
        # Convert mask to a 3-channel image to put a colored dot
        colored_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Find contours in the mask
        # Note: cv2.findContours returns 3 values in older OpenCV versions used with Python 2.7
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        
        # Ensure at least one contour was found
        if contours:
            # Find the largest contour based on area
            largest_contour = max(contours, key=cv2.contourArea)
            
