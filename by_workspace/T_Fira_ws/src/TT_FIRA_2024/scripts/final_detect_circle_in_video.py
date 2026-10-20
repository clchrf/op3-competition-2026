import cv2
import numpy as np

cap = cv2.VideoCapture(0)

delay_between_frames = 10  

while True:
    
    ret, img = cap.read()

    if not ret:
        break  

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_blurred = cv2.blur(gray, (3, 3))

    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT,
                                        1, 10000, param1=50, param2=30,
                                        minRadius=25, maxRadius=50)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))

        for i in detected_circles[0, :]:
            a, b, r = i[0], i[1], i[2]
            print(a, b)
            
            cv2.circle(img, (a, b), r, (0, 255, 0), 3)
            cv2.circle(img, (a, b), 1, (0, 100, 100), 3)


    cv2.imshow("show image", img)


    key = cv2.waitKey(delay_between_frames)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()