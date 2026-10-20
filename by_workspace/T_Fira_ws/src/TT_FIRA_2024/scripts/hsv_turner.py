    #!/usr/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


## rosrun cv_camera cv_camera_node

class HSVTuner:
    def __init__(self):
        rospy.init_node("hsv_tuner", anonymous=True)

        self.bridge = CvBridge()
        self.frame = None
        self.image_topic = rospy.get_param("~image_topic", "/cv_camera/image_raw")
        self.save_path = rospy.get_param("~save_path", "/home/robotis/hsv_range.txt")

        cv2.namedWindow("HSV Control", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("HSV Control", 500, 400)
        cv2.resizeWindow("Original", 640, 480)
        cv2.resizeWindow("Mask", 640, 480)
        cv2.resizeWindow("Result", 640, 480)

        cv2.createTrackbar("H Min", "HSV Control", 0, 179, self.nothing)
        cv2.createTrackbar("H Max", "HSV Control", 179, 179, self.nothing)
        cv2.createTrackbar("S Min", "HSV Control", 0, 255, self.nothing)
        cv2.createTrackbar("S Max", "HSV Control", 255, 255, self.nothing)
        cv2.createTrackbar("V Min", "HSV Control", 0, 255, self.nothing)
        cv2.createTrackbar("V Max", "HSV Control", 255, 255, self.nothing)

        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo("HSV tuner started.")
        rospy.loginfo("Image topic: %s", self.image_topic)
        rospy.loginfo("P: print HSV range")
        rospy.loginfo("S: save HSV range")
        rospy.loginfo("R: reset trackbars")
        rospy.loginfo("Q or ESC: quit")

    def nothing(self, value):
        pass

    def image_callback(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as error:
            rospy.logerr(error)

    def get_hsv_range(self):
        h_min = cv2.getTrackbarPos("H Min", "HSV Control")
        h_max = cv2.getTrackbarPos("H Max", "HSV Control")
        s_min = cv2.getTrackbarPos("S Min", "HSV Control")
        s_max = cv2.getTrackbarPos("S Max", "HSV Control")
        v_min = cv2.getTrackbarPos("V Min", "HSV Control")
        v_max = cv2.getTrackbarPos("V Max", "HSV Control")

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        return lower, upper

    def print_hsv_range(self):
        lower, upper = self.get_hsv_range()

        print("")
        print("========== HSV RANGE ==========")
        print("HSV_LOWER = np.array([%d, %d, %d])" % (lower[0], lower[1], lower[2]))
        print("HSV_UPPER = np.array([%d, %d, %d])" % (upper[0], upper[1], upper[2]))
        print("cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)")
        print("================================")
        print("")

    def save_hsv_range(self):
        lower, upper = self.get_hsv_range()

        try:
            output = open(self.save_path, "w")
            output.write("HSV_LOWER = np.array([%d, %d, %d])\n" % (lower[0], lower[1], lower[2]))
            output.write("HSV_UPPER = np.array([%d, %d, %d])\n" % (upper[0], upper[1], upper[2]))
            output.close()

            print("")
            print("[SAVE] HSV range saved to:")
            print(self.save_path)
            self.print_hsv_range()
        except IOError as error:
            rospy.logerr("Cannot save HSV range: %s", error)

    def reset_trackbars(self):
        cv2.setTrackbarPos("H Min", "HSV Control", 0)
        cv2.setTrackbarPos("H Max", "HSV Control", 179)
        cv2.setTrackbarPos("S Min", "HSV Control", 0)
        cv2.setTrackbarPos("S Max", "HSV Control", 255)
        cv2.setTrackbarPos("V Min", "HSV Control", 0)
        cv2.setTrackbarPos("V Max", "HSV Control", 255)

        print("[RESET] HSV trackbars reset.")

    def run(self):
        rate = rospy.Rate(30)

        while not rospy.is_shutdown():
            if self.frame is not None:
                frame = self.frame.copy()
                lower, upper = self.get_hsv_range()

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)

                kernel = np.ones((3, 3), np.uint8)
                mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

                result = cv2.bitwise_and(frame, frame, mask=mask_clean)

                text_lower = "LOWER: H=%d S=%d V=%d" % (lower[0], lower[1], lower[2])
                text_upper = "UPPER: H=%d S=%d V=%d" % (upper[0], upper[1], upper[2])

                cv2.putText(frame, text_lower, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, text_upper, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imshow("Original", frame)
                cv2.imshow("Mask", mask_clean)
                cv2.imshow("Result", result)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("p"):
                self.print_hsv_range()
            elif key == ord("s"):
                self.save_hsv_range()
            elif key == ord("r"):
                self.reset_trackbars()
            elif key == ord("q") or key == 27:
                rospy.signal_shutdown("HSV tuner closed.")
                break

            rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        tuner = HSVTuner()
        tuner.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()