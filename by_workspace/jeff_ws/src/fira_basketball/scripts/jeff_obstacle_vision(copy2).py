#! /usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import heapq
import math

incomputer = False
if not incomputer:
    import rospy
    from std_msgs.msg import Float32, Int8
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    from fira_basketball.msg import ThreeDouble


DEBUG_MODE = True
DEBUG_MARKER = False

camera_ang = 60
x_angle = 0
camera_h = 44.0
hfov = 53.0
vfov = 38.0
hfov_rad = np.radians(hfov)
vfov_rad = np.radians(vfov)

# Hard obstacle radius. Keep 12 to avoid fake no-path.
obs_circle_default = 12
obs_circle = obs_circle_default
_resize = 0.4
closest_obs_z = 1000000

# Soft cost makes A* prefer the center of free space, not obstacle edge.
SAFE_PX = 14
DANGER_WEIGHT = 2.0

mapx = 120
mapz = 250
directionx = 0.0
directiony = 0.0
find_the_path = 1.0

fig, ax = plt.subplots()


def update_plot(px, py, color="red"):
    ax.clear()
    ax.set_xlim(-mapx, mapx)
    ax.set_ylim(0, mapz)
    ax.set_aspect("equal")
    ax.scatter(px, py, color=color, s=1)
    plt.draw()
    plt.pause(0.001)


def a_star(grid, danger=None):
    start_point = 0
    start = (grid.shape[0] - 1 - start_point, int((grid.shape[1] - 1) / 2))
    goal = 0

    def heuristic(node, goal_row):
        return node[0] - goal_row

    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = np.full((grid.shape[0], grid.shape[1], 2), -1, dtype=int)
    g_score = np.full(grid.shape, float("inf"), dtype=np.float64)
    g_score[start] = 0
    f_score = np.full(grid.shape, float("inf"), dtype=np.float64)
    f_score[start] = heuristic(start, goal)

    while open_list:
        current = heapq.heappop(open_list)[1]

        if current[0] == goal:
            final_path = []
            while current != start:
                final_path.append(current)
                current = tuple(came_from[current[0], current[1]])
            final_path.append(start)
            final_path.reverse()
            return final_path

        orderx = [-1, 0, 1]
        ordery = [-2, 0, 2]

        for i in orderx:
            for j in ordery:
                x = current[0] + i
                y = current[1] + j

                if (x == current[0] and y == current[1]) or x < 0 or y < 0 or x >= grid.shape[0] or y >= grid.shape[1] or grid[x, y] != 0:
                    continue

                step_cost = np.sqrt((x - current[0]) ** 2 + (y - current[1]) ** 2)
                danger_cost = danger[x, y] if danger is not None else 0.0
                tentative_g_score = g_score[current] + step_cost + danger_cost

                if tentative_g_score < g_score[x, y]:
                    came_from[x, y] = current
                    g_score[x, y] = tentative_g_score
                    f_score[x, y] = tentative_g_score + heuristic((x, y), goal)
                    heapq.heappush(open_list, (f_score[x, y], (x, y)))

    return []


if incomputer:
    def empty(v):
        pass

    cv2.namedWindow("track_bar")
    cv2.resizeWindow("track_bar", 640, 320)
    cv2.createTrackbar("x_angle", "track_bar", 0, 360, empty)
    cv2.createTrackbar("z_angle", "track_bar", 0, 360, empty)
    cv2.createTrackbar("hfov", "track_bar", 0, 100, empty)
    cv2.createTrackbar("vfov", "track_bar", 0, 100, empty)
    cv2.setTrackbarPos("x_angle", "track_bar", 180)
    cv2.setTrackbarPos("z_angle", "track_bar", 180 + int(camera_ang))
    cv2.setTrackbarPos("hfov", "track_bar", int(hfov))
    cv2.setTrackbarPos("vfov", "track_bar", int(vfov))
    cap = cv2.VideoCapture(0)


def rotate_point(point, center, angle):
    angle_rad = np.radians(angle)
    translated_point = (point[0] - center[0], point[1] - center[1])

    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    rotated_point = np.dot(rotation_matrix, translated_point)
    final_point = (rotated_point[0] + center[0], rotated_point[1] + center[1])
    return final_point


def create_map(data):
    global obs_circle, closest_obs_z, mapx, mapz
    global directionx, directiony, find_the_path

    if incomputer:
        ret, oimg = cap.read()
        if not ret:
            return
    else:
        bridge = CvBridge()
        try:
            oimg = bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

    oimg = cv2.GaussianBlur(oimg, (5, 5), 0)
    oimg = cv2.copyMakeBorder(oimg, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_w = oimg.shape[1]
    img_h = oimg.shape[0]
    hsv_img = cv2.cvtColor(oimg, cv2.COLOR_BGR2HSV)

    px = []
    py = []
    list_x = []
    list_y = []

    for idx in range(0, len(lower)):
        raw_mask = cv2.inRange(hsv_img, lower[idx], upper[idx])

        # Yellow low obstacle can be thick/flat. Use mask morphology instead of only edge.
        if cl[idx] in ["y", "wood", "w"]:
            output = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            output = cv2.morphologyEx(output, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            output = cv2.dilate(output, np.ones((5, 5), np.uint8), iterations=1)
        else:
            output = cv2.Laplacian(raw_mask, -1, 1, 1)
            ret, output = cv2.threshold(output, 30, 255, cv2.THRESH_BINARY)

        try:
            _, contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except ValueError:
            contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            if cl[idx] == "w" and area > img_w * img_h * 0.35:
                continue

            stapx = []
            stapy = []

            for i in range(0, len(contour), 10):
                xpix = contour[i][0][0]
                ypix = contour[i][0][1]

                if cl[idx] in ["y", "wood", "w"]:
                    stapx.append(xpix)
                    stapy.append(ypix)
                else:
                    if (contour[i - 1][0][0] - contour[i][0][0]) == 0:
                        continue
                    slop = abs((contour[i - 1][0][1] - contour[i][0][1]) / (contour[i - 1][0][0] - contour[i][0][0]))

                    try:
                        incontour = cv2.pointPolygonTest(contour, (int(xpix), int(ypix - 1)), False)
                    except Exception:
                        incontour = -10

                    if incontour > 0 and slop < 0.2:
                        stapx.append(xpix)
                        stapy.append(ypix)

            if len(stapx) > 0:
                list_x.append(stapx)
                list_y.append(stapy)

    for group in range(0, len(list_x)):
        for seq in range(0, len(list_x[group])):
            oimg = cv2.circle(oimg, (list_x[group][seq], list_y[group][seq]), 2, (255, 255, 0), 3)

    try:
        ztheta = np.degrees(np.arctan((0 - (img_h / 2.0)) * np.tan(np.radians(vfov / 2.0)) / (img_h / 2.0))) + camera_ang
        z = (camera_h / np.sin((ztheta) / 180.0 * np.pi)) * np.sin((90 - ztheta) / 180.0 * np.pi)

        rotate_ang = np.radians(abs(camera_ang - ztheta))
        rotate_z = z / np.cos(np.radians(ztheta))
        hfov_rotate = 2 * np.arcsin(np.tan(hfov_rad / 2.0) / np.sqrt(np.tan(rotate_ang) ** 2 + (1 / (np.cos(hfov_rad / 2.0))) ** 2))
        hfov_rotate = np.degrees(hfov_rotate)

        xtheta = np.degrees(np.arctan((img_w - (img_w / 2.0)) * np.tan(np.radians(hfov_rotate / 2.0)) / (img_w / 2.0)))
        x = rotate_z * np.tan(np.radians(xtheta))

        x_rotate = rotate_point((x, z), (0, 0), -abs(x_angle))
        x = max(x, x_rotate[0])
        z = max(z, x_rotate[1])

        mapx = min(abs(int(x - 5)), 300)
        mapz = min(abs(int(z - 2)), 450)
    except Exception:
        mapx = 100
        mapz = 80

    grid = np.zeros((mapz, mapx * 2), np.uint8)
    closest_obs_z = 1000000

    for group in range(0, len(list_x)):
        for seq in range(0, len(list_x[group])):
            try:
                ztheta = np.degrees(np.arctan((list_y[group][seq] - (img_h / 2.0)) * np.tan(np.radians(vfov / 2.0)) / (img_h / 2.0))) + camera_ang
                z = (camera_h / np.sin((ztheta) / 180.0 * np.pi)) * np.sin((90 - ztheta) / 180.0 * np.pi)

                rotate_ang = np.radians(abs(camera_ang - ztheta))
                rotate_z = z / np.cos(np.radians(ztheta))
                hfov_rotate = 2 * np.arcsin(np.tan(hfov_rad / 2.0) / np.sqrt(np.tan(rotate_ang) ** 2 + (1 / (np.cos(hfov_rad / 2.0))) ** 2))
                hfov_rotate = np.degrees(hfov_rotate)

                xtheta = np.degrees(np.arctan((list_x[group][seq] - (img_w / 2.0)) * np.tan(np.radians(hfov_rotate / 2.0)) / (img_w / 2.0)))
                x = rotate_z * np.tan(np.radians(xtheta))
                x, z = rotate_point((x, z), (0, 0), -x_angle)

                px.append(x)
                py.append(z)

                if z < closest_obs_z:
                    closest_obs_z = z

                if (mapz - int(z)) > 0:
                    cv2.circle(grid, (int(x) + mapx, mapz - int(z)), obs_circle, 255, -1)
            except Exception:
                pass

    grid = cv2.resize(grid, (0, 0), fx=_resize, fy=_resize)
    grid = cv2.copyMakeBorder(grid, 1, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)

    start_col = int((grid.shape[1] - 1) / 2)
    start_row = grid.shape[0] - 1
    cv2.circle(grid, (start_col, start_row), 3, 0, -1)

    # Soft safety cost: keeps path away from obstacle edge without increasing hard radius.
    free = np.where(grid > 0, 0, 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(free, cv2.DIST_L2, 3)
    danger = np.clip(SAFE_PX - dist_px, 0, SAFE_PX).astype(np.float32) * DANGER_WEIGHT
    cv2.circle(danger, (start_col, start_row), 6, 0, -1)

    final_path = a_star(grid, danger)

    if final_path == []:
        print("no path")
        directionx = -1.0
        directiony = 0.0
        find_the_path = 0.0
    else:
        dir_len = int(len(final_path) * 0.55)
        dir_len = max(1, min(dir_len, len(final_path) - 1))

        normalization = np.sqrt((final_path[dir_len][1] - final_path[0][1]) ** 2 + (final_path[dir_len][0] - final_path[0][0]) ** 2)

        if normalization < 1e-6:
            directionx = 1.0
            directiony = 0.0
        else:
            directionx = -(final_path[dir_len][0] - final_path[0][0]) / normalization
            directiony = -(final_path[dir_len][1] - final_path[0][1]) / normalization

        find_the_path = 1.0

        # Keep A* direction continuous. Do not force directiony to +1 / -1.
        if abs(directiony) < 0.12:
            directiony = 0.0
        else:
            directiony = float(np.clip(directiony, -0.65, 0.65))

        directionx = max(0.25, float(directionx))

    print("dir: ", directionx, directiony, find_the_path, "closest_z:", closest_obs_z)

    map_with_line = grid.copy()
    for i in range(0, len(final_path)):
        map_with_line[final_path[i][0]][final_path[i][1]] = 128

    size_show = 5 / _resize
    map_with_line = cv2.resize(map_with_line, (0, 0), fx=size_show, fy=size_show)

    cv2.imshow("Video Feed", oimg)
    cv2.imshow("map_with_line", map_with_line)
    cv2.waitKey(1)


def head_angle(msg):
    global camera_ang
    camera_ang = msg.data


# HSV ranges: yellow, blue, wood, white
# white: H 0~179, S 0~255, V 200~255
# wood : H 0~39,  S 22~255, V 63~255
lower = np.array([
    [20, 13, 221],   # yellow
    [70, 84, 131],    # blue
    [0, 22, 63],      # wood
    [0, 0, 200]       # white line / white start
])

upper = np.array([
    [56, 255, 255],   # yellow
    [161, 232, 255],  # blue
    [39, 255, 255],   # wood
    [179, 30, 255]   # white line / white start
])

cl = ["y", "b", "wood", "w"]


def main():
    if not incomputer:
        rospy.Subscriber("/cv_camera/image_raw", Image, create_map, queue_size=1)
        rospy.Subscriber("obstacle_head_angle", Int8, head_angle, queue_size=1)
        pub = rospy.Publisher("obstacle_vision_data", ThreeDouble, queue_size=1)
        pub_z = rospy.Publisher("obstacle_closest_z", Float32, queue_size=1)

        rospy.init_node("hybrid_obstacle_vision")

        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            msg = ThreeDouble()
            msg.first = directionx
            msg.second = directiony
            msg.third = find_the_path
            pub.publish(msg)
            pub_z.publish(Float32(data=closest_obs_z))
            rate.sleep()
    else:
        while True:
            create_map(0)
            global camera_ang, x_angle, hfov, vfov, hfov_rad, vfov_rad
            x_angle = cv2.getTrackbarPos("x_angle", "track_bar") - 180
            camera_ang = cv2.getTrackbarPos("z_angle", "track_bar") - 180
            hfov = cv2.getTrackbarPos("hfov", "track_bar")
            vfov = cv2.getTrackbarPos("vfov", "track_bar")
            hfov_rad = np.radians(hfov)
            vfov_rad = np.radians(vfov)


if __name__ == "__main__":
    main()
