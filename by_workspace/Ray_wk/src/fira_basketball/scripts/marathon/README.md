# Marathon Vision Package

This ROS package is for HuroCup Marathon Pro League.

Main tasks:
1. Detect AprilTag 36h11 markers.
2. Detect arrow markers: forward, left, right.
3. Output marker class, center position, confidence, and tag ID.
4. Provide information to OP3 walking control node.

Folder usage:
- scripts/: Python detection and capture scripts
- launch/: ROS launch files
- config/: parameter files
- models/: YOLO model files
- data/: captured images and labels
