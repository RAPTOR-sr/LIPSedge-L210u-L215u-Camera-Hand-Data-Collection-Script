# LIPSedge™ L210u/L215u Camera Data Collection Script

## Overview
This script enables **data collection, hand detection, and 3D point cloud visualization** using the **LIPSedge™ L210u/L215u depth cameras**.  
It integrates **OpenNI2, OpenCV, OpenGL, and MediaPipe** to capture RGB, Depth, IR, and Point Cloud data with real-time hand tracking.

---

## Features
- Real-time RGB, Depth, and IR frame capture.
- Hand landmark detection using MediaPipe.
- Point cloud generation and visualization with OpenGL.
- Automatic dataset saving (RGB, Depth, IR, Hand Point Cloud, and Full Point Cloud).
- User-defined Class ID and User ID for organized data storage.

---
## Update the openni2.initialize() path in data_collection.py to your SDK path:
```
openni2.initialize("C:\\Program Files\\LIPSedge Camera SDK 1.02\\...\\OpenNI2\\Redist")
```
---
## Output 
```
dataset/
  ├─ rgb/<class_id>/<user_id>/
  ├─ depth/<class_id>/<user_id>/
  ├─ amplitude/<class_id>/<user_id>/
  ├─ point_cloud/<class_id>/<user_id>/
  └─ full_point_cloud/<class_id>/<user_id>/
```
---

## Requirements
### Hardware
- LIPSedge™ L210u or L215u Camera
- Compatible system with OpenNI2 SDK installed.

### Software
- Python 3.8+
- Dependencies:
  ```bash
  pip install numpy opencv-python mediapipe pyopengl glfw scikit-learn
