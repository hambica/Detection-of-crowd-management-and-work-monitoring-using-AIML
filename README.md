Project Title:
Detection of Crowd Management & Work Monitoring using AIML

Introduction:
This project leverages existing CCTV infrastructure combined with AI and machine learning to detect and monitor human presence for crowd management and workplace surveillance. By utilizing the YOLOv8 (You Only Look Once version 8) object detection model, the system can accurately identify people or groups in both images and video footage, it can also raise alerts when crowd size exceeds a defined threshold — making it useful for both safety and efficiency monitoring in public or organizational spaces.

## 🎯 Purpose

### 🔹 Crowd Management
- Monitor crowd density and movement in real-time.
- Detect overcrowding situations to prevent hazards or delays.

### 🔹 Work Monitoring
- Observe workplace zones for policy compliance.
- Improve operational visibility and detect anomalies in restricted areas.


## ⚙️ Install required libraries
pip install ultralytics

pip install opencv-python

pip install matplotlib

pip install numpy

How to Use the Application:
Launch the GUI: 
Run the Python script. The main interface will open.

Load YOLOv8 Model:

Click on "Generate & Load YoloV8 Model" to initialize the object detection model.

Detect from Images:

Click "Crowd Management from Images" to select and analyze a static image.

Bounding boxes will be displayed around detected individuals.

Detect from Videos:

Click "Crowd Management from Videos" to upload and process a video file.

Live object detection is shown frame-by-frame with bounding boxes and counts.

Press ‘Q’ to exit the video playback.

View Training Graph:

Click "YoloV8 Training Graph" to display a visual summary of model training performance.

Exit the Application:

Click "Exit" to close the application.

⚠️ Alert System
An alert popup will be triggered if the number of people exceeds the set threshold:

ALERT_THRESHOLD = 30  # You can change this in main1.py
