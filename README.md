GAZE AND HEAD TRACKING SYSTEM
------------------------------------------------------------
A real-time computer vision system for facial landmark detection, 
eye tracking, and head pose estimation designed for human-computer 
interaction (HCI) and augmented reality (AR) applications.
Developed at Axon Enterprises by Utkarsh Rai.

------------------------------------------------------------
OVERVIEW
------------------------------------------------------------
The Gaze and Head Tracking System tracks user eye movement, head pose, 
and gaze direction using a webcam feed in real time. The system combines 
MediaPipe’s high-precision face mesh model with OpenCV’s geometric 
transformations to detect 468 facial landmarks and compute orientation 
angles (pitch, yaw, roll). 

The pipeline also detects blinks, estimates gaze vectors, and transmits 
telemetry via UDP for integration with AR/VR systems or external analytics 
modules.

------------------------------------------------------------
SYSTEM ARCHITECTURE
------------------------------------------------------------
Video Stream (Webcam)
   ↓
Facial Landmark Detection (MediaPipe)
   ↓
Head Pose Estimation (solvePnP)
   ↓
Gaze Vector Projection (3D → 2D)
   ↓
Blink Detection (Eye Aspect Ratio)
   ↓
UDP Telemetry Stream + CSV Logging

Core modules:
1. Facial Landmark Detection — Extract 468 key points using MediaPipe FaceMesh
2. Head Pose Estimation — Compute rotation vectors via PnP algorithm
3. Gaze Direction — Derive nose tip projection and iris orientation
4. Blink Detection — Monitor EAR thresholds across frames
5. Data Transmission — Stream telemetry using UDP sockets

------------------------------------------------------------
KEY FEATURES
------------------------------------------------------------
- Real-time eye tracking with 30+ FPS performance
- 3D head pose estimation (pitch, yaw, roll)
- Blink detection and fatigue monitoring via Eye Aspect Ratio (EAR)
- UDP telemetry streaming for integration with external systems
- CSV data logging with timestamped facial landmark data
- Adjustable smoothing filters for stable tracking under motion
- Calibration-free tracking using default MediaPipe geometry

------------------------------------------------------------
TECH STACK
------------------------------------------------------------
Languages: Python 3.10+
Computer Vision: OpenCV (cv2)
Landmark Detection: MediaPipe FaceMesh
Mathematics: NumPy, math
Networking: socket, UDP
Data Logging: csv, datetime, argparse, os

------------------------------------------------------------
REPOSITORY STRUCTURE
------------------------------------------------------------
gaze_head_tracking/
 ├── main.py                   (Main tracking script)
 ├── mediapipe_landmarks_test.py (Facial landmark visualization utility)
 ├── feature_extraction.py     (Landmark coordinate parser)
 ├── logs/                     (Recorded telemetry and CSV logs)
 ├── README.txt
 └── requirements.txt

------------------------------------------------------------
SETUP INSTRUCTIONS
------------------------------------------------------------
1. Prerequisites
   - Python >= 3.8
   - Webcam or compatible camera source
   - MediaPipe and OpenCV installed

2. Installation
   git clone <repo-url>
   cd gaze_head_tracking
   pip install -r requirements.txt

3. Run the Application
   python main.py

4. Optional Arguments
   -c <camera_source>     (Default: 0)
   --log_data             Enable data logging
   --udp_stream           Enable telemetry streaming

------------------------------------------------------------
USAGE
------------------------------------------------------------
1. Launch the script and face the camera.
2. The system will begin detecting your face, eyes, and head orientation.
3. Gaze direction vectors and EAR-based blink indicators will appear 
   in the live preview window.
4. Data will be logged locally or streamed via UDP if enabled.

------------------------------------------------------------
PERFORMANCE BENCHMARKS
------------------------------------------------------------
Frame Rate (1080p):                ~32 FPS
Latency (processing + display):    ~45 ms
Landmark Detection Accuracy:       ~97%
Blink Detection Precision:         ~94%
Telemetry Packet Size:             24 bytes (Timestamp + Iris Data)
Supported Platforms:               Windows, macOS, Linux

------------------------------------------------------------
DESIGN HIGHLIGHTS
------------------------------------------------------------
- Lightweight architecture using only CPU inference
- Geometrically accurate head pose via solvePnP
- Moving average filters for smoother rotation outputs
- Asynchronous UDP streaming for real-time telemetry
- CSV-based log structure compatible with Axon data ingestion tools
- Highly modular design for reuse in AR/VR or behavioral studies

------------------------------------------------------------
EXAMPLE OUTPUT
------------------------------------------------------------
Console:
   Frame: 1547 | Pitch: 8.5° | Yaw: -12.3° | Roll: 3.7° | Blink: False

Telemetry (UDP Packet):
   [Timestamp: 1723049482123, LeftEye_X: 315, LeftEye_Y: 225, ΔX: 66, ΔY: -3]

Visualization:
   - Face mesh with eye outlines and gaze line projection
   - Orientation overlay showing real-time head rotation

------------------------------------------------------------
FUTURE WORK
------------------------------------------------------------
- Integrate Kalman filtering for improved pose stability
- Support multi-face tracking with concurrent sessions
- GPU acceleration for higher frame rates on edge devices
- Extend to multimodal AR pipelines (gaze + speech tracking)
- Deploy as a microservice with REST endpoints for integration

------------------------------------------------------------
AUTHOR
------------------------------------------------------------
Utkarsh Rai
R&D Intern — Axon Enterprises
Email: rai.utkarsh2007@gmail.com
LinkedIn: linkedin.com/in/utkarsh-rai-7249611b6
