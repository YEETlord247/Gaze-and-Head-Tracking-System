"""
Eye Tracking and Head Pose Estimation

This script is designed to perform real-time eye tracking and head pose estimation using a webcam feed. 
It utilizes the MediaPipe library for facial landmark detection, which informs both eye tracking and 
head pose calculations. The purpose is to track the user's eye movements and head orientation, 
which can be applied in various domains such as HCI (Human-Computer Interaction), gaming, and accessibility tools.

Features:
- Real-time eye tracking to count blinks and calculate the eye aspect ratio for each frame.
- Head pose estimation to determine the orientation of the user's head in terms of pitch, yaw, and roll angles.
- Calibration feature to set the initial head pose as the reference zero position.
- Data logging for further analysis and debugging.

Requirements:
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- Other Dependencies: math, socket, argparse, time, csv, datetime, os

Methodology:
- The script uses the 468 facial landmarks provided by MediaPipe's FaceMesh model.
- Eye tracking is achieved by calculating the Eye Aspect Ratio (EAR) for each eye and detecting blinks based on EAR thresholds.
- Head pose is estimated using the solvePnP algorithm with a predefined 3D facial model and corresponding 2D landmarks detected from the camera feed.
- Angles are normalized to intuitive ranges (pitch: [-90, 90], yaw and roll: [-180, 180]).

Theory:
- EAR is used as a simple yet effective metric for eye closure detection.
- Head pose angles are derived using a perspective-n-point approach, which estimates an object's pose from its 2D image points and 3D model points.

UDP Packet Structure:
- The UDP packet consists of a timestamp and four other integer values.
- Packet Type: Mixed (int64 for timestamp, int32 for other values)
- Packet Structure: [timestamp (int64), l_cx (int32), l_cy (int32), l_dx (int32), l_dy (int32)]
- Packet Size: 24 bytes (8 bytes for int64 timestamp, 4 bytes each for the four int32 values)

Example Packets:
- Example 1: [1623447890123, 315, 225, 66, -3]
- Example 2: [1623447891123, 227, 68, -1, 316]

Parameters:
You can change parameters such as face width, moving average window, webcam ID, terminal outputs, on-screen data, logging detail, etc., from the code.

Author: Alireza Bagheri
GitHub: https://github.com/alireza787b/Python-Gaze-Face-Tracker
Email: p30planets@gmail.com
LinkedIn: https://www.linkedin.com/in/alireza787b
Date: November 2023

Inspiration:
Initially inspired by Asadullah Dal's iris segmentation project (https://github.com/Asadullah-Dal17/iris-Segmentation-mediapipe-python). 
The blink detection feature is also contributed by Asadullah Dal (GitHub: Asadullah-Dal17).

Usage:
- Run the script in a Python environment with the necessary dependencies installed. The script accepts command-line arguments for camera source configuration.
- Press 'c' to recalibrate the head pose estimation to the current orientation.
- Press 't' to toggle 360-degree background tracking on/off.
- Press 'z' to reset camera rotation to 0°.
- Press 'r' to start/stop logging.
- Press 'q' to exit the program.
- Output is displayed in a window with live feed and annotations, and logged to a CSV file for further analysis.

Ensure that all dependencies, especially MediaPipe, OpenCV, and NuxmPy, are installed before running the script.

Note:
This project is intended for educational and research purposes in fields like aviation, human-computer interaction, and more.
"""


import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import os
from AngleBuffer import AngleBuffer

# Update officer image loading to include alpha channel
officer_img = cv.imread("/Users/urai/Desktop/Gaze Tracker/Python-Gaze-Face-Tracker copy/officer.png", cv.IMREAD_UNCHANGED)
if officer_img is None:
    raise FileNotFoundError("officer.png not found. Please ensure the file exists at the specified path.")

#-----------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------

# Parameters Documentation

## User-Specific Measurements
# USER_FACE_WIDTH: The horizontal distance between the outer edges of the user's cheekbones in millimeters. 
# This measurement is used to scale the 3D model points for head pose estimation.
# Measure your face width and adjust the value accordingly.
USER_FACE_WIDTH = 140  # [mm]

## Camera Parameters (not currently used in calculations)
# NOSE_TO_CAMERA_DISTANCE: The distance from the tip of the nose to the camera lens in millimeters.
# Intended for future use where accurate physical distance measurements may be necessary.
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]

## Configuration Parameters
# PRINT_DATA: Enable or disable the printing of data to the console for debugging.
PRINT_DATA = True

# DEFAULT_WEBCAM: Default camera source index. '0' usually refers to the built-in webcam.
DEFAULT_WEBCAM = 0

# SHOW_ALL_FEATURES: If True, display all facial landmarks on the video feed.
SHOW_ALL_FEATURES = True

# LOG_DATA: Enable or disable logging of data to a CSV file.
LOG_DATA = True

# LOG_ALL_FEATURES: If True, log all facial landmarks to the CSV file.
LOG_ALL_FEATURES = False

# ENABLE_HEAD_POSE: Enable the head position and orientation estimator.
ENABLE_HEAD_POSE = True

## Grid Visualization Parameters
# GRID_SIZE: Size of the grid window (width, height)
GRID_SIZE = (800, 600)

# GRID_SCALE: How far the gaze line extends (higher = more sensitive)
GRID_SCALE = 50

# SHOW_GRID: Enable or disable the grid visualization window
SHOW_GRID = True

## 360-Degree Tracking Parameters
# ENABLE_360_TRACKING: Enable background-based camera movement tracking
ENABLE_360_TRACKING = True

# OPTICAL_FLOW_PARAMS: Parameters for optical flow calculation
OPTICAL_FLOW_PARAMS = {
    'maxCorners': 200,
    'qualityLevel': 0.01,
    'minDistance': 10,
    'blockSize': 3,
    'gradientSize': 3,
    'useHarrisDetector': False,
    'k': 0.04
}

# OPTICAL_FLOW_LK_PARAMS: Parameters for Lucas-Kanade optical flow
OPTICAL_FLOW_LK_PARAMS = {
    'winSize': (15, 15),
    'maxLevel': 2,
    'criteria': (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
}

# CAMERA_ROTATION_SMOOTHING: Smoothing factor for camera rotation (0-1)
CAMERA_ROTATION_SMOOTHING = 0.8

# MIN_MOVEMENT_THRESHOLD: Minimum movement threshold to register camera rotation (pixels)
MIN_MOVEMENT_THRESHOLD = 5.0

## Real-World Calibration Parameters
# DISTANCE_TO_TARGET: Distance from camera (shoulder) to target person in meters
DISTANCE_TO_TARGET = 2.0  # 2 meters (default, will be dynamically calculated)

# CAMERA_FOV: GoPro horizontal field of view in degrees
CAMERA_FOV = 120.0  # GoPro Hero typical wide FOV

# Face measurement parameters for distance estimation
AVERAGE_FACE_WIDTH_MM = 140  # Average human face width in millimeters
CAMERA_FOCAL_LENGTH_MM = 2.8  # GoPro Hero focal length (approximate)
SENSOR_WIDTH_MM = 6.17  # GoPro Hero sensor width (approximate)

# Dynamic distance estimation
estimated_distance = 2.0  # Will be updated in real-time

# Calculate real-world scale: meters per pixel (will be updated dynamically)
# At distance D with FOV, the horizontal width in real-world = 2 * D * tan(FOV/2)
REAL_WORLD_WIDTH = 2 * estimated_distance * math.tan(math.radians(CAMERA_FOV / 2))
METERS_PER_PIXEL = REAL_WORLD_WIDTH / GRID_SIZE[0]

# Grid spacing in real-world units
GRID_SPACING_METERS = 0.5  # 50cm grid lines
GRID_SPACING_PIXELS = int(GRID_SPACING_METERS / METERS_PER_PIXEL)

## Firearm Detection Parameters
# ALERT_THRESHOLD: Time in seconds of consecutive gaze contact needed for alert
ALERT_THRESHOLD = 1.0

# Human figure position in real-world coordinates (meters from camera center)
HUMAN_OFFSET_X_METERS = 0.3   # 30cm to the right
HUMAN_OFFSET_Y_METERS = 0.0   # same level as camera

# Firearm hitbox in real-world coordinates (relative to human center)
FIREARM_OFFSET_X_METERS = 0.15  # 15cm to the right of human center (hip holster)
FIREARM_OFFSET_Y_METERS = 1.2   # 1.2m down from human center (off the grid)
FIREARM_SIZE_METERS = 0.1       # 10cm hitbox size

# Optionally, make the grid window taller to visualize off-screen hitboxes
# GRID_SIZE = (800, 1200)  # Uncomment to double the height

## Logging Configuration
# LOG_FOLDER: Directory where log files will be stored.
LOG_FOLDER = "logs"

## Server Configuration
# SERVER_IP: IP address of the server for sending data via UDP (default is localhost).
SERVER_IP = "127.0.0.1"

# SERVER_PORT: Port number for the server to listen on.
SERVER_PORT = 7070

## Blink Detection Parameters
# SHOW_ON_SCREEN_DATA: If True, display blink count and head pose angles on the video feed.
SHOW_ON_SCREEN_DATA = True

# TOTAL_BLINKS: Counter for the total number of blinks detected.
TOTAL_BLINKS = 0

# EYES_BLINK_FRAME_COUNTER: Counter for consecutive frames with detected potential blinks.
EYES_BLINK_FRAME_COUNTER = 0

# BLINK_THRESHOLD: Eye aspect ratio threshold below which a blink is registered.
BLINK_THRESHOLD = 0.51

# EYE_AR_CONSEC_FRAMES: Number of consecutive frames below the threshold required to confirm a blink.
EYE_AR_CONSEC_FRAMES = 2

## Head Pose Estimation Landmark Indices
# These indices correspond to the specific facial landmarks used for head pose estimation.
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

## MediaPipe Model Confidence Parameters
# These thresholds determine how confidently the model must detect or track to consider the results valid.
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

## Angle Normalization Parameters
# MOVING_AVERAGE_WINDOW: The number of frames over which to calculate the moving average for smoothing angles.
MOVING_AVERAGE_WINDOW = 10

# Initial Calibration Flags
# initial_pitch, initial_yaw, initial_roll: Store the initial head pose angles for calibration purposes.
# calibrated: A flag indicating whether the initial calibration has been performed.
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False

# User-configurable parameters
PRINT_DATA = True  # Enable/disable data printing
DEFAULT_WEBCAM = 0  # Default webcam number
SHOW_ALL_FEATURES = True  # Show all facial landmarks if True
LOG_DATA = True  # Enable logging to CSV
LOG_ALL_FEATURES = False  # Log all facial landmarks if True
LOG_FOLDER = "logs"  # Folder to store log files

# Server configuration
SERVER_IP = "127.0.0.1"  # Set the server IP address (localhost)
SERVER_PORT = 7070  # Set the server port

# eyes blinking variables
SHOW_BLINK_COUNT_ON_SCREEN = True  # Toggle to show the blink count on the video feed
TOTAL_BLINKS = 0  # Tracks the total number of blinks detected
EYES_BLINK_FRAME_COUNTER = (
    0  # Counts the number of consecutive frames with a potential blink
)
BLINK_THRESHOLD = 0.51  # Threshold for the eye aspect ratio to trigger a blink
EYE_AR_CONSEC_FRAMES = (
    2  # Number of consecutive frames below the threshold to confirm a blink
)
# SERVER_ADDRESS: Tuple containing the SERVER_IP and SERVER_PORT for UDP communication.
SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)

# Firearm detection state variables
firearm_contact_start_time = None
firearm_alert_active = False
last_contact_time = None

# 360-degree tracking state variables
global camera_rotation_angle, previous_frame, previous_points, camera_rotation_history
camera_rotation_angle = 0.0  # Current camera rotation angle in degrees
previous_frame = None
previous_points = None
camera_rotation_history = []  # Store recent rotation angles for smoothing
MAX_ROTATION_HISTORY = 10

camera_rotation_angle_x = 0.0
camera_rotation_angle_y = 0.0
camera_rotation_history_x = []
camera_rotation_history_y = []

#If set to false it will wait for your command (hittig 'r') to start logging.
IS_RECORDING = False  # Controls whether data is being logged

# Command-line arguments for camera source
parser = argparse.ArgumentParser(description="Eye Tracking Application")
parser.add_argument(
    "-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM)
)
args = parser.parse_args()

# Iris and eye corners landmarks indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]  # Left eye Left Corner
L_H_RIGHT = [133]  # Left eye Right Corner
R_H_LEFT = [362]  # Right eye Left Corner
R_H_RIGHT = [263]  # Right eye Right Corner

# Blinking Detection landmark's indices.
# P0, P3, P4, P5, P8, P11, P12, P13
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]

# Face Selected points indices for Head Pose Estimation
_indices_pose = [1, 33, 61, 199, 263, 291]

# Server address for UDP socket communication
SERVER_ADDRESS = (SERVER_IP, 7070)


# Function to calculate vector position
def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1


def euclidean_distance_3D(points):
    """Calculates the Euclidean distance between two points in 3D space.

    Args:
        points: A list of 3D points.

    Returns:
        The Euclidean distance between the two points.

        # Comment: This function calculates the Euclidean distance between two points in 3D space.
    """

    # Get the three points.
    P0, P3, P4, P5, P8, P11, P12, P13 = points

    # Calculate the numerator.
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )

    # Calculate the denominator.
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3

    # Calculate the distance.
    distance = numerator / denominator

    return distance

def estimate_head_pose(landmarks, image_size):
    # Scale factor based on user's face width (assumes model face width is 150mm)
    scale_factor = USER_FACE_WIDTH / 150.0
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),        # Chin
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),     # Left eye left corner
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),      # Right eye right corner
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),    # Left Mouth corner
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)      # Right mouth corner
    ])
    


    # Camera internals
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1))

    # 2D image points from landmarks, using defined indices
    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],            # Nose tip
        landmarks[CHIN_INDEX],                # Chin
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],  # Left eye left corner
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],  # Right eye right corner
        landmarks[LEFT_MOUTH_CORNER_INDEX],      # Left mouth corner
        landmarks[RIGHT_MOUTH_CORNER_INDEX]      # Right mouth corner
    ], dtype="double")


        # Solve for pose
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)

    # Combine rotation matrix and translation vector to form a 3x4 projection matrix
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))

    # Decompose the projection matrix to extract Euler angles
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]


     # Normalize the pitch angle
    pitch = normalize_pitch(pitch)

    return pitch, yaw, roll

def normalize_pitch(pitch):
    """
    Normalize the pitch angle to be within the range of [-90, 90].

    Args:
        pitch (float): The raw pitch angle in degrees.

    Returns:
        float: The normalized pitch angle.
    """
    # Map the pitch angle to the range [-180, 180]
    if pitch > 180:
        pitch -= 360

    # Invert the pitch angle for intuitive up/down movement
    pitch = -pitch

    # Ensure that the pitch is within the range of [-90, 90]
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
        
    pitch = -pitch

    return pitch



# This function calculates the blinking ratio of a person.

def meters_to_pixels(meters_x, meters_y):
    """Convert real-world coordinates (meters) to pixel coordinates"""
    center_x, center_y = GRID_SIZE[0] // 2, GRID_SIZE[1] // 2
    pixel_x = int(center_x + meters_x / METERS_PER_PIXEL)
    pixel_y = int(center_y + meters_y / METERS_PER_PIXEL)
    return pixel_x, pixel_y

def pixels_to_meters(pixel_x, pixel_y):
    """Convert pixel coordinates to real-world coordinates (meters)"""
    center_x, center_y = GRID_SIZE[0] // 2, GRID_SIZE[1] // 2
    meters_x = (pixel_x - center_x) * METERS_PER_PIXEL
    meters_y = (pixel_y - center_y) * METERS_PER_PIXEL
    return meters_x, meters_y

def estimate_distance_from_face(mesh_points, img_width):
    """
    Estimate distance to person based on face width in image
    Uses the distance between outer eye corners as face width reference
    """
    # Get outer eye corner landmarks
    left_eye_outer = mesh_points[LEFT_EYE_OUTER_CORNER][0]
    right_eye_outer = mesh_points[RIGHT_EYE_OUTER_CORNER][0]
    
    # Calculate face width in pixels
    face_width_pixels = abs(left_eye_outer[0] - right_eye_outer[0])
    
    # Calculate focal length in pixels from camera specs
    focal_length_pixels = (CAMERA_FOCAL_LENGTH_MM / SENSOR_WIDTH_MM) * img_width
    
    # Distance calculation using pinhole camera model
    # Distance = (Real_object_size * Focal_length_pixels) / Object_size_pixels
    distance_mm = (AVERAGE_FACE_WIDTH_MM * focal_length_pixels) / face_width_pixels
    distance_meters = distance_mm / 1000.0
    
    return distance_meters

def update_real_world_scale(distance):
    """Update the real-world coordinate scaling based on estimated distance"""
    global REAL_WORLD_WIDTH, METERS_PER_PIXEL
    REAL_WORLD_WIDTH = 2 * distance * math.tan(math.radians(CAMERA_FOV / 2))
    METERS_PER_PIXEL = REAL_WORLD_WIDTH / GRID_SIZE[0]

def create_grid_window(distance_to_person):
    """
    Create a grid visualization window with coordinate system
    Camera is at origin (center), positive X is right, positive Y is down
    Grid lines represent real-world distances
    """
    grid = np.zeros((GRID_SIZE[1], GRID_SIZE[0], 3), dtype=np.uint8)
    
    center_x, center_y = GRID_SIZE[0] // 2, GRID_SIZE[1] // 2
    
    # Draw grid lines with real-world spacing
    # Vertical lines (every 50cm)
    for i in range(-10, 11):
        meters_from_center = i * GRID_SPACING_METERS
        x_pixel = int(center_x + meters_from_center / METERS_PER_PIXEL)
        if 0 <= x_pixel < GRID_SIZE[0]:
            cv.line(grid, (x_pixel, 0), (x_pixel, GRID_SIZE[1]), (50, 50, 50), 1)
            # Add distance labels
            if i != 0:  # Don't label the center line
                cv.putText(grid, f"{meters_from_center:.1f}m", 
                          (x_pixel - 15, GRID_SIZE[1] - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    
    # Horizontal lines (every 50cm)
    for i in range(-10, 11):
        meters_from_center = i * GRID_SPACING_METERS
        y_pixel = int(center_y + meters_from_center / METERS_PER_PIXEL)
        if 0 <= y_pixel < GRID_SIZE[1]:
            cv.line(grid, (0, y_pixel), (GRID_SIZE[0], y_pixel), (50, 50, 50), 1)
            # Add distance labels
            if i != 0:  # Don't label the center line
                cv.putText(grid, f"{meters_from_center:.1f}m", 
                          (5, y_pixel + 5), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
    
    # Draw center axes (camera position)
    cv.line(grid, (center_x, 0), (center_x, GRID_SIZE[1]), (100, 100, 100), 2)  # Y-axis
    cv.line(grid, (0, center_y), (GRID_SIZE[0], center_y), (100, 100, 100), 2)  # X-axis
    
    # Mark camera position (origin)
    cv.circle(grid, (center_x, center_y), 8, (255, 255, 255), -1)
    cv.putText(grid, "Camera", (center_x - 30, center_y - 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(grid, "0.0m", (center_x + 15, center_y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw officer image using real-world coordinates
    # Move officer image up by 0.5 meters
    officer_offset_y_meters = HUMAN_OFFSET_Y_METERS - 0.5
    origin_x, origin_y = meters_to_pixels(HUMAN_OFFSET_X_METERS, officer_offset_y_meters)
    
    # Scale officer image to 1.8m height in real-world units
    target_height_meters = 1.8
    target_height_pixels = int(target_height_meters / METERS_PER_PIXEL)
    scale_factor = target_height_pixels / officer_img.shape[0]
    scaled_width = int(officer_img.shape[1] * scale_factor)
    scaled_officer = cv.resize(officer_img, (scaled_width, target_height_pixels), interpolation=cv.INTER_AREA)
    
    # Overlay the officer image on the grid with alpha blending (always draw, crop if needed)
    grid_h, grid_w = grid.shape[:2]
    img_h, img_w = scaled_officer.shape[:2]
    # Calculate top-left position so that the right shoulder (top-right corner) is at origin
    top_left_x = origin_x - img_w
    top_left_y = origin_y

    # Calculate the region of interest (ROI) for both grid and officer image
    roi_x1 = max(top_left_x, 0)
    roi_y1 = max(top_left_y, 0)
    roi_x2 = min(top_left_x + img_w, grid_w)
    roi_y2 = min(top_left_y + img_h, grid_h)

    # Corresponding region in the officer image
    img_roi_x1 = roi_x1 - top_left_x
    img_roi_y1 = roi_y1 - top_left_y
    img_roi_x2 = img_roi_x1 + (roi_x2 - roi_x1)
    img_roi_y2 = img_roi_y1 + (roi_y2 - roi_y1)

    if roi_x1 < roi_x2 and roi_y1 < roi_y2:
        officer_roi = scaled_officer[img_roi_y1:img_roi_y2, img_roi_x1:img_roi_x2]
        grid_roi = grid[roi_y1:roi_y2, roi_x1:roi_x2]
        if officer_roi.shape[2] == 4:
            officer_rgb = officer_roi[..., :3]
            alpha = officer_roi[..., 3:] / 255.0
            grid[roi_y1:roi_y2, roi_x1:roi_x2] = (alpha * officer_rgb + (1 - alpha) * grid_roi).astype(np.uint8)
        else:
            grid[roi_y1:roi_y2, roi_x1:roi_x2] = officer_roi
    
    # Move firearm hitbox 0.5 meters to the left
    firearm_x, firearm_y = meters_to_pixels(HUMAN_OFFSET_X_METERS + FIREARM_OFFSET_X_METERS - 0.5, HUMAN_OFFSET_Y_METERS + FIREARM_OFFSET_Y_METERS)
    firearm_size_pixels = int(FIREARM_SIZE_METERS / METERS_PER_PIXEL)
    cv.rectangle(grid, 
                (firearm_x - firearm_size_pixels//2, firearm_y - firearm_size_pixels//2),
                (firearm_x + firearm_size_pixels//2, firearm_y + firearm_size_pixels//2),
                (0, 255, 255), 2)  # Yellow hitbox
    cv.putText(grid, "FIREARM", (firearm_x - 30, firearm_y - 35), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Add calibration info
    cv.putText(grid, f"Estimated Distance: {distance_to_person:.2f}m", (10, GRID_SIZE[1] - 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv.putText(grid, f"FOV: {CAMERA_FOV}°", (10, GRID_SIZE[1] - 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(grid, f"Scale: {METERS_PER_PIXEL:.3f}m/px", (10, GRID_SIZE[1] - 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(grid, f"Face Width: {AVERAGE_FACE_WIDTH_MM}mm", (10, GRID_SIZE[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return grid

def blinking_ratio(landmarks):
    """Calculates the blinking ratio of a person.

    Args:
        landmarks: A facial landmarks in 3D normalized.

    Returns:
        The blinking ratio of the person, between 0 and 1, where 0 is fully open and 1 is fully closed.

    """

    # Get the right eye ratio.
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])

    # Get the left eye ratio.
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])

    # Calculate the blinking ratio.
    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2

    return ratio

# In detect_background_movement, accumulate both x and y pan angles
# Return both horizontal and vertical pan angles (in degrees)
def detect_background_movement(current_frame, previous_frame, previous_points):
    """
    Detect camera movement using optical flow on background features
    Returns (horizontal_pan_angle, vertical_pan_angle), new_points
    """
    global camera_rotation_history
    if previous_frame is None or previous_points is None:
        new_points = cv.goodFeaturesToTrack(
            current_frame,
            OPTICAL_FLOW_PARAMS['maxCorners'],
            OPTICAL_FLOW_PARAMS['qualityLevel'],
            OPTICAL_FLOW_PARAMS['minDistance'],
            mask=None,
            blockSize=OPTICAL_FLOW_PARAMS['blockSize'],
            useHarrisDetector=OPTICAL_FLOW_PARAMS['useHarrisDetector'],
            k=OPTICAL_FLOW_PARAMS['k']
        )
        return (0.0, 0.0), new_points
    new_points, status, error = cv.calcOpticalFlowPyrLK(
        previous_frame, current_frame, previous_points, None,
        **OPTICAL_FLOW_LK_PARAMS
    )
    if new_points is None:
        new_points = cv.goodFeaturesToTrack(
            current_frame,
            OPTICAL_FLOW_PARAMS['maxCorners'],
            OPTICAL_FLOW_PARAMS['qualityLevel'],
            OPTICAL_FLOW_PARAMS['minDistance'],
            mask=None,
            blockSize=OPTICAL_FLOW_PARAMS['blockSize'],
            useHarrisDetector=OPTICAL_FLOW_PARAMS['useHarrisDetector'],
            k=OPTICAL_FLOW_PARAMS['k']
        )
        return (0.0, 0.0), new_points
    good_old = previous_points[status == 1]
    good_new = new_points[status == 1]
    if len(good_old) < 10:
        new_points = cv.goodFeaturesToTrack(
            current_frame,
            OPTICAL_FLOW_PARAMS['maxCorners'],
            OPTICAL_FLOW_PARAMS['qualityLevel'],
            OPTICAL_FLOW_PARAMS['minDistance'],
            mask=None,
            blockSize=OPTICAL_FLOW_PARAMS['blockSize'],
            useHarrisDetector=OPTICAL_FLOW_PARAMS['useHarrisDetector'],
            k=OPTICAL_FLOW_PARAMS['k']
        )
        return (0.0, 0.0), new_points
    movement_vectors = good_new - good_old
    avg_horizontal_movement = np.mean(movement_vectors[:, 0])
    avg_vertical_movement = np.mean(movement_vectors[:, 1])
    pixels_per_degree_x = current_frame.shape[1] / CAMERA_FOV
    pixels_per_degree_y = current_frame.shape[0] / CAMERA_FOV  # Assume square FOV for simplicity
    rotation_angle_x = avg_horizontal_movement / pixels_per_degree_x
    rotation_angle_y = avg_vertical_movement / pixels_per_degree_y
    if abs(avg_horizontal_movement) < MIN_MOVEMENT_THRESHOLD:
        rotation_angle_x = 0.0
    if abs(avg_vertical_movement) < MIN_MOVEMENT_THRESHOLD:
        rotation_angle_y = 0.0
    # Smooth both angles
    if 'camera_rotation_history_x' not in globals():
        global camera_rotation_history_x, camera_rotation_history_y
        camera_rotation_history_x = []
        camera_rotation_history_y = []
    camera_rotation_history_x.append(rotation_angle_x)
    camera_rotation_history_y.append(rotation_angle_y)
    if len(camera_rotation_history_x) > MAX_ROTATION_HISTORY:
        camera_rotation_history_x.pop(0)
    if len(camera_rotation_history_y) > MAX_ROTATION_HISTORY:
        camera_rotation_history_y.pop(0)
    smoothed_rotation_x = np.mean(camera_rotation_history_x)
    smoothed_rotation_y = np.mean(camera_rotation_history_y)
    return (smoothed_rotation_x, smoothed_rotation_y), new_points

def update_grid_with_gaze_and_pan(angle_x, angle_y, distance_to_person, camera_rotation_x=0.0, camera_rotation_y=0.0):
    """
    Update grid visualization with gaze direction and camera pan (both x and y).
    Only the origin and endpoint of the gaze vector are shifted by camera pan.
    The officer, firearm, and grid remain fixed at the center.
    """
    global firearm_contact_start_time, firearm_alert_active, last_contact_time
    grid = create_grid_window(distance_to_person)
    shift_pixels_x = int((camera_rotation_x / CAMERA_FOV) * GRID_SIZE[0])
    shift_pixels_y = int((camera_rotation_y / CAMERA_FOV) * GRID_SIZE[1])
    center_x, center_y = GRID_SIZE[0] // 2, GRID_SIZE[1] // 2
    gaze_origin_x = center_x - shift_pixels_x
    gaze_origin_y = center_y - shift_pixels_y
    gaze_x = int(gaze_origin_x - angle_y * GRID_SCALE)
    gaze_y = int(gaze_origin_y - angle_x * GRID_SCALE)
    # Firearm and officer positions are NOT shifted
    firearm_x, firearm_y = meters_to_pixels(HUMAN_OFFSET_X_METERS + FIREARM_OFFSET_X_METERS - 0.5, 
                                           HUMAN_OFFSET_Y_METERS + FIREARM_OFFSET_Y_METERS)
    firearm_size_pixels = int(FIREARM_SIZE_METERS / METERS_PER_PIXEL)
    target_height_meters = 1.8
    target_height_pixels = int(target_height_meters / METERS_PER_PIXEL)
    scale_factor = target_height_pixels / officer_img.shape[0]
    scaled_width = int(officer_img.shape[1] * scale_factor)
    img_h, img_w = target_height_pixels, scaled_width
    officer_offset_y_meters = HUMAN_OFFSET_Y_METERS - 0.5
    origin_x, origin_y = meters_to_pixels(HUMAN_OFFSET_X_METERS, officer_offset_y_meters)
    top_left_x = origin_x - img_w
    top_left_y = origin_y
    officer_left = top_left_x
    officer_right = top_left_x + img_w
    officer_top = top_left_y
    officer_bottom = top_left_y + img_h
    # Overlay the officer image (fixed at center)
    grid_h, grid_w = grid.shape[:2]
    if 0 <= officer_right and officer_left < grid_w:
        roi_x1 = max(top_left_x, 0)
        roi_y1 = max(top_left_y, 0)
        roi_x2 = min(top_left_x + img_w, grid_w)
        roi_y2 = min(top_left_y + img_h, grid_h)
        img_roi_x1 = roi_x1 - top_left_x
        img_roi_y1 = roi_y1 - top_left_y
        img_roi_x2 = img_roi_x1 + (roi_x2 - roi_x1)
        img_roi_y2 = img_roi_y1 + (roi_y2 - roi_y1)
        if roi_x1 < roi_x2 and roi_y1 < roi_y2:
            officer_roi = cv.resize(officer_img, (scaled_width, target_height_pixels), interpolation=cv.INTER_AREA)[img_roi_y1:img_roi_y2, img_roi_x1:img_roi_x2]
            grid_roi = grid[roi_y1:roi_y2, roi_x1:roi_x2]
            if officer_roi.shape[2] == 4:
                officer_rgb = officer_roi[..., :3]
                alpha = officer_roi[..., 3:] / 255.0
                grid[roi_y1:roi_y2, roi_x1:roi_x2] = (alpha * officer_rgb + (1 - alpha) * grid_roi).astype(np.uint8)
            else:
                grid[roi_y1:roi_y2, roi_x1:roi_x2] = officer_roi
    # Firearm hitbox (fixed)
    square_size = 10
    gaze_left = gaze_x - square_size
    gaze_right = gaze_x + square_size
    gaze_top = gaze_y - square_size
    gaze_bottom = gaze_y + square_size
    firearm_left = firearm_x - firearm_size_pixels//2
    firearm_right = firearm_x + firearm_size_pixels//2
    firearm_top = firearm_y - firearm_size_pixels//2
    firearm_bottom = firearm_y + firearm_size_pixels//2
    collision = (gaze_right >= firearm_left and gaze_left <= firearm_right and
                gaze_bottom >= firearm_top and gaze_top <= firearm_bottom)
    gaze_meters_x, gaze_meters_y = pixels_to_meters(gaze_x, gaze_y)
    firearm_meters_x, firearm_meters_y = pixels_to_meters(firearm_x, firearm_y)
    print(f"[DEBUG] Gaze at ({gaze_meters_x:.2f}m, {gaze_meters_y:.2f}m), Firearm at ({firearm_meters_x:.2f}m, {firearm_meters_y:.2f}m), Collision: {collision}")
    current_time = time.time()
    global officer_contact_start_time, officer_alert_active
    if collision:
        if firearm_contact_start_time is None:
            firearm_contact_start_time = current_time
        contact_duration = current_time - firearm_contact_start_time
        if contact_duration >= ALERT_THRESHOLD:
            firearm_alert_active = True
        last_contact_time = current_time
        cv.rectangle(grid, (firearm_left, firearm_top), (firearm_right, firearm_bottom), (0, 0, 255), -1)
    else:
        firearm_contact_start_time = None
        firearm_alert_active = False
        if last_contact_time is None or (current_time - last_contact_time) > 0.5:
            firearm_alert_active = False
    # Draw gaze vector (origin and endpoint are shifted)
    cv.line(grid, (gaze_origin_x, gaze_origin_y), (gaze_x, gaze_y), (255, 0, 255), 2)
    # Add 'Bystander' label at the origin in pink
    cv.putText(grid, 'Bystander', (int(gaze_origin_x) + 10, int(gaze_origin_y) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    gaze_color = (0, 0, 255) if not collision else (0, 255, 255)
    cv.rectangle(grid, (gaze_x - square_size, gaze_y - square_size), (gaze_x + square_size, gaze_y + square_size), gaze_color, -1)
    cv.putText(grid, f"Gaze: {gaze_meters_x:.2f}m, {gaze_meters_y:.2f}m", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv.putText(grid, f"Angles: Pitch={angle_x:.1f}°, Yaw={angle_y:.1f}°", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv.putText(grid, f"Camera Pan: {camera_rotation_x:.1f}°, {camera_rotation_y:.1f}°", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    if collision:
        contact_duration = current_time - firearm_contact_start_time
        cv.putText(grid, f"FIREARM CONTACT: {contact_duration:.1f}s", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if firearm_alert_active:
        cv.putText(grid, "*** FIREARM GAZE ALERT ***", (center_x - 150, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        if int(current_time * 4) % 2:
            cv.rectangle(grid, (10, 10), (GRID_SIZE[0] - 10, 120), (0, 0, 255), 5)
    officer_collision = (gaze_right >= officer_left and gaze_left <= officer_right and
                        gaze_bottom >= officer_top and gaze_top <= officer_bottom)
    if collision:
        officer_collision = False
    if 'officer_contact_start_time' not in globals():
        officer_contact_start_time = None
    if 'officer_alert_active' not in globals():
        officer_alert_active = False
    if officer_collision:
        if officer_contact_start_time is None:
            officer_contact_start_time = current_time
        contact_duration = current_time - officer_contact_start_time
        if contact_duration >= 0.5:
            officer_alert_active = True
        last_contact_time = current_time
    else:
        officer_contact_start_time = None
        if last_contact_time is None or (current_time - last_contact_time) > 0.5:
            officer_alert_active = False
    if officer_alert_active and not firearm_alert_active:
        alert_text = "* OFFICER GAZE ALERT *"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.9
        thickness = 3
        (text_width, text_height), baseline = cv.getTextSize(alert_text, font, font_scale, thickness)
        box_x = center_x - text_width // 2 - 20
        box_y = 140 - text_height - 20
        box_w = text_width + 40
        box_h = text_height + 40
        if int(current_time * 2) % 2:
            cv.rectangle(grid, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 255), 3)
        text_x = center_x - text_width // 2
        text_y = 140 + text_height // 2
        cv.putText(grid, alert_text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)
    # Draw pitch, yaw, roll tracker at the top left
    cv.putText(grid, f"Pitch: {angle_x:.1f}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv.putText(grid, f"Yaw: {angle_y:.1f}", (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv.putText(grid, f"Roll: {roll:.1f}", (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return grid


# Initializing MediaPipe face mesh and camera
if PRINT_DATA:
    print("Initializing the face mesh and camera...")
    if PRINT_DATA:
        head_pose_status = "enabled" if ENABLE_HEAD_POSE else "disabled"
        print(f"Head pose estimation is {head_pose_status}.")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
cam_source = int(args.camSource)
cap = cv.VideoCapture(cam_source)

# Initializing socket for data transmission
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Preparing for CSV logging
csv_data = []
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# Column names for CSV file
column_names = [
    "Timestamp (ms)",
    "Left Eye Center X",
    "Left Eye Center Y",
    "Right Eye Center X",
    "Right Eye Center Y",
    "Left Iris Relative Pos Dx",
    "Left Iris Relative Pos Dy",
    "Right Iris Relative Pos Dx",
    "Right Iris Relative Pos Dy",
    "Total Blink Count",
]
# Add head pose columns if head pose estimation is enabled
if ENABLE_HEAD_POSE:
    column_names.extend(["Pitch", "Yaw", "Roll"])
    
if LOG_ALL_FEATURES:
    column_names.extend(
        [f"Landmark_{i}_X" for i in range(468)]
        + [f"Landmark_{i}_Y" for i in range(468)]
    )

# Main loop for video capture and processing
try:
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)  # Adjust size for smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flipping the frame for a mirror effect
        # I think we better not flip to correspond with real world... need to make sure later...
        #frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        
        # Background tracking for 360-degree camera movement
        if ENABLE_360_TRACKING:
            # Convert to grayscale for optical flow
            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Detect background movement
            (rotation_angle_x, rotation_angle_y), new_points = detect_background_movement(gray_frame, previous_frame, previous_points)
            
            # Update camera rotation angle
            camera_rotation_angle_x += rotation_angle_x
            camera_rotation_angle_y += rotation_angle_y
            
            # Store for next frame
            previous_frame = gray_frame.copy()
            previous_points = new_points
            
            # Display background tracking info on frame
            if PRINT_DATA:
                cv.putText(frame, f"Camera Rotation X: {camera_rotation_angle_x:.1f}°, Y: {camera_rotation_angle_y:.1f}°", 
                          (10, img_h - 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv.putText(frame, f"Frame Rotation X: {rotation_angle_x:.1f}°, Y: {rotation_angle_y:.1f}°", 
                          (10, img_h - 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            # Estimate distance to person based on face width
            current_distance = estimate_distance_from_face(mesh_points, img_w)
            
            # Update real-world coordinate scaling based on estimated distance
            update_real_world_scale(current_distance)

            # Get the 3D landmarks from facemesh x, y and z(z is distance from 0 points)
            # just normalize values
            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )
            # getting the head pose estimation 3d points
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [img_w, img_h, 1]
            )
            head_pose_points_2D = mesh_points[_indices_pose]

            # collect nose three dimension and two dimension points
            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]

            # create the camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            head_pose_points_3D = head_pose_points_3D.astype(np.float64)
            head_pose_points_2D = head_pose_points_2D.astype(np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv.solvePnP(
                head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
            )
            # Get rotational matrix
            rotation_matrix, jac = cv.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

            # Get the y rotation degree
            angle_x = angles[0] * 360
            angle_y = angles[1] * 360
            z = angles[2] * 360

            # if angle cross the values then
            threshold_angle = 10
            # See where the user's head tilting
            if angle_y < -threshold_angle:
                face_looks = "Left"
            elif angle_y > threshold_angle:
                face_looks = "Right"
            elif angle_x < -threshold_angle:
                face_looks = "Down"
            elif angle_x > threshold_angle:
                face_looks = "Up"
            else:
                face_looks = "Forward"
            if SHOW_ON_SCREEN_DATA:
                cv.putText(
                    frame,
                    f"Face Looking at {face_looks}",
                    (img_w - 400, 80),
                    cv.FONT_HERSHEY_TRIPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )
            # Display the nose direction
            nose_3d_projection, jacobian = cv.projectPoints(
                nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = nose_2D_point
            p2 = (
                int(nose_2D_point[0] + angle_y * 10),
                int(nose_2D_point[1] - angle_x * 10),
            )

            cv.line(frame, p1, p2, (255, 0, 255), 3)
            # getting the blinking ratio
            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            # print(f"Blinking ratio : {ratio}")
            # checking if ear less then or equal to required threshold if yes then
            # count the number of frame frame while eyes are closed.
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
            # else check if eyes are closed is greater EYE_AR_CONSEC_FRAMES frame then
            # count the this as a blink
            # make frame counter equal to zero

            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0
            
            # Display all facial landmarks if enabled
            if SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            # Process and display eye features
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Highlighting the irises and corners of the eyes
            cv.circle(
                frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Left iris
            cv.circle(
                frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA
            )  # Right iris
            cv.circle(
                frame, mesh_points[LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Left eye right corner
            cv.circle(
                frame, mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Left eye left corner
            cv.circle(
                frame, mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA
            )  # Right eye right corner
            cv.circle(
                frame, mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA
            )  # Right eye left corner

            # Calculating relative positions
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)

            # Printing data if enabled
            if PRINT_DATA:
                print(f"Total Blinks: {TOTAL_BLINKS}")
                print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}\n")
                # Check if head pose estimation is enabled
                if ENABLE_HEAD_POSE:
                    pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                    angle_buffer.add([pitch, yaw, roll])
                    pitch, yaw, roll = angle_buffer.get_average()

                    # Set initial angles on first successful estimation or recalibrate
                    if initial_pitch is None or (key == ord('c') and calibrated):
                        initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                        calibrated = True
                        if PRINT_DATA:
                            print("Head pose recalibrated.")

                    # Adjust angles based on initial calibration
                    if calibrated:
                        pitch -= initial_pitch
                        yaw -= initial_yaw
                        roll -= initial_roll
                    
                    
                    if PRINT_DATA:
                        print(f"Head Pose Angles: Pitch={pitch}, Yaw={yaw}, Roll={roll}")
            # Logging data
            if LOG_DATA:
                timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
                log_entry = [
                    timestamp,
                    l_cx,
                    l_cy,
                    r_cx,
                    r_cy,
                    l_dx,
                    l_dy,
                    r_dx,
                    r_dy,
                    TOTAL_BLINKS,
                ]  # Include blink count in CSV
                log_entry = [timestamp, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy, TOTAL_BLINKS]  # Include blink count in CSV
                
                # Append head pose data if enabled
                if ENABLE_HEAD_POSE:
                    log_entry.extend([pitch, yaw, roll])
                csv_data.append(log_entry)
                if LOG_ALL_FEATURES:
                    log_entry.extend([p for point in mesh_points for p in point])
                csv_data.append(log_entry)

            # Sending data through socket
            timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
            # Create a packet with mixed types (int64 for timestamp and int32 for the rest)
            packet = np.array([timestamp], dtype=np.int64).tobytes() + np.array([l_cx, l_cy, l_dx, l_dy], dtype=np.int32).tobytes()

            SERVER_ADDRESS = ("127.0.0.1", 7070)
            iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            iris_socket.sendto(packet, SERVER_ADDRESS)

            print(f'Sent UDP packet to {SERVER_ADDRESS}: {packet}')


        # Writing the on screen data on the frame
            if SHOW_ON_SCREEN_DATA:
                if IS_RECORDING:
                    cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)  # Red circle at the top-left corner
                cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                if ENABLE_HEAD_POSE:
                    cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                    cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)


        
        # Displaying the processed frame
        cv.imshow("Eye Tracking", frame)
        
        # Display grid visualization if enabled and face detected
        if SHOW_GRID and results.multi_face_landmarks:
            grid_frame = update_grid_with_gaze_and_pan(angle_x, angle_y, current_distance, camera_rotation_angle_x, camera_rotation_angle_y)
            cv.imshow("Gaze Grid", grid_frame)
        
        # Handle key presses
        key = cv.waitKey(1) & 0xFF

        # Calibrate on 'c' key press
        if key == ord('c'):
            initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
            if PRINT_DATA:
                print("Head pose recalibrated.")
                
        # Reset camera rotation on 'z' key press
        if key == ord('z'):
            camera_rotation_angle_x = 0.0
            camera_rotation_angle_y = 0.0
            camera_rotation_history_x.clear()
            camera_rotation_history_y.clear()
            if PRINT_DATA:
                print("Camera rotation reset to 0°.")
                
        # Toggle 360-degree tracking on 't' key press
        if key == ord('t'):
            ENABLE_360_TRACKING = not ENABLE_360_TRACKING
            if PRINT_DATA:
                status = "enabled" if ENABLE_360_TRACKING else "disabled"
                print(f"360-degree tracking {status}.")
                
        # Inside the main loop, handle the 'r' key press
        if key == ord('r'):
            
            IS_RECORDING = not IS_RECORDING
            if IS_RECORDING:
                print("Recording started.")
            else:
                print("Recording paused.")


        # Exit on 'q' key press
        if key == ord('q'):
            if PRINT_DATA:
                print("Exiting program...")
            break
        
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Releasing camera and closing windows
    cap.release()
    cv.destroyAllWindows()
    iris_socket.close()
    if PRINT_DATA:
        print("Program exited successfully.")

    # Writing data to CSV file
    if LOG_DATA and IS_RECORDING:
        if PRINT_DATA:
            print("Writing data to CSV...")
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(
            LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv"
        )
        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)  # Writing column names
            writer.writerows(csv_data)  # Writing data rows
        if PRINT_DATA:
            print(f"Data written to {csv_file_name}")

  