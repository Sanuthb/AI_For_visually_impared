import cv2
import numpy as np

def get_brightness(frame_data=None):
    """
    Determine brightness level from provided frame data.
    If no frame is provided, attempts to capture from camera (fallback).
    """
    # Use provided frame if available (preferred method)
    if frame_data is not None:
        frame = frame_data
    else:
        # Fallback: Try to capture from camera
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Error: Could not open camera.")
            return "unknown"

        ret, frame = cam.read()
        cam.release()  # Ensure camera is released

        if not ret:
            print("Error: Could not capture frame.")
            return "unknown"

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate mean brightness
    brightness = np.mean(gray)
    print(f"Brightness Value: {brightness}")  # Debugging info

    # Categorize brightness levels
    if brightness < 50:
        return "dark"
    elif brightness < 150:
        return "moderate"
    else:
        return "bright"