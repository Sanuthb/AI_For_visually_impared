import cv2
import numpy as np

def get_brightness():
    """Capture an image from the webcam and determine brightness level."""
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



