import cv2
import numpy as np

def get_brightness():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    if not ret:
        cam.release()
        return "unknown"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    cam.release()  

    if brightness < 50:
        return "dark"
    elif brightness < 150:
        return "moderate"
    else:
        return "bright"
