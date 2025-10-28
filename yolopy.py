from ultralytics import YOLO
import cv2
import pyttsx3 # Assuming pyttsx3 is available for a client-side announcement helper

# --- Client-Side TTS Helper (Similar to your Flask server's announce) ---
# NOTE: The client app should generally handle its own TTS, independent of the Flask server.
def announce_client_side(text):
    """Helper function for local, high-priority announcements (like warnings)."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        # Using runAndWait can block the video thread; for real-time,
        # consider running TTS in a separate thread.
        engine.runAndWait() 
    except Exception as e:
        print(f"Client-side TTS error: {e}")

class YOLOv8:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Load YOLOv8 model

    def detect_objects(self, frame):
        """
        Performs inference and returns a list of detected objects 
        including bounding box coordinates.
        """
        # Set verbose=False to suppress detection output, keeping it clean for real-time
        results = self.model(frame, verbose=False)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.model.names[cls_id]  # Get class name
                
                # Extract integer bounding box coordinates (xyxy format)
                x_min, y_min, x_max, y_max = [int(val) for val in box.xyxy[0].tolist()]
                
                detected_objects.append({
                    "label": label, 
                    "confidence": confidence,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                })

        return detected_objects