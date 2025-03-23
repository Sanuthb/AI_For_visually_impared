from ultralytics import YOLO
import cv2

class YOLOv8:
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # Load YOLOv8 model

    def detect_objects(self, frame):
        results = self.model(frame)  # Perform inference
        detected_objects = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = self.model.names[cls_id]  # Get class name
                detected_objects.append((label, confidence))

        return detected_objects
