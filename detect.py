import cv2
import functions  

def detect_intent_texts(project_id, session_id, texts, language_code):
    """
    Detects user intent from spoken text.
    """
    text = texts[0].lower()

    if any(keyword in text for keyword in ["time", "clock", "hour"]):
        return "Time", "Fetching the current time."
    elif any(keyword in text for keyword in ["describe", "surroundings", "what do you see"]):
        return "Describe", "Describing the scene."
    elif any(keyword in text for keyword in ["brightness", "light level", "dark"]):
        return "Brightness", "Checking the brightness level."
    elif any(keyword in text for keyword in ["read", "text", "document"]):
        return "Read", "Reading detected text."
    else:
        return "GeneralQuery", text  

def describe_scene(model, engine):
    """
    Captures an image, detects objects using YOLOv8, and describes the surroundings.
    """
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()  

    if not ret:
        engine.text_speech("Couldn't capture image.")
        return

    # Object Detection using YOLOv8
    results = model(frame)  # Run YOLOv8 inference
    detected_objects = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]  # Get class name
            detected_objects.append(label)

    # Remove duplicates
    unique_objects = set(detected_objects)

    # Brightness Analysis
    brightness_str = functions.get_brightness()  
    brightness_levels = {
        "dark": "dimly-lit",
        "moderate": "moderately-lit",
        "bright": "well-lit"
    }
    light_condition = brightness_levels.get(brightness_str.lower(), "unknown lighting")

    # Safety Analysis
    unsafe_objects = {"fire", "knife", "hole", "broken glass"}
    is_safe = not any(obj in unsafe_objects for obj in unique_objects)

    # Generating Description
    if unique_objects:
        object_list = ", ".join(unique_objects)
        response = f"I see {object_list}. The room is {light_condition}. "
        response += "It seems safe to enter." if is_safe else "Be careful, I detect some hazards."
    else:
        response = "I don't see anything recognizable."

    engine.text_speech(response)

def detect_text(engine):
    """
    Captures an image and attempts to read text.
    """
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if not ret:
        engine.text_speech("Couldn't capture image.")
        return

    # Placeholder for OCR Implementation (Tesseract or Google Vision)
    engine.text_speech("Text detection is not implemented yet. Consider using Tesseract OCR.")


