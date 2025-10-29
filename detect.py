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
    elif any(keyword in text for keyword in ["navigate",'location',"destination","route"]):
        return "Navigate","Finding the routes"
    else:
        return "GeneralQuery", text 

def describe_scene(model, engine, frame_data):
    """
    Receives image data from the monitoring thread, detects objects using YOLOv8, 
    and describes the surroundings.
    """
    
    # --- FIX: Removed all cv2.VideoCapture() logic ---
    if frame_data is None:
        engine.text_speech("Couldn't capture image. Camera feed is down.")
        return
    # frame variable now holds the data passed from main.py
    frame = frame_data 

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
    # NOTE: Since frame_data is available, you could calculate brightness from the frame here 
    # instead of using a separate function if 'functions.get_brightness()' doesn't need a live camera.
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
        response = f"I see {object_list}. The environment is {light_condition}. "
        response += "It seems safe to proceed." if is_safe else "Be careful, I detect some hazards."
    else:
        response = f"I don't see anything recognizable. The lighting is {light_condition}."

    engine.text_speech(response)

def detect_text(engine, frame_data): # <--- UPDATED SIGNATURE
    """
    Receives image data and attempts to read text.
    """
    
    # --- FIX: Removed all cv2.VideoCapture() logic ---
    if frame_data is None:
        engine.text_speech("Couldn't capture image. Camera feed is not ready.")
        return

    # Placeholder for OCR Implementation (Tesseract or Google Vision)
    engine.text_speech("Text detection is not implemented yet. Please integrate Tesseract OCR using the provided frame data.")

