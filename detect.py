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
    
    if frame_data is None:
        engine.text_speech("Couldn't capture image. Camera feed is down.")
        return
    
    frame = frame_data 

    # Object Detection using YOLOv8 - CRITICAL: Use verbose=False and don't display
    results = model(frame, verbose=False)  # Run YOLOv8 inference without verbose output
    detected_objects = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]  # Get class name
            detected_objects.append(label)

    # Remove duplicates
    unique_objects = set(detected_objects)

    # Brightness Analysis - Use the frame_data directly
    brightness_str = functions.get_brightness(frame_data)
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

def detect_text(engine, frame_data):
    """
    Receives image data and attempts to read text using OCR.
    CRITICAL: NO camera access - uses provided frame_data only.
    """
    
    if frame_data is None:
        engine.text_speech("Couldn't capture image. Camera feed is not ready.")
        return

    # Save the frame temporarily for OCR processing
    temp_image_path = "temp_ocr_image.jpg"
    try:
        cv2.imwrite(temp_image_path, frame_data)
        
        # Import OCR functionality
        try:
            import easyocr
            reader = easyocr.Reader(['en'], verbose=False)
            results = reader.readtext(temp_image_path, detail=0)
            
            if results:
                extracted_text = " ".join(results)
                engine.text_speech(f"I detected the following text: {extracted_text}")
            else:
                engine.text_speech("No text detected in the image.")
                
        except ImportError:
            engine.text_speech("Text detection library is not installed. Please install EasyOCR.")
        except Exception as e:
            print(f"OCR Error: {e}")
            engine.text_speech("There was an error reading the text.")
            
    except Exception as e:
        print(f"Error saving frame: {e}")
        engine.text_speech("Could not process the image for text detection.")