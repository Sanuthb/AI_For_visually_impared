import speech
import detect
import datetime
import functions
import gemini 
from read import read_text_from_camera
from ultralytics import YOLO 

# --- IMPORTS ---
import cv2
import threading
import time
import subprocess
import requests
import os
import webbrowser
# ----------------

# =======================================================
# === GLOBAL STATE FOR ANTI-CHATTERING (YOLO ONLY) ===
# =======================================================
# Tracks the last spoken YOLO warning to prevent repetition (e.g., "person ahead" sticking).
LAST_ANNOUNCED_WARNING = "" 

# =======================================================
# === 1. HELPER FUNCTIONS FOR PROACTIVE MONITORING (Vision Logic) ===
# =======================================================

def get_object_direction(frame_width, x_min, x_max):
    """Calculates the object's horizontal direction relative to the frame."""
    
    # Define directional boundaries
    LEFT_BOUNDARY = frame_width * 0.35 
    RIGHT_BOUNDARY = frame_width * 0.65
    
    box_center_x = (x_min + x_max) / 2
    
    if box_center_x < LEFT_BOUNDARY:
        return "to your left"
    elif box_center_x > RIGHT_BOUNDARY:
        return "to your right"
    else:
        return "directly ahead"

def check_proximity_based_on_area(frame_height, y_max, box_height):
    """Estimates proximity based on box size and vertical position."""
    frame_bottom_proximity = y_max / frame_height
    
    # Heuristic: Object is near the bottom AND is large (critical)
    if frame_bottom_proximity > 0.6 and box_height > (frame_height * 0.3):
        return "CRITICAL"
    return "SAFE"

# =======================================================
# === 2. REAL-TIME MONITORING LOOP (YOLO - Fast) ===
# =======================================================

def real_time_monitoring_thread(yolo_model, speech_engine):
    """Handles continuous video capture and critical obstacle detection."""
    global LAST_ANNOUNCED_WARNING 

    OBSTACLE_CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'dog', 'cat', 'chair', 'traffic light', 'stop sign', 'bench']
    
    announcement_cooldown = 0
    COOLDOWN_LIMIT = 25 # Cooldown in frames/cycles (approx 1-2 seconds)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream/camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        
        # Perform detection
        results = yolo_model(frame, verbose=False)
        warnings = []
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                x_min, y_min, x_max, y_max = [int(val) for val in box.xyxy[0].tolist()]

                # Proactive Obstacle Check
                if label in OBSTACLE_CLASSES:
                    box_height = y_max - y_min
                    proximity = check_proximity_based_on_area(frame_height, y_max, box_height)

                    if proximity == "CRITICAL":
                        direction = get_object_direction(frame_width, x_min, x_max)
                        warnings.append(f"Warning! {label} {direction}")

                # Optional: Draw boxes for visualization/debugging
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        
        current_critical_warning = warnings[0] if warnings else ""

        # 2. TTS Announcement Logic (Rate Limited & State Tracked)
        if announcement_cooldown == 0 and current_critical_warning:
            
            # Only announce if the warning text has changed (Anti-chattering)
            if current_critical_warning != LAST_ANNOUNCED_WARNING:
                
                speech_engine.text_speech(current_critical_warning) 
                
                # Update the state and the cooldown
                LAST_ANNOUNCED_WARNING = current_critical_warning
                announcement_cooldown = COOLDOWN_LIMIT
            
            # If the warning is the SAME, reset the cooldown to wait for next announcement cycle
            elif current_critical_warning == LAST_ANNOUNCED_WARNING:
                 announcement_cooldown = COOLDOWN_LIMIT
                 
        elif announcement_cooldown > 0:
            announcement_cooldown -= 1
            
        # Clear the warning state if no obstacle is detected and the cooldown is active
        if not warnings and announcement_cooldown == 0:
             LAST_ANNOUNCED_WARNING = ""
        
        # Display the frame
        cv2.imshow('Proactive Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time monitoring stopped.")

# =======================================================
# === 3. MAIN EXECUTION (Voice Command Loop) ===
# =======================================================

if __name__ == '__main__':
    
    # Start the Flask server in the background
    try:
        # NOTE: Changing 'python3' to 'python' often fixes Windows subprocess issues
        subprocess.Popen(["python", "Navigation.py"]) 
        print("Navigation server started in background.")
    except Exception as e:
        print(f"Failed to start Navigation.py: {e}")
    
    # Initialization
    model = YOLO("yolov8s.pt")  
    project_id = "blindbot-4f356"
    engine = speech.speech_to_text()

    listening = False
    
    # 1. Start the Real-Time Monitoring Thread (YOLO)
    monitor_thread = threading.Thread(target=real_time_monitoring_thread, args=(model, engine,))
    monitor_thread.daemon = True 
    monitor_thread.start()
    print("Real-time YOLO monitoring started.")
    
    # NOTE: The slow, repetitive Gemini monitoring thread has been intentionally removed.
    # The user must use the 'Describe' command for detailed analysis.

    while True:
        if not listening:
            resp = engine.recognize_speech_from_mic()
            print(f"You said: {resp}")

            if resp and "alexa" in resp.lower():
                engine.text_speech("Hi my name is alexa. How can I assist you?")
                listening = True

        else:
            resp = engine.recognize_speech_from_mic()
            intent, text = None, None 

            if resp:
                print(f"User said: {resp}") 
                
                try:
                    intent, text = detect.detect_intent_texts(project_id, 0, [resp], 'en')
                    print(f"Detected Intent: {intent}, Detected Text: {text}") 
                except Exception as e:
                    print(f"Error detecting intent: {e}")
                    engine.text_speech("I could not process that, please try again.")
                    continue 

                if intent: 
                    if intent == "Describe":
                        # This command provides the detailed, non-repetitive Gemini analysis.
                        engine.text_speech("Describing scene")
                        detect.describe_scene(model, engine)
                    elif intent == "Brightness":
                        brightness = functions.get_brightness()
                        engine.text_speech(f"It is {brightness} outside")
                    elif intent == "Read":
                        engine.text_speech("I will capture an image and read any text I find.")
                        read_text_from_camera()
                    elif intent == "Time":
                        currentDT = datetime.datetime.now()
                        engine.text_speech(f"The time is {currentDT.hour} hours and {currentDT.minute} minutes")
                    elif intent == "GeneralQuery":
                        description = gemini.fetch_description(text)
                        engine.text_speech(description)

                        engine.text_speech("Would you like more details?")
                        follow_up = engine.recognize_speech_from_mic()

                        if follow_up and "yes" in follow_up.lower():
                            additional_info = gemini.ask_gemini(text)
                            engine.text_speech(additional_info)
                    elif intent == "Navigate":
                        
                        engine.text_speech("Where would you like to go?")
                        destination_resp = engine.recognize_speech_from_mic()

                        if destination_resp:
                            engine.text_speech(f"Searching location for {destination_resp}")
                            try:
                                # 1. Geocode Destination
                                dest_url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(destination_resp)}"
                                headers = {"User-Agent": "blindbot/1.0"}
                                resdes = requests.get(dest_url, headers=headers)
                                dest_data = []

                                if resdes.status_code == 200 and resdes.text.strip():
                                    dest_data = resdes.json()
                                else:
                                    engine.text_speech("Could not fetch the location from the internet.")

                                if len(dest_data) > 0:
                                    dest_location = dest_data[0]
                                    dest_lat = float(dest_location["lat"])
                                    dest_lon = float(dest_location["lon"])

                                    # 2. Start Navigation on Flask Server
                                    engine.text_speech("Destination found. Starting navigation.")
                                    time.sleep(1) 
                                    response = requests.post("http://localhost:5002/receive_location", json={
                                        "dest_latitude": dest_lat,
                                        "dest_longitude": dest_lon
                                    })

                                    if response.status_code == 200:
                                        response_data = response.json()
                                        first_step = response_data.get("current_step", {})
                                        initial_instruction = first_step.get("direction", "Navigation started. Listen for turn-by-turn guidance.")
                                        engine.text_speech(initial_instruction)
                                    else:
                                        engine.text_speech("Failed to start navigation on the server.")
                                else:
                                    engine.text_speech("Could not find the destination location you mentioned.")
                            except Exception as e:
                                print("Geolocation error:", e)
                                engine.text_speech("Something went wrong while fetching the coordinates.")
                        else:
                            engine.text_speech("I didn't catch the destination.")
                else:
                    engine.text_speech("Sorry, I could not understand.")