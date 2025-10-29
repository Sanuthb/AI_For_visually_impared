import speech
import detect
import datetime
import functions
import gemini 
from read import read_text_from_camera
from ultralytics import YOLO 
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- IMPORTS ---
import cv2
import numpy as np
import threading
import time
import subprocess
import requests
import os
import webbrowser
# ----------------

# =======================================================
# === GLOBAL STATE AND RESOURCE MANAGEMENT (CRITICAL) ===
# =======================================================
LAST_ANNOUNCED_WARNING = "" 
LATEST_FRAME = None          # Stores the latest frame from the monitoring thread
FRAME_LOCK = threading.Lock() # Lock to ensure thread-safe access to LATEST_FRAME

# =======================================================
# === 1. HELPER FUNCTIONS ===
# =======================================================

def translate_direction(dx, dy, threshold=5):
    """Translates pixel movement into human-readable direction."""
    if abs(dx) < threshold and abs(dy) < threshold:
        return "stationary"
    
    if abs(dx) > abs(dy):
        return "moving right" if dx > 0 else "moving left"
    else:
        return "moving closer" if dy > 0 else "moving farther"

def check_proximity_deep_sort(frame_height, y1, y2):
    """Estimates proximity based on box position and height."""
    box_height = y2 - y1
    frame_bottom_proximity = y2 / frame_height
    
    if frame_bottom_proximity > 0.65 and box_height > (frame_height * 0.3):
        return True
    return False

# --- Camera Access Helper for Intents ---
def capture_latest_frame():
    """Retrieves the latest frame from the monitoring thread's buffer."""
    with FRAME_LOCK:
        global LATEST_FRAME
        if LATEST_FRAME is not None:
            return LATEST_FRAME.copy() 
        return None


# =======================================================
# === 2. REAL-TIME MONITORING LOOP (YOLO + DeepSort) ===
# =======================================================

def real_time_monitoring_thread(yolo_model, speech_engine):
    """Handles continuous video capture, DeepSort tracking, and critical obstacle warnings."""
    global LAST_ANNOUNCED_WARNING, LATEST_FRAME

    # --- Initialization ---
    tracker = DeepSort(max_age=30)
    prev_positions = {}
    OBSTACLE_CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'dog', 'cat', 'chair', 'traffic light', 'stop sign', 'bench']
    
    announcement_cooldown = 0
    COOLDOWN_LIMIT = 25 
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream/camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream failed. Stopping monitoring thread.")
            break

        frame_height, frame_width, _ = frame.shape
        
        # --- UPDATE GLOBAL FRAME BUFFER (CRITICAL STEP) ---
        with FRAME_LOCK:
            LATEST_FRAME = frame.copy() 
        # --------------------------------------------------
        
        # --- YOLO Detection ---
        results = yolo_model(frame, stream=True, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes.xyxy
            confs = r.boxes.conf
            classes = r.boxes.cls
            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box.cpu().numpy()
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), str(yolo_model.names[int(cls)])))
        
        # --- DeepSort Tracking & Warning Logic ---
        tracks = tracker.update_tracks(detections, frame=frame)
        warnings = []

        for track in tracks:
            if not track.is_confirmed(): continue

            track_id = track.track_id
            cls_name = track.get_det_class()
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            if cls_name in OBSTACLE_CLASSES:
                is_critical = check_proximity_deep_sort(frame_height, y1, y2)

                if is_critical:
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                    direction_text = "directly ahead"
                    
                    if track_id in prev_positions:
                        px, py = prev_positions[track_id]
                        dx, dy = cx - px, cy - py
                        direction_text = translate_direction(dx, dy)
                        
                    warnings.append(f"CRITICAL: {cls_name} {direction_text}")
                    prev_positions[track_id] = (cx, cy)
                else:
                    if track_id in prev_positions:
                        del prev_positions[track_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id} {cls_name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        # --- TTS Announcement Logic (Anti-chattering) ---
        current_critical_warning = warnings[0] if warnings else ""

        if announcement_cooldown == 0 and current_critical_warning:
            if current_critical_warning != LAST_ANNOUNCED_WARNING:
                speech_engine.text_speech(current_critical_warning) 
                
                LAST_ANNOUNCED_WARNING = current_critical_warning
                announcement_cooldown = COOLDOWN_LIMIT
            
            elif current_critical_warning == LAST_ANNOUNCED_WARNING:
                 announcement_cooldown = COOLDOWN_LIMIT
                 
        elif announcement_cooldown > 0:
            announcement_cooldown -= 1
            
        if not warnings and announcement_cooldown == 0:
             LAST_ANNOUNCED_WARNING = ""
        
        cv2.imshow('Proactive Monitoring', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Real-time monitoring stopped.")


# =======================================================
# === 3. INTENT EXECUTION THREAD WRAPPER ===
# =======================================================

def execute_intent_async(target_func, *args, **kwargs):
    """Runs a time-consuming user intent in a new thread to prevent blocking the main voice loop."""
    t = threading.Thread(target=target_func, args=args, kwargs=kwargs)
    t.start()


# =======================================================
# === 4. MAIN EXECUTION (Voice Command Loop) ===
# =======================================================

if __name__ == '__main__':
    
    # Start the Flask server in the background
    try:
        subprocess.Popen(["python", "Navigation.py"]) 
        print("Navigation server started in background.")
    except Exception as e:
        print(f"Failed to start Navigation.py: {e}")
    
    # MODEL INITIALIZATION (GLOBAL SCOPE FIX)
    model = YOLO("yolov8n.pt") 
    project_id = "blindbot-4f356"
    engine = speech.speech_to_text()

    listening = False
    
    # 1. Start the Real-Time Monitoring Thread (YOLO + DeepSort)
    monitor_thread = threading.Thread(target=real_time_monitoring_thread, args=(model, engine,))
    monitor_thread.daemon = True 
    monitor_thread.start()
    print("Real-time DeepSort monitoring started.")

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
                    # --- Intents that BLOCK (run in a new thread) ---
                    if intent == "Describe":
                        engine.text_speech("Capturing scene image and describing.")
                        
                        def run_describe_async(yolo_model, speech_engine):
                            frame_to_analyze = capture_latest_frame()
                            
                            if frame_to_analyze is None:
                                speech_engine.text_speech("Error: Camera feed is not ready. Please wait for monitoring to start.")
                                return

                            # NOTE: frame_to_analyze is passed as the third argument now (the fix)
                            detect.describe_scene(yolo_model, speech_engine, frame_to_analyze)
                        
                        execute_intent_async(run_describe_async, model, engine)
                        
                    elif intent == "Read":
                        engine.text_speech("I will capture an image and read any text I find.")
                        
                        # 🐛 FIX: Frame capture logic must be wrapped and passed to detect_text
                        def run_read_async(speech_engine):
                            frame_to_analyze = capture_latest_frame()
                            
                            if frame_to_analyze is None:
                                speech_engine.text_speech("Error: Camera feed is not ready. Please wait for monitoring to start.")
                                return
                            
                            # NOTE: Assuming read.py or detect.py has a function updated to take frame data
                            # Based on previous context, we use detect.detect_text(engine, frame_data)
                            detect.detect_text(speech_engine, frame_to_analyze) 
                        
                        execute_intent_async(run_read_async, engine)
                        
                    elif intent == "GeneralQuery":
                        engine.text_speech("Searching for information.")
                        
                        def run_general_query(query_text):
                            description = gemini.fetch_description(query_text)
                            engine.text_speech(description)
                            engine.text_speech("Would you like more details?")
                            follow_up = engine.recognize_speech_from_mic()
                            if follow_up and "yes" in follow_up.lower():
                                additional_info = gemini.ask_gemini(query_text)
                                engine.text_speech(additional_info)

                        execute_intent_async(run_general_query, text)

                    elif intent == "Navigate":
                        engine.text_speech("Where would you like to go?")
                        destination_resp = engine.recognize_speech_from_mic()

                        def run_navigation(dest_response):
                            if not dest_response:
                                engine.text_speech("I didn't catch the destination.")
                                return

                            engine.text_speech(f"Searching location for {dest_response}")
                            try:
                                # 1. Geocode Destination 
                                dest_url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(dest_response)}"
                                headers = {"User-Agent": "blindbot/1.0"}
                                resdes = requests.get(dest_url, headers=headers)
                                dest_data = resdes.json() if resdes.status_code == 200 and resdes.text.strip() else []

                                if dest_data:
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
                                        first_step = response.json().get("current_step", {})
                                        initial_instruction = first_step.get("direction", "Navigation started. Listen for turn-by-turn guidance.")
                                        engine.text_speech(initial_instruction)
                                    else:
                                        engine.text_speech("Failed to start navigation on the server.")
                                else:
                                    engine.text_speech("Could not find the destination location you mentioned.")
                            except Exception as e:
                                print(f"Geolocation error: {e}")
                                engine.text_speech("Something went wrong while fetching the coordinates.")
                        
                        execute_intent_async(run_navigation, destination_resp)


                    # --- Intents that DO NOT BLOCK (run immediately) ---
                    elif intent == "Brightness":
                        brightness = functions.get_brightness()
                        engine.text_speech(f"It is {brightness} outside")
                    
                    elif intent == "Time":
                        currentDT = datetime.datetime.now()
                        engine.text_speech(f"The time is {currentDT.hour} hours and {currentDT.minute} minutes")

                else:
                    engine.text_speech("Sorry, I could not understand.")