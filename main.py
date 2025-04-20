import speech
import detect
import datetime
import functions
import gemini  
from read import read_text_from_camera
from ultralytics import YOLO  

import subprocess
subprocess.Popen(["python3", "Navigation.py"])

# Load YOLO model
model = YOLO("yolov8s.pt")  

project_id = "blindbot-4f356"
engine = speech.speech_to_text()

listening = False

while True:
    if not listening:
        resp = engine.recognize_speech_from_mic()
        print(f"You said: {resp}")

        if resp and "alexa" in resp.lower():
            engine.text_speech("Hi my name is Alexa. How can I assist you?")
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
                import webbrowser
                import time
                import requests
                import os

                engine.text_speech("Where would you like to go?")
                destination_resp = engine.recognize_speech_from_mic()

                if destination_resp:
                    engine.text_speech(f"Searching location for {destination_resp}")
                    try:
                        dest_url = f"https://nominatim.openstreetmap.org/search?format=json&q={requests.utils.quote(destination_resp)}"
                        headers = {"User-Agent": "blindbot/1.0"}
                        dest_data=0

                        resdes = requests.get(dest_url, headers=headers)
                        if resdes.status_code == 200 and resdes.text.strip():
                                    dest_data = resdes.json()
                        else:
                                    print("Failed to fetch location data.")
                                    print("Status code:", response.status_code)
                                    print("Response text:", response.text)
                                    engine.text_speech("Could not fetch the location from the internet.")

                        if len(dest_data) > 0:
                                    dest_location = dest_data[0]
                                    dest_lat = float(dest_location["lat"])
                                    dest_lon = float(dest_location["lon"])

                                    engine.text_speech("Opening map and fetching directions.")
                                    webbrowser.open("file://" + os.path.realpath("index.html"))
                                    time.sleep(5)
                                    response = requests.post("http://localhost:5002/receive_location", json={
                                        "dest_latitude": dest_lat,
                                        "dest_longitude": dest_lon
                                    })

                                    if response.status_code == 200:
                                        summary = response.json().get("summary", "")
                                        engine.text_speech(summary)
                                    else:
                                        engine.text_speech("Failed to retrieve navigation summary.")
                        else:
                                    engine.text_speech("Could not find the destination location you mentioned.")
                    except Exception as e:
                        print("Geolocation error:", e)
                        engine.text_speech("Something went wrong while fetching the coordinates.")
                else:
                    engine.text_speech("I didn't catch the destination.")
        else:
            engine.text_speech("Sorry, I could not understand.")
