import speech
import detect
import datetime
import functions
import gemini  # Import Gemini AI functions
from read import read_text_from_camera
from ultralytics import YOLO  

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
            else:
                engine.text_speech(f"I detected: {intent}. Response: {text}")
        else:
            engine.text_speech("Sorry, I could not understand.")
