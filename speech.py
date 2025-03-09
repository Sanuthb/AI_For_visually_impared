import speech_recognition as sr
import pyttsx3  

class Speech:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()

        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id) 
        self.engine.setProperty('rate', 150)  

    def recognize_speech_from_mic(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = self.recognizer.listen(source)

        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return "Error with speech recognition service"

    def text_speech(self, text):
        print(f"Speaking: {text}")  
        self.engine.say(text)
        self.engine.runAndWait()

def speech_to_text():
    return Speech()
