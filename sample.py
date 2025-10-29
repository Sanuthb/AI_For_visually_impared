import speech_recognition as sr
import pyttsx3 
import threading

# Global Lock to ensure only one thread initializes or uses the engine at a time
TTS_LOCK = threading.Lock() 

class Speech:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # The engine is NOT initialized here to prevent shared state conflict.

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
        """
        Fix: Initializes a new engine for every call, speaks, and cleanly stops it 
        within the lock to achieve thread safety.
        """
        print(f"Speaking: {text}") 
        
        # 1. Acquire the lock to ensure sequential access
        with TTS_LOCK:
            try:
                # 2. Initialize the engine FRESH inside the lock
                engine = pyttsx3.init()
                
                # 3. Set properties
                voices = engine.getProperty('voices')
                if voices:
                    engine.setProperty('voice', voices[0].id) 
                engine.setProperty('rate', 150)
                
                # 4. Speak
                engine.say(text)
                
                # 5. Run and wait (Blocks the current thread until speech is done)
                engine.runAndWait() 
                
                # 6. Stop/Destroy the engine instance (CRUCIAL STEP)
                engine.stop()
                
            except Exception as e:
                print(f"FATAL TTS ERROR: {e}") 

def speech_to_text():
    return Speech()