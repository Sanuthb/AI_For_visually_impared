import cv2
import pyttsx3
import speech_recognition as sr
from paddleocr import PaddleOCR
import os
import numpy as np
from gemini_api import fetch_description, fetch_additional_info  # Assuming Gemini API integration

class ImageReader:
    def __init__(self):
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        # Set speech rate and volume
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)
        # Initialize camera
        self.camera = None

    def start_camera(self):
        """Initialize the camera"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise Exception("Could not open camera")

    def capture_image(self):
        """Capture an image from the camera"""
        if self.camera is None:
            self.start_camera()
        
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to capture image")
        return frame

    def save_image(self, image, filename="captured_image.jpg"):
        """Save the captured image"""
        cv2.imwrite(filename, image)
        return filename

    def read_image(self, image_path):
        """
        Read text from an image and convert it to speech
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return None

        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image '{image_path}'.")
            return None

        # Perform OCR
        result = self.ocr.ocr(img, cls=True)
        
        if not result or not result[0]:
            print("No text detected in the image.")
            return None

        # Extract text from OCR result
        text = ""
        for line in result[0]:
            text += line[1][0] + " "

        return text.strip()

    def read_and_speak(self, text):
        """Convert text to speech"""
        if text:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error during text-to-speech conversion: {str(e)}")

    def get_voice_input(self):
        """Capture user's voice input and convert it to text"""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening for a response (yes or no)...")
            self.read_and_speak("Please say yes or no.")
            
            try:
                recognizer.adjust_for_ambient_noise(source)  # Adjust for background noise
                audio = recognizer.listen(source, timeout=5)  # Capture audio with timeout
                user_response = recognizer.recognize_google(audio).lower()
                print(f"Recognized voice input: {user_response}")
                return user_response
            except sr.UnknownValueError:
                print("Could not understand the audio.")
                return None
            except sr.RequestError:
                print("Error connecting to speech recognition service.")
                return None

    def get_gemini_info(self, text):
        """Fetch information from Gemini API and ask for additional info"""
        description = fetch_description(text)
        self.read_and_speak(description)
        
        self.read_and_speak("Do you want more details about it?")
        user_response = self.get_voice_input()
        
        if user_response in ["yes", "y"]:
            additional_info = fetch_additional_info(text)
            self.read_and_speak(additional_info)
        elif user_response in ["no", "n"]:
            self.read_and_speak("Okay, no additional details provided.")
        else:
            self.read_and_speak("Sorry, I didn't understand your response.")

    def cleanup(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


def read_text_from_camera():
    """Function to be called from main.py"""
    reader = ImageReader()
    try:
        # Capture image
        frame = reader.capture_image()
        # Save the captured image
        image_path = reader.save_image(frame)
        # Read text from the image
        text = reader.read_image(image_path)
        if text:
            print("Detected text:")
            print(text)
            reader.read_and_speak(text)
            reader.get_gemini_info(text)
        else:
            print("No text detected in the image.")
            reader.read_and_speak("No text detected in the image.")
    except Exception as e:
        print(f"Error: {str(e)}")
        reader.read_and_speak("Sorry, there was an error processing the image.")
    finally:
        reader.cleanup()


if __name__ == "__main__":
    read_text_from_camera()
