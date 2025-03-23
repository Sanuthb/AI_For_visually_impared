import cv2
import pyttsx3
import speech_recognition as sr
import easyocr
import os
from gemini import fetch_description  # Gemini AI for summarization

class ImageReader:
    def __init__(self):
        # Initialize EasyOCR Reader
        self.reader = easyocr.Reader(['en'])

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

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

    def extract_text(self, image_path):
        """Extract text from the image using EasyOCR"""
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return None

        results = self.reader.readtext(image_path, detail=0)  # Extract text only
        if results:
            extracted_text = " ".join(results)  # Convert tokens into a sentence
            return extracted_text.strip()
        return None

    def read_and_speak(self, text):
        """Convert text to speech"""
        if text:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error during text-to-speech conversion: {str(e)}")

    def get_gemini_summary(self, text):
        """Fetch meaningful sentence from Gemini AI"""
        summarized_text = fetch_description(text)  # Gemini generates a 1-2 line summary
        return summarized_text

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
        # Extract raw text from the image
        raw_text = reader.extract_text(image_path)
        
        if raw_text:
            print("Raw Extracted Text:")
            print(raw_text)

            # Get a meaningful sentence using Gemini AI
            summarized_text = reader.get_gemini_summary(raw_text)
            print("\nFinal Sentence (Gemini AI):")
            print(summarized_text)

            # Speak the final sentence
            reader.read_and_speak(summarized_text)
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
