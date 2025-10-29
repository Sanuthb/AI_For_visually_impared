import cv2
import pyttsx3
import speech_recognition as sr
import easyocr
import os
from gemini import fetch_sentence  # Gemini AI for sentence reconstruction

class ImageReader:
    def __init__(self):
        # Initialize EasyOCR Reader
        self.reader = easyocr.Reader(['en'], verbose=False)

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

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
        try:
            summarized_text = fetch_sentence(text)  # Gemini generates a meaningful sentence
            return summarized_text
        except Exception as e:
            print(f"Gemini API error: {e}")
            return text  # Return original text if Gemini fails


def read_text_from_frame(frame_data, speech_engine=None):
    """
    Function to be called from main.py - uses provided frame instead of camera.
    CRITICAL: NO camera access - uses provided frame_data only.
    """
    if frame_data is None:
        if speech_engine:
            speech_engine.text_speech("No frame data available for text reading.")
        print("Error: No frame data provided")
        return
    
    reader = ImageReader()
    try:
        # Save the provided frame
        image_path = reader.save_image(frame_data, "temp_read_image.jpg")
        
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
            if speech_engine:
                speech_engine.text_speech(summarized_text)
            else:
                reader.read_and_speak(summarized_text)
        else:
            print("No text detected in the image.")
            message = "No text detected in the image."
            if speech_engine:
                speech_engine.text_speech(message)
            else:
                reader.read_and_speak(message)

    except Exception as e:
        print(f"Error: {str(e)}")
        error_msg = "Sorry, there was an error processing the image."
        if speech_engine:
            speech_engine.text_speech(error_msg)
        else:
            reader.read_and_speak(error_msg)


# Legacy function for backward compatibility (DO NOT USE - opens camera)
def read_text_from_camera():
    """
    DEPRECATED: This function opens a camera which conflicts with monitoring thread.
    Use read_text_from_frame() instead.
    """
    print("WARNING: read_text_from_camera() is deprecated. Use read_text_from_frame() instead.")
    reader = ImageReader()
    try:
        # This will fail if monitoring thread is using the camera
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("Could not open camera")
        
        ret, frame = camera.read()
        camera.release()
        
        if not ret:
            raise Exception("Failed to capture image")
            
        # Use the new frame-based function
        read_text_from_frame(frame, None)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        reader.read_and_speak("Sorry, there was an error accessing the camera.")


if __name__ == "__main__":
    print("Testing text reading from camera (standalone mode)")
    read_text_from_camera()