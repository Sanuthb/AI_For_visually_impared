import google.generativeai as genai

genai.configure(api_key="AIzaSyA0evdRguiOGlIQ0JgIzwrx-Bb0JVtcjLY")
model = genai.GenerativeModel("gemini-2.0-flash")

def fetch_description(query):
    """Fetch brief description from Gemini AI"""
    response = model.generate_content(f"Provide a brief description of {query}. Keep it short and concise.")
    return response.text if response else "No description available."

def fetch_additional_info(query):
    """Fetch additional information from Gemini AI"""
    response = model.generate_content(f"Provide additional important details about {query}, but keep it very short.")
    return response.text if response else "No additional information available."


