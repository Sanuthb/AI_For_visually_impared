import google.generativeai as genai

genai.configure(api_key="AIzaSyCxAqDEbJp-BoBdXCT9SW43K8loaSpxi_0")
model = genai.GenerativeModel("gemini-2.0-flash")

def fetch_description(query):
    """Fetch brief description from Gemini AI"""
    response = model.generate_content(f"Provide a brief description of {query}. Keep it short and concise.")
    return response.text if response else "No description available."

def fetch_additional_info(query):
    """Fetch additional information from Gemini AI"""
    response = model.generate_content(f"Provide additional important details about {query}, but keep it very short.")
    return response.text if response else "No additional information available."

def ask_gemini(query):
    """Fetch a human-like response from Gemini AI"""
    try:
        response = model.generate_content(query)
        return response.text if response else "I'm not sure about that."
    except Exception as e:
        return f"Error fetching response: {e}"

def fetch_sentence(ocr_text):
    """Convert extracted OCR text into a meaningful sentence while keeping the original intent."""
    response = model.generate_content(
        f"Take the following OCR-extracted text and reconstruct it into a proper, meaningful sentence while maintaining its original intent: {ocr_text}"
    )
    return response.text.strip() if response and response.text else "Could not generate a meaningful sentence."
