from paddleocr import PaddleOCR
from transformers import pipeline

# Initialize OCR model
ocr = PaddleOCR(lang='en')

# Function to extract text from image
def extract_text_from_image(image_path):
    result = ocr.ocr(image_path)
    extracted_text = " ".join([res[1][0] for res in result[0]])  # Extract detected text
    return extracted_text

# Function to summarize text
def summarize_text(text, max_length=100):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Main function
def analyze_image(image_path):
    extracted_text = extract_text_from_image(image_path)
    print("\nExtracted Text:\n", extracted_text)

    if len(extracted_text.strip()) > 50:  # Summarize only if text is long enough
        summary = summarize_text(extracted_text)
        print("\nSummarized Text:\n", summary)
    else:
        print("\nSummary not required (text too short).")

# Example usage
image_path = "sample.jpg"  # Replace with your image path
analyze_image(image_path)


