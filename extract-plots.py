import pytesseract
import cv2
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import json
import numpy as np

# Set up Tesseract for OCR (change the path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load LayoutLMv3 model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# Function to extract text using Tesseract OCR
def extract_text_with_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    return text_data

# Function to detect rectangular plot boundaries using OpenCV
def detect_plots(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect contours in the edge map
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plots = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # If contour has 4 points, we assume it's a rectangular plot
            x, y, w, h = cv2.boundingRect(contour)
            plots.append((x, y, w, h))
    return plots

# Function to extract plot details and return JSON
def extract_plot_details(image):
    plots = detect_plots(image)
    text_data = extract_text_with_ocr(image)
    
    plot_details = []
    for plot in plots:
        x, y, w, h = plot
        # Extract the text within the bounding box of the plot
        plot_text = []
        for i, (tx, ty, tw, th) in enumerate(zip(text_data['left'], text_data['top'], text_data['width'], text_data['height'])):
            if x <= tx <= x + w and y <= ty <= y + h:
                plot_text.append(text_data['text'][i])
        
        plot_number = ' '.join(plot_text).strip()
        if plot_number:
            plot_details.append({
                "plot_number": plot_number,
                "coordinates": [
                    {"x": x, "y": y},
                    {"x": x + w, "y": y},
                    {"x": x + w, "y": y + h},
                    {"x": x, "y": y + h}
                ]
            })
    
    return plot_details

# Main function
def main(image_path):
    image = cv2.imread(image_path)
    
    # Extract plot details
    plot_details = extract_plot_details(image)
    
    # Output result as JSON
    result = json.dumps(plot_details, indent=4)
    print(result)
    
    # Optionally, save the result to a file
    with open("plot_details.json", "w") as json_file:
        json_file.write(result)

# Run the extraction on a sample JPG image
if __name__ == "__main__":
    main("real_estate_layout.jpg")
