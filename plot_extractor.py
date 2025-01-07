# plot_extractor.py
import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Any

class PlotExtractor:
    def __init__(self):
        # Configure pytesseract path if needed
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        denoised = cv2.medianBlur(binary, 3)
        return denoised

    def detect_plots(self, image: np.ndarray) -> List[Dict[str, Any]]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plots = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 1000:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            roi = gray[y:y+h, x:x+w]
            text = pytesseract.image_to_string(roi)
            
            plot_info = self.parse_plot_info(text, x, y, w, h)
            if plot_info:
                plots.append(plot_info)
        
        return plots

    def parse_plot_info(self, text: str, x: int, y: int, w: int, h: int) -> Dict[str, Any]:
        lines = text.strip().split('\n')
        
        plot_info = {
            'location': {'x': x, 'y': y},
            'dimensions': {'width': w, 'height': h},
            'area': w * h,
            'raw_text': text
        }
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['plot', 'lot', 'number']):
                numbers = ''.join(filter(str.isalnum, line))
                if numbers:
                    plot_info['plot_number'] = numbers
                    break
        
        return plot_info

    def visualize_results(self, image: np.ndarray, plots: List[Dict[str, Any]]) -> None:
        vis_image = image.copy()
        
        for plot in plots:
            x = plot['location']['x']
            y = plot['location']['y']
            w = plot['dimensions']['width']
            h = plot['dimensions']['height']
            
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if 'plot_number' in plot:
                cv2.putText(vis_image, f"Plot {plot['plot_number']}", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def save_to_json(self, plots: List[Dict[str, Any]], filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(plots, f, indent=2)

# Save this as example_usage.py
def process_layout_image(image_path: str) -> None:
    extractor = PlotExtractor()
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    preprocessed = extractor.preprocess_image(image)
    plots = extractor.detect_plots(preprocessed)
    extractor.visualize_results(image, plots)
    extractor.save_to_json(plots, 'plot_information.json')
    
    return plots

if __name__ == "__main__":
    # Example usage
    image_path = "layout_image.jpg"
    plots = process_layout_image(image_path)
    print(f"Found {len(plots)} plots:")
    for plot in plots:
        print(f"\nPlot {plot.get('plot_number', 'Unknown')}:")
        print(f"Location: ({plot['location']['x']}, {plot['location']['y']})")
        print(f"Dimensions: {plot['dimensions']['width']}x{plot['dimensions']['height']}")
        print(f"Area: {plot['area']} square pixels")