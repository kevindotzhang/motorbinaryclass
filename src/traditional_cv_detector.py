import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torchvision import transforms
from PIL import Image

class TraditionalMotorDetector:
    def __init__(self, classifier_path):
        # Load your existing binary classifier
        from torchvision import models
        self.classifier = models.resnet18()
        self.classifier.fc = torch.nn.Linear(self.classifier.fc.in_features, 1)
        self.classifier.load_state_dict(torch.load(classifier_path))
        self.classifier.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def detect_motors(self, image_path):
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 1000  # Adjust based on your images
        motor_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        results = []
        for contour in motor_contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Crop motor region
            motor_crop = img[y:y+h, x:x+w]
            
            # Classify using your binary classifier
            pil_image = Image.fromarray(cv2.cvtColor(motor_crop, cv2.COLOR_BGR2RGB))
            tensor = self.transform(pil_image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.classifier(tensor)
                is_large = torch.sigmoid(output).item() > 0.6
            
            results.append({
                'bbox': [x, y, w, h],
                'contour': contour,
                'class': 'motor_large' if is_large else 'motor_small'
            })
        
        return results
