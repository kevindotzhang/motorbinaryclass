import torch
from torchvision import models, transforms
from PIL import Image
import sys
import os

# Load model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
model_path = os.path.join(BASE_DIR, 'models', 'large_small_motor_classifier_focal_augmented.pth')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Predict function
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        label = "large" if prob > 0.6 else "small"
    print(f"Prediction: {label} (confidence: {prob:.4f})")

# Run from command line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)