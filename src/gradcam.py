import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
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

# Grad-CAM hook
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward(torch.ones_like(output))
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()
        if cam.max() == cam.min():
            cam = np.zeros_like(cam)
        else:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

# Run Grad-CAM on image
def show_gradcam(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    cam_generator = GradCAM(model, model.layer4)
    cam = cam_generator.generate_cam(input_tensor)

    img_np = np.array(image.resize((256, 256)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed_img = np.clip(0.5 * heatmap + 0.5 * img_np, 0, 255).astype(np.uint8)

    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')
    plt.show()

# Example usage
# show_gradcam("data/raw/old/Motor_Large motor_012.jpg")

# Use given an image
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python gradcam.py <image_path>")
        exit(1)
    show_gradcam(sys.argv[1])