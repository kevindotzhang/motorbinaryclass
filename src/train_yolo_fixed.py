
from ultralytics import YOLO
import torch

# Using fixed data path
model = YOLO('yolov8n-seg.pt')

results = model.train(
    data=r'.\data\mixed_motors\yolo_data_fixed.yaml',
    epochs=300,
    batch=8,
    imgsz=640,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    project=r'.\models',
    name='motor_seg_fixed',
    patience=50,
    
    # Heavy augmentation for limited data
    degrees=30,
    translate=0.2,
    scale=0.5,
    shear=10,
    perspective=0.001,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.3,
    
    # Training parameters
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.001,
    warmup_epochs=5,
    close_mosaic=10,
    
    # Prevent overfitting
    dropout=0.1
)