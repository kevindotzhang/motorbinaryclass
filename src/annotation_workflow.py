#!/usr/bin/env python3
"""
Complete annotation workflow for mixed motor images
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path
import subprocess
import yaml

class AnnotationWorkflow:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.mixed_dir = os.path.join(base_dir, 'data', 'mixed_motors')
        self.annotations_dir = os.path.join(self.mixed_dir, 'annotations')
        
    def setup_labelme_project(self):
        """Setup project for LabelMe annotation"""
        print("=== Setting up LabelMe Project ===")
        
        # Read mixed images CSV
        csv_path = os.path.join(self.mixed_dir, 'mixed_motor_images_filtered.csv')
        df = pd.read_csv(csv_path)
        
        # Create labelme directory
        labelme_dir = os.path.join(self.mixed_dir, 'labelme_project')
        images_dir = os.path.join(labelme_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # Copy images and create initial annotation files
        print(f"Copying {len(df)} images...")
        for idx, row in df.iterrows():
            src = row['image_path']
            # Create consistent naming
            ext = os.path.splitext(src)[1]
            dst_name = f"mixed_{idx:04d}{ext}"
            dst = os.path.join(images_dir, dst_name)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
                # Create empty annotation file
                annotation = {
                    "version": "5.0.1",
                    "flags": {},
                    "shapes": [],
                    "imagePath": dst_name,
                    "imageData": None,
                    "imageHeight": None,
                    "imageWidth": None
                }
                
                json_path = os.path.join(images_dir, f"mixed_{idx:04d}.json")
                with open(json_path, 'w') as f:
                    json.dump(annotation, f, indent=2)
            else:
                print(f"Warning: Image not found - {src}")
        
        # Create label config
        label_config = {
            "motor_small": {"color": [0, 255, 0], "shortcut": "s"},
            "motor_large": {"color": [255, 0, 0], "shortcut": "l"}
        }
        
        config_path = os.path.join(labelme_dir, 'label_config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(label_config, f)
        
        # Create mapping file
        mapping_data = []
        for idx, row in df.iterrows():
            mapping_data.append({
                'new_name': f"mixed_{idx:04d}{os.path.splitext(row['image_path'])[1]}",
                'original_path': row['image_path'],
                'index': idx
            })
        
        mapping_df = pd.DataFrame(mapping_data)
        mapping_df.to_csv(os.path.join(labelme_dir, 'image_mapping.csv'), index=False)
        
        print(f"\nLabelMe project created at: {labelme_dir}")
        
        return labelme_dir
    
    def convert_labelme_to_coco(self, labelme_dir):
        """Convert LabelMe annotations to COCO format"""
        print("\n=== Converting to COCO Format ===")
        
        images_dir = os.path.join(labelme_dir, 'images')
        output_path = os.path.join(self.annotations_dir, 'annotations_coco.json')
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "motor_small", "supercategory": "motor"},
                {"id": 2, "name": "motor_large", "supercategory": "motor"}
            ]
        }
        
        annotation_id = 1
        
        # Process each JSON file
        json_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.json')])
        
        for img_id, json_file in enumerate(json_files, 1):
            json_path = os.path.join(images_dir, json_file)
            
            with open(json_path, 'r') as f:
                labelme_data = json.load(f)
            
            # Add image info
            img_name = labelme_data['imagePath']
            img_path = os.path.join(images_dir, img_name)
            
            if os.path.exists(img_path):
                from PIL import Image
                img = Image.open(img_path)
                width, height = img.size
                
                coco_data['images'].append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": width,
                    "height": height
                })
                
                # Add annotations
                for shape in labelme_data['shapes']:
                    if shape['shape_type'] == 'polygon':
                        points = shape['points']
                        
                        # Flatten points
                        segmentation = [[coord for point in points for coord in point]]
                        
                        # Calculate bounding box
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                        area = (x_max - x_min) * (y_max - y_min)
                        
                        category_id = 1 if shape['label'] == 'motor_small' else 2
                        
                        coco_data['annotations'].append({
                            "id": annotation_id,
                            "image_id": img_id,
                            "category_id": category_id,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
        
        # Save COCO format
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"COCO annotations saved to: {output_path}")
        print(f"Total images: {len(coco_data['images'])}")
        print(f"Total annotations: {len(coco_data['annotations'])}")
        
        # Create summary
        small_count = sum(1 for ann in coco_data['annotations'] if ann['category_id'] == 1)
        large_count = sum(1 for ann in coco_data['annotations'] if ann['category_id'] == 2)
        print(f"Small motors annotated: {small_count}")
        print(f"Large motors annotated: {large_count}")
        
        return output_path
    
    def create_yolo_annotations(self, coco_path):
        """Convert COCO to YOLO format for training"""
        print("\n=== Creating YOLO Format ===")
        
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)
        
        yolo_dir = os.path.join(self.annotations_dir, 'yolo_format')
        labels_dir = os.path.join(yolo_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)
        
        # Create image id to filename mapping
        id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}
        id_to_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Create YOLO annotations
        for img_id, annotations in annotations_by_image.items():
            filename = id_to_filename[img_id]
            width, height = id_to_dims[img_id]
            
            txt_filename = os.path.splitext(filename)[0] + '.txt'
            txt_path = os.path.join(labels_dir, txt_filename)
            
            with open(txt_path, 'w') as f:
                for ann in annotations:
                    # YOLO format: class_id center_x center_y width height (normalized)
                    bbox = ann['bbox']
                    x_center = (bbox[0] + bbox[2]/2) / width
                    y_center = (bbox[1] + bbox[3]/2) / height
                    w = bbox[2] / width
                    h = bbox[3] / height
                    
                    # Class: 0 for small, 1 for large
                    class_id = 0 if ann['category_id'] == 1 else 1
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        
        # Create YOLO config files
        data_yaml = {
            'path': self.mixed_dir,
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,
            'names': ['motor_small', 'motor_large']
        }
        
        yaml_path = os.path.join(yolo_dir, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"YOLO annotations saved to: {yolo_dir}")
        print(f"Total annotated images: {len(annotations_by_image)}")
        
        return yolo_dir
    
    def split_dataset(self, labelme_dir, train_ratio=0.8):
        """Split dataset into train/val sets"""
        print("\n=== Splitting Dataset ===")
        
        images_dir = os.path.join(labelme_dir, 'images')
        
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(image_files)
        
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Create split directories
        for split_name, files in [('train', train_files), ('val', val_files)]:
            split_dir = os.path.join(self.mixed_dir, 'images', split_name)
            os.makedirs(split_dir, exist_ok=True)
            
            for file in files:
                src = os.path.join(images_dir, file)
                dst = os.path.join(split_dir, file)
                shutil.copy2(src, dst)
        
        print(f"Train set: {len(train_files)} images")
        print(f"Val set: {len(val_files)} images")
        
        return train_files, val_files


# Usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Annotation workflow for mixed motors')
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory path')
    parser.add_argument('--setup', action='store_true', help='Setup LabelMe project')
    parser.add_argument('--convert', action='store_true', help='Convert annotations to COCO/YOLO')
    parser.add_argument('--labelme_dir', type=str, help='LabelMe project directory')
    
    args = parser.parse_args()
    
    workflow = AnnotationWorkflow(args.base_dir)
    
    if args.setup:
        labelme_dir = workflow.setup_labelme_project()
    
    if args.convert and args.labelme_dir:
        coco_path = workflow.convert_labelme_to_coco(args.labelme_dir)
        yolo_dir = workflow.create_yolo_annotations(coco_path)
        workflow.split_dataset(args.labelme_dir)
        
        print("\n=== Ready for Training ===")