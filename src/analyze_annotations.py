import os
import json
import pandas as pd
from pathlib import Path

def analyze_annotations(base_dir):
    """
    Analyze annotation coverage and statistics for motor segmentation dataset.
    """
    labelme_dir = os.path.join(base_dir, 'data', 'mixed_motors', 'labelme_project', 'images')
    json_files = list(Path(labelme_dir).glob('*.json'))
    
    print(f"=== Annotation Analysis ===")
    print(f"Total images: 70") # Only 70 because these are the ones I lablled as mixed/blurry when I did the binary classifier but I also removed the blurry ones to help with the model
    print(f"Annotated images: {len(json_files)}")
    print(f"Unannotated images: {77 - len(json_files)}")
    
    annotation_stats = []
    total_masks = 0
    motor_counts = {'motor_small': 0, 'motor_large': 0}
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        num_shapes = len(data.get('shapes', []))
        total_masks += num_shapes
        
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            if label in motor_counts:
                motor_counts[label] += 1
        
        annotation_stats.append({
            'image': json_file.stem,
            'num_motors': num_shapes,
            'image_path': data.get('imagePath', '')
        })
    
    print(f"\n=== Mask Statistics ===")
    print(f"Total masks created: {total_masks}")
    print(f"Small motors: {motor_counts['motor_small']}")
    print(f"Large motors: {motor_counts['motor_large']}")
    
    if len(annotation_stats) > 0:
        df = pd.DataFrame(annotation_stats)
        print(f"\n=== Per-Image Statistics ===")
        print(f"Average motors per image: {df['num_motors'].mean():.2f}")
        print(f"Max motors in an image: {df['num_motors'].max()}")
        print(f"Images with 0 motors: {len(df[df['num_motors'] == 0])}")
        print(f"Images with 2+ motors (mixed): {len(df[df['num_motors'] >= 2])}")
        
        print("\n=== Motor Count Distribution ===")
        distribution = df['num_motors'].value_counts().sort_index()
        for count, freq in distribution.items():
            print(f"{count} motors: {freq} images")
        
        output_path = os.path.join(base_dir, 'data', 'mixed_motors', 'annotation_stats.csv')
        df.to_csv(output_path, index=False)
        print(f"\nDetailed stats saved to: {output_path}")
        
        mixed_images = len(df[df['num_motors'] >= 2])
        total_masks = df['num_motors'].sum()
        
        print(f"\n=== Summary ===")
        print(f"Mixed images with annotations: {mixed_images}")
        print(f"Total segmentation masks: {total_masks}")
        
        if mixed_images >= 10 and total_masks >= 30:
            print("Status: Ready for training")
        else:
            print("Status: Additional annotations recommended")
    
    return annotation_stats

if __name__ == "__main__":
    base_dir = '.'
    analyze_annotations(base_dir)