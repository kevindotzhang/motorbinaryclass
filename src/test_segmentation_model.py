import os
from ultralytics import YOLO
import pandas as pd
from PIL import Image
import numpy as np

def test_segmentation_model(base_dir):
    """Test the trained segmentation model on mixed images"""
    
    # Load the trained model
    model_path = os.path.join(base_dir, 'models', 'motor_seg_fixed', 'weights', 'best.pt')
    model = YOLO(model_path)
    
    print(f"=== Testing Segmentation Model ===")
    print(f"Model loaded from: {model_path}")
    
    # Test on a few mixed images
    test_images_dir = os.path.join(base_dir, 'data', 'mixed_motors', 'yolo_dataset', 'images')
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))][:5]
    
    results_data = []
    
    for img_name in test_images:
        img_path = os.path.join(test_images_dir, img_name)
        
        # Run inference
        results = model(img_path, conf=0.25)
        
        # Count detections
        motor_counts = {'small': 0, 'large': 0}
        
        if results[0].boxes is not None:
            for cls in results[0].boxes.cls:
                class_name = model.names[int(cls)]
                motor_type = 'small' if 'small' in class_name else 'large'
                motor_counts[motor_type] += 1
        
        total = sum(motor_counts.values())
        
        print(f"\n{img_name}:")
        print(f"  Total motors detected: {total}")
        print(f"  Small motors: {motor_counts['small']}")
        print(f"  Large motors: {motor_counts['large']}")
        
        # Save visualization
        viz_path = img_path.replace('.jpg', '_detected.jpg')
        results[0].save(viz_path)
        
        results_data.append({
            'image': img_name,
            'total_motors': total,
            'small_motors': motor_counts['small'],
            'large_motors': motor_counts['large'],
            'small_proportion': motor_counts['small'] / total if total > 0 else 0,
            'large_proportion': motor_counts['large'] / total if total > 0 else 0
        })
    
    # Save results
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(base_dir, 'data', 'mixed_motors', 'test_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Visualizations saved with '_detected.jpg' suffix")
    
    return results_df

def analyze_all_mixed_images(base_dir):
    """Analyze all mixed images including non-annotated ones"""
    
    print("\n=== Analyzing All Mixed Images ===")
    
    # Load model
    model_path = os.path.join(base_dir, 'models', 'motor_seg_fixed', 'weights', 'best.pt')
    model = YOLO(model_path)
    
    # Load binary classifier for single motor images
    binary_model_path = os.path.join(base_dir, 'models', 'large_small_motor_classifier_focal_augmented.pth')
    
    # Load mixed images list
    mixed_csv = os.path.join(base_dir, 'data', 'mixed_motors', 'mixed_motor_images.csv')
    df = pd.read_csv(mixed_csv)
    
    results_list = []
    
    print(f"Analyzing {len(df)} images...")
    
    for idx, row in df.iterrows():
        if not os.path.exists(row['image_path']):
            continue
            
        # Run segmentation model
        seg_results = model(row['image_path'], conf=0.25)
        
        motor_counts = {'small': 0, 'large': 0}
        
        if seg_results[0].boxes is not None:
            for cls in seg_results[0].boxes.cls:
                class_name = model.names[int(cls)]
                motor_type = 'small' if 'small' in class_name else 'large'
                motor_counts[motor_type] += 1
        
        total = sum(motor_counts.values())
        
        # Determine analysis type
        if total == 0:
            analysis_type = "No motors detected (likely single/blurry)"
        elif total == 1:
            analysis_type = "Single motor"
        else:
            analysis_type = "Mixed motors"
        
        results_list.append({
            'image_path': row['image_path'],
            'original_label': row['label'],
            'detected_total': total,
            'detected_small': motor_counts['small'],
            'detected_large': motor_counts['large'],
            'small_proportion': motor_counts['small'] / total if total > 0 else 0,
            'large_proportion': motor_counts['large'] / total if total > 0 else 0,
            'analysis_type': analysis_type
        })
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(df)} images...")
    
    # Save comprehensive results
    results_df = pd.DataFrame(results_list)
    output_path = os.path.join(base_dir, 'data', 'mixed_motors', 'comprehensive_analysis.csv')
    results_df.to_csv(output_path, index=False)
    
    # Summary statistics
    print("\n=== Analysis Summary ===")
    print(f"Total images analyzed: {len(results_df)}")
    print(f"\nDetection breakdown:")
    print(results_df['analysis_type'].value_counts())
    
    print(f"\nImages by motor count:")
    print(results_df['detected_total'].value_counts().sort_index())
    
    # Average proportions for mixed images
    mixed_df = results_df[results_df['detected_total'] > 1]
    if len(mixed_df) > 0:
        print(f"\nFor mixed images (n={len(mixed_df)}):")
        print(f"  Average small motor proportion: {mixed_df['small_proportion'].mean():.2%}")
        print(f"  Average large motor proportion: {mixed_df['large_proportion'].mean():.2%}")
    
    print(f"\nFull results saved to: {output_path}")
    
    return results_df

if __name__ == "__main__":
    base_dir = '.'
    
    # Test on sample images
    test_results = test_segmentation_model(base_dir)
    
    # Analyze all mixed images
    all_results = analyze_all_mixed_images(base_dir)