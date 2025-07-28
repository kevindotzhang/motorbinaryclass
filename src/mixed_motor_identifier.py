import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

class MixedMotorAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.output_dir = os.path.join(base_dir, 'data', 'mixed_motors')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_labeled_data(self):
        """Load reviewed labels from CSV."""
        csv_path = os.path.join(self.data_dir, 'labels_large_small_reviewed.csv')
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found!")
            return None
        
        df = pd.read_csv(csv_path)
        print(f"Total labeled images: {len(df)}")
        print(f"Label distribution:\n{df['label'].value_counts()}")
        return df
    
    def get_mixed_images(self, df):
        """Extract images labeled as 'mixed'."""
        mixed_df = df[df['label'] == 'mixed'].copy()
        print(f"\nFound {len(mixed_df)} mixed motor images")
        return mixed_df
    
    def analyze_mixed_images(self, mixed_df):
        """Analyze properties of mixed motor images."""
        if len(mixed_df) == 0:
            print("No mixed images found!")
            return
        
        image_stats = []
        
        for idx, row in mixed_df.iterrows():
            img_path = row['image_path']
            if os.path.exists(img_path):
                img = Image.open(img_path)
                width, height = img.size
                img_array = np.array(img)
                
                stats = {
                    'image_path': img_path,
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height,
                    'file_size_kb': os.path.getsize(img_path) / 1024,
                    'mode': img.mode,
                    'mean_brightness': np.mean(img_array) if len(img_array.shape) > 2 else np.mean(img_array)
                }
                image_stats.append(stats)
            else:
                print(f"Warning: Image not found - {img_path}")
        
        stats_df = pd.DataFrame(image_stats)
        
        print("\n=== Mixed Images Analysis ===")
        print(f"Average dimensions: {stats_df['width'].mean():.0f} x {stats_df['height'].mean():.0f}")
        print(f"Dimension range: ({stats_df['width'].min()}-{stats_df['width'].max()}) x "
              f"({stats_df['height'].min()}-{stats_df['height'].max()})")
        print(f"Average file size: {stats_df['file_size_kb'].mean():.1f} KB")
        print(f"Image modes: {Counter(stats_df['mode'])}")
        
        return stats_df
    
    def create_mixed_images_list(self, mixed_df):
        """Create CSV for mixed motor images with annotation columns."""
        output_path = os.path.join(self.output_dir, 'mixed_motor_images.csv')
        
        mixed_df_annotate = mixed_df.copy()
        mixed_df_annotate['num_motors'] = None
        mixed_df_annotate['num_large'] = None
        mixed_df_annotate['num_small'] = None
        mixed_df_annotate['annotated'] = False
        
        mixed_df_annotate.to_csv(output_path, index=False)
        print(f"\nMixed images list saved to: {output_path}")
        return output_path
    
    def visualize_sample_mixed_images(self, mixed_df, num_samples=6):
        """Create visualization of sample mixed motor images."""
        if len(mixed_df) == 0:
            print("No mixed images to visualize!")
            return
        
        num_samples = min(num_samples, len(mixed_df))
        sample_df = mixed_df.sample(n=num_samples, random_state=42)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(sample_df.iterrows()):
            if idx < num_samples:
                img_path = row['image_path']
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    axes[idx].imshow(img)
                    axes[idx].set_title(os.path.basename(img_path)[:20] + '...')
                    axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mixed_images_sample.png'))
        print(f"\nSample visualization saved")
        plt.show()
    
    def prepare_annotation_structure(self):
        """Create directory structure for annotations."""
        annotation_dirs = [
            os.path.join(self.output_dir, 'images'),
            os.path.join(self.output_dir, 'annotations'),
            os.path.join(self.output_dir, 'masks')
        ]
        
        for dir_path in annotation_dirs:
            os.makedirs(dir_path, exist_ok=True)
            
        print("\nAnnotation directory structure created")
    
    def run_analysis(self):
        """Execute complete analysis pipeline."""
        print("=== Mixed Motor Image Analysis ===\n")
        
        df = self.load_labeled_data()
        if df is None:
            return
        
        mixed_df = self.get_mixed_images(df)
        
        if len(mixed_df) > 0:
            stats_df = self.analyze_mixed_images(mixed_df)
            self.create_mixed_images_list(mixed_df)
            self.visualize_sample_mixed_images(mixed_df)
            self.prepare_annotation_structure()
            
            stats_path = os.path.join(self.output_dir, 'mixed_images_stats.csv')
            stats_df.to_csv(stats_path, index=False)
            print(f"\nStatistics saved to: {stats_path}")
            
            return mixed_df, stats_df
        else:
            print("\nNo mixed images found in dataset.")
            return None, None


if __name__ == "__main__":
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    analyzer = MixedMotorAnalyzer(BASE_DIR)
    mixed_df, stats_df = analyzer.run_analysis()