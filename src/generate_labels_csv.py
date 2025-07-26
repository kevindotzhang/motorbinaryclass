import os
import pandas as pd

# Local paths based on your structure
raw_old_path = "../data/raw/old"
raw_new_path = "../data/raw/new"
output_csv_path = "../data/labels_large_small.csv"

image_data = []

def collect_labeled_images(root_path):
    for root, _, files in os.walk(root_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                label = None
                if "large" in file.lower():
                    label = "large"
                elif "small" in file.lower():
                    label = "small"
                if label:
                    full_path = os.path.join(root, file)
                    image_data.append({"image_path": full_path, "label": label})

# Run for both sets
collect_labeled_images(raw_old_path)
collect_labeled_images(raw_new_path)

# Save to CSV
df = pd.DataFrame(image_data)
df.to_csv(output_csv_path, index=False)

print(f"Saved {len(df)} labeled image entries to {output_csv_path}")
