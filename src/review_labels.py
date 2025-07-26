import os
import streamlit as st
import pandas as pd
from PIL import Image

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_CSV = os.path.join(BASE_DIR, 'data', 'labels_large_small.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'data', 'labels_large_small_reviewed.csv')

# Existing Labels
df = pd.read_csv(DATA_CSV)
if os.path.exists(OUTPUT_CSV):
    reviewed_df = pd.read_csv(OUTPUT_CSV)
    reviewed_paths = set(reviewed_df['image_path'])
else:
    reviewed_df = pd.DataFrame(columns=['image_path', 'label'])
    reviewed_paths = set()

# Getting next image to review
remaining_df = df[~df['image_path'].isin(reviewed_paths)]

st.title('Quick Review for Large/Small Motor Images')

if remaining_df.empty:
    st.success("All images have been reviewed!")
else:
    row = remaining_df.iloc[0]
    image_path = row['image_path']
    st.image(Image.open(image_path), caption=image_path, use_column_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Large Image"):
            reviewed_df = pd.concat([reviewed_df, pd.DataFrame([{'image_path': image_path, 'label': 'large'}])], ignore_index=True)
            reviewed_df.to_csv(OUTPUT_CSV, index=False)
            st.rerun()
    with col2:
        if st.button("Small Image"):
            reviewed_df = pd.concat([reviewed_df, pd.DataFrame([{'image_path': image_path, 'label': 'small'}])], ignore_index=True)
            reviewed_df.to_csv(OUTPUT_CSV, index=False)
            st.rerun()
    with col3:
        if st.button("Mixed Image"):
            reviewed_df = pd.concat([reviewed_df, pd.DataFrame([{'image_path': image_path, 'label': 'mixed'}])], ignore_index=True)
            reviewed_df.to_csv(OUTPUT_CSV, index=False)
            st.rerun()