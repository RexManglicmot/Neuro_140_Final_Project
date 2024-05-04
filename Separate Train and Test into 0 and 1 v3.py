#4/6/24
#11:34 pm
#BELEIVED IT WORKED

import os
import shutil
import pandas as pd


def index_images(base_dir):
    """Index all images by their names and return a mapping to their full paths."""
    image_paths = {}
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png'):  # Assuming all images are .png, adjust if necessary
                image_paths[file] = os.path.join(root, file)
    return image_paths


def organize_images(csv_path, image_paths, organized_dir):
    # Ensure the base organized directory and subdirectories "0" and "1" exist
    for label in ['0', '1']:
        label_dir = os.path.join(organized_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        image_name = row['Image Index']
        classification = str(row['abnormal'])

        if image_name in image_paths:
            src_path = image_paths[image_name]
            dest_path = os.path.join(organized_dir, classification, os.path.basename(src_path))
            shutil.copy(src_path, dest_path)
            print(f"Copied: {src_path} to {dest_path}")
        else:
            print(f"Image not found in index: {image_name}")


# Base directory where your images are stored
base_images_dir = '/archive (13) resize images'

# Organized directories
train_organized_dir = os.path.join(base_images_dir, "train_organized_cmon")
test_organized_dir = os.path.join(base_images_dir, "test_organized_cmon")

# CSV files
train_csv_path = '/train_dataset.csv'
test_csv_path = '/test_dataset.csv'

# Index all images in the base directory
image_paths = index_images(base_images_dir)

# Organize images based on the CSV files
organize_images(train_csv_path, image_paths, train_organized_dir)
organize_images(test_csv_path, image_paths, test_organized_dir)
