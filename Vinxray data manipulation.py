#4/7/24
#DONE
#had to transform this thannnnnnngggg

import pandas as pd
import os
import shutil

def organize_images(csv_path, src_dir, dest_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Create subdirectories for each class
    for label in ['0', '1']:
        label_dir = os.path.join(dest_dir, label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Loop through the DataFrame and move/copy images
    for index, row in df.iterrows():
        src_path = os.path.join(src_dir, row['full_path'])
        dest_path = os.path.join(dest_dir, str(row['abnormal']), os.path.basename(row['full_path']))

        if os.path.exists(src_path):
            # Uncomment the line below to move files
            # shutil.move(src_path, dest_path)

            # Uncomment the line below to copy files instead
            shutil.copy(src_path, dest_path)
        else:
            print(f"File not found: {src_path}")


# Base directory where your images are currently stored
base_images_dir = '/vinxray/train'

# Destination directories
train_viet_dir = '/test viet'
train_test_dir = '/train viet'

# Paths to the CSV files
train_vt_csv_path = '/train_vt.csv'
test_vt_csv_path = '/test_vt.csv'

# Organize images based on the CSV files
organize_images(train_vt_csv_path, base_images_dir, train_viet_dir)
organize_images(test_vt_csv_path, base_images_dir, train_test_dir)
