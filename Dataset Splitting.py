import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil


#copy files into pycharm
csv_file_path = '/Users/Rex/PycharmProjects/Neuro 140/archive (13)/Data_Entry_2017.csv'
df = pd.read_csv(csv_file_path)
print(df)

#get column names
column_names = df.columns
print(column_names)
#result only want to keep the first 2 columns the rest is unncessary

#keep the first two columns
df_subset = df.iloc[:, :2]
print(df_subset)

#view columns number
column_name = 'Finding Labels'
value_counts = df_subset[column_name].value_counts()
print(value_counts)

#create a copy
df_2 = df_subset.copy()

# Check if "|" is present in each cell of the specified column and relabel as "Multi"
df_2[column_name] = df[column_name].apply(lambda x: 'Multi' if '|' in str(x) else x)

# Display the updated DataFrame
print(df_2)

#recheck if it has 800-something
value_counts_2 = df_2[column_name].value_counts()
print(value_counts_2)

#resize all the images to a 224X224 image
from PIL import Image
import os

#def resize_images_in_folder(input_folder, output_folder, target_size=(224, 224)):
    # Create the output folder if it doesn't exist
    #if not os.path.exists(output_folder):
        #os.makedirs(output_folder)

    # Recursive function for traversing subfolders
    #def process_folder(folder_path):
        #for item in os.listdir(folder_path):
            #item_path = os.path.join(folder_path, item)

            #if os.path.isdir(item_path):
                # If it's a subfolder, recursively process it
                #process_folder(item_path)
            #else:
                # If it's a file, check if it's an image
                #valid_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
                #if any(item.lower().endswith(ext) for ext in valid_extensions):
                    # Create corresponding subfolder structure in the output directory
                    #relative_path = os.path.relpath(item_path, input_folder)
                    #output_subfolder = os.path.join(output_folder, os.path.dirname(relative_path))
                    #if not os.path.exists(output_subfolder):
                        #os.makedirs(output_subfolder)

                    # Open the image
                    #with Image.open(item_path) as img:
                        # Resize the image
                        #resized_img = img.resize(target_size)

                        # Save the resized image
                        #output_path = os.path.join(output_subfolder, os.path.basename(item_path))
                        #resized_img.save(output_path)

    # Start processing from the top-level input folder
    #process_folder(input_folder)

#if __name__ == "__main__":
    # Specify your input and output folders
    #input_folder = "/Users/Rex/PycharmProjects/Neuro 140/archive (13)"
    #output_folder = "/Users/Rex/PycharmProjects/Neuro 140/archive (13) resize images"

    # Specify the target size for resizing
    #target_size = (224, 224)

    # Resize images and save to the output folder
    #resize_images_in_folder(input_folder, output_folder, target_size)

#create a copy
df_3 = df_2.copy()

# Create a new column 'New Label Column' with 0 for 'No Finding' and 1 for other values
df_3['abnormal'] = np.where(df_3['Finding Labels'] == 'No Finding', 0, 1)
df_3 = df_3.rename(columns={'Finding Labels': 'class_name'})
print(df_3)


#### WORKED 3/8/24
# Assuming you have a DataFrame named your_dataset with 'abnormal' column
# Replace this with your actual data loading process

# Number of observations for train and test sets
#num_train_observations = 13000
#num_test_observations = 1000

# Percentage of 1s and 0s for train and test sets
#train_percentage_1 = 0.7
#test_percentage_1 = 0.7

# Create a mask for abnormal = 1 and abnormal = 0
#mask_abnormal_1 = df_3['abnormal'] == 1
#mask_abnormal_0 = df_3['abnormal'] == 0

# Randomly sample from abnormal = 1 for train set
#train_abnormal_1 = df_3[mask_abnormal_1].sample(n=int(num_train_observations * train_percentage_1), random_state=42)

# Randomly sample from abnormal = 0 for train set
#train_abnormal_0 = df_3[mask_abnormal_0].sample(n=int(num_train_observations * (1 - train_percentage_1)), random_state=42)

# Concatenate train sets
#train_dataset = pd.concat([train_abnormal_1, train_abnormal_0], axis=0)

# Remove sampled entries from the original dataset
#df_3 = df_3.drop(train_dataset.index)

# Randomly sample from remaining abnormal = 1 for test set
#test_abnormal_1 = df_3[mask_abnormal_1].sample(n=int(num_test_observations * test_percentage_1), random_state=42)

# Randomly sample from remaining abnormal = 0 for test set
#test_abnormal_0 = df_3[mask_abnormal_0].sample(n=int(num_test_observations * (1 - test_percentage_1)), random_state=42)

# Concatenate test sets
#test_dataset = pd.concat([test_abnormal_1, test_abnormal_0], axis=0)

# Save the datasets to separate CSV files
#train_dataset.to_csv('train_dataset.csv', index=False)
#test_dataset.to_csv('test_dataset.csv', index=False)

#WORKED 2/8/24 on TEST
# Replace with the actual path to your CSV file
#csv_file_path = '/Users/Rex/PycharmProjects/Neuro 140/test_dataset.csv'
#df = pd.read_csv(csv_file_path)

# Replace with the actual path to the main folder containing subfolders
#main_folder_path = '/Users/Rex/PycharmProjects/Neuro 140/archive (13) resize images'

# Replace with the desired path for the new folder where copies will be placed
#output_folder_path = '/Users/Rex/PycharmProjects/Neuro 140/archive (13) resize images/test'

# Create the output folder if it doesn't exist
#os.makedirs(output_folder_path, exist_ok=True)

# Replace 'ImageName' with the actual column name containing image names in your CSV file
#for index, row in df.iterrows():
    #image_name = row['Image Index']  # Replace with the actual column name

    # Traverse through each subfolder in the main folder
    #for subfolder_name in os.listdir(main_folder_path):
        #subfolder_path = os.path.join(main_folder_path, subfolder_name)

        # Check if the current item in the main folder is a subfolder
        #if os.path.isdir(subfolder_path):
            # Traverse through each inner folder within the subfolder
            #for inner_folder_name in os.listdir(subfolder_path):
                #inner_folder_path = os.path.join(subfolder_path, inner_folder_name)

                # Check if the current item in the subfolder is an inner folder
                #if os.path.isdir(inner_folder_path):
                    # Construct the full path to the image file within the inner folder
                    #image_path = os.path.join(inner_folder_path, image_name)

                    # Check if the image file exists before copying
                    #if os.path.exists(image_path):
                        #output_path = os.path.join(output_folder_path, f'{subfolder_name}_{inner_folder_name}_{image_name}')
                        #shutil.copy(image_path, output_path)

#print("Copying completed.")

#worked
# Replace with the actual path to your CSV file
#csv_file_path = '/Users/Rex/PycharmProjects/Neuro 140/train_dataset.csv'
#df = pd.read_csv(csv_file_path)

# Replace with the actual path to the main folder containing subfolders
#main_folder_path = '/Users/Rex/PycharmProjects/Neuro 140/archive (13) resize images'

# Replace with the desired path for the new folder where copies will be placed
#output_folder_path = '/Users/Rex/PycharmProjects/Neuro 140/archive (13) resize images/train'

# Create the outputz folder if it doesn't exist
#os.makedirs(output_folder_path, exist_ok=True)

# Replace 'ImageName' with the actual column name containing image names in your CSV file
#for index, row in df.iterrows():
    #image_name = row['Image Index']  # Replace with the actual column name

    # Traverse through each subfolder in the main folder
    #for subfolder_name in os.listdir(main_folder_path):
        #subfolder_path = os.path.join(main_folder_path, subfolder_name)

        # Check if the current item in the main folder is a subfolder
        #if os.path.isdir(subfolder_path):
            # Traverse through each inner folder within the subfolder
            #for inner_folder_name in os.listdir(subfolder_path):
                #inner_folder_path = os.path.join(subfolder_path, inner_folder_name)

                # Check if the current item in the subfolder is an inner folder
                #if os.path.isdir(inner_folder_path):
                    # Construct the full path to the image file within the inner folder
                    #image_path = os.path.join(inner_folder_path, image_name)

                    # Check if the image file exists before copying
                    #if os.path.exists(image_path):
                        #output_path = os.path.join(output_folder_path, f'{subfolder_name}_{inner_folder_name}_{image_name}')
                        #shutil.copy(image_path, output_path)

#print("Copying completed.")

