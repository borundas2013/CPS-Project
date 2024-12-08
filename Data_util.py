import os
import pandas as pd
import numpy as np
from PIL import Image
import shutil
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def collect_gyro_accel_files():
    onedrive_path = os.path.expanduser("OneDrive_M/")
    dataset_folder = "dataset_M/"

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    files_moved = 0

    for root, dirs, files in os.walk(onedrive_path):
        for file in files:
    
            if file.endswith("_gyro_accel.csv"):  
                rel_path = os.path.relpath(root, onedrive_path)
                source_file = os.path.join(root, file)
                dest_file = os.path.join(dataset_folder, file)
                try:
                    shutil.move(source_file, dest_file)
                    files_moved += 1
                    print(f"Moved {file} to {dest_file}")
                except Exception as e:
                    print(f"Error moving {file}: {str(e)}")

    print(f"File collection complete. Moved {files_moved} files.")

def process_dataset_files():

    dataset_folder = "dataset_M/"
    if not os.path.exists(dataset_folder):
        print("Dataset folder not found.")
        return
    files_processed = 0

    for filename in os.listdir(dataset_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(dataset_folder, filename)
            
            try:
                df = pd.read_csv(file_path)
                df = df.iloc[:, 1:-1]
                df.to_csv(file_path, index=False)
                files_processed += 1
                print(f"Processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print(f"Processing complete. Modified {files_processed} files.")

def create_images_from_data():
    
    dataset_folder = "dataset_M/"
    image_folder = "images_M/"

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    if not os.path.exists(dataset_folder):
        print("Dataset folder not found.")
        return

    files_processed = 0
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(dataset_folder, filename)
            image_path = os.path.join(image_folder, filename.replace('.csv', '.png'))
            base, gyro, accel = filename[:-4].rsplit("_", 2)

            gyro_filename = f"{base}_{gyro}"
            accel_filename = f"{base}_{accel}"
           
            try:
               
                df = pd.read_csv(file_path)
                first_three_cols = df.iloc[:, :3]
                remaining_cols = df.iloc[:, 3:]
                plt.figure(figsize=(12, 6))
                for col in first_three_cols.columns:
                    plt.plot(df[col])
                plt.axis('off')
                first_image_path = os.path.join(image_folder, gyro_filename+'_M.png')
                plt.savefig(first_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                plt.figure(figsize=(12, 6))
                for col in remaining_cols.columns:
                    plt.plot(df[col])
                plt.axis('off')
                second_image_path = os.path.join(image_folder, accel_filename+'_M.png')
                plt.savefig(second_image_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                files_processed += 1
                print(f"Created image for {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print(f"Image creation complete. Created {files_processed} images.")

def augment_images():
    image_folder = "images_M/"
    
    if not os.path.exists(image_folder):
        print("Images folder not found.")
        return

    augmented_count = 0
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            
            try:

                img = Image.open(image_path)
                base_name = filename[:-4]  

                img_rot90 = img.rotate(90)
                img_rot90.save(os.path.join(image_folder, f"{base_name}_rot90.png"))
                augmented_count += 1
                
                img_hflip = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_hflip.save(os.path.join(image_folder, f"{base_name}_hflip.png"))
                augmented_count += 1

                img_vflip = img.transpose(Image.FLIP_TOP_BOTTOM)
                img_vflip.save(os.path.join(image_folder, f"{base_name}_vflip.png"))
                augmented_count += 1
                
                print(f"Created augmentations for {filename}")
                
            except Exception as e:
                print(f"Error augmenting {filename}: {str(e)}")
                
    print(f"Augmentation complete. Created {augmented_count} new images.")

def create_labels_csv():
  
    image_folder = "images_M/"
    if not os.path.exists(image_folder):
        print("Images folder not found.")
        return

    label_files = {}
    label_dict = {}
    current_label = 0
    
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
           
            label_name = filename.split('_variation')[0].replace('_', ' ')
            
            if label_name not in label_files:
                label_files[label_name] = []
            label_files[label_name].append(filename)
   
            if label_name not in label_dict:
                label_dict[label_name] = current_label
                current_label += 1
    
    data = []
    
    for label_name, files in label_files.items():
        if len(files) >= 2:  
            for i in range(0, len(files)-1, 2):
                data.append([
                    files[i],          
                    files[i+1],        
                    label_name,       
                    label_dict[label_name], 
                    1 
                ])
    
    csv_path = os.path.join(image_folder, 'labels_M.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename1', 'filename2', 'label_name', 'label','Gender'])
        writer.writerows(data)
        
    print(f"Created labels CSV file with {len(data)} paired entries")
    print("Label mappings:", label_dict)
    print("Files per label:", {k:len(v) for k,v in label_files.items() if len(v) >= 2})



if __name__ == "__main__":
    collect_gyro_accel_files()
    process_dataset_files()
    create_images_from_data()
    augment_images()
    create_labels_csv()
   