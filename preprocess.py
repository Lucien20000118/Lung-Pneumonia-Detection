import kagglehub
import shutil
import os
import random


# Reorganize the second dataset
def reorganize_dataset(base_path):
    # Combine NORMAL and PNEUMONIA from train, val, and test
    combined_normal = []
    combined_pneumonia = []

    for dataset_type in ["train", "val", "test"]:
        normal_path = os.path.join(base_path, dataset_type, "NORMAL")
        pneumonia_path = os.path.join(base_path, dataset_type, "PNEUMONIA")

        combined_normal.extend([os.path.join(normal_path, img) for img in os.listdir(normal_path) if os.path.isfile(os.path.join(normal_path, img))])
        combined_pneumonia.extend([os.path.join(pneumonia_path, img) for img in os.listdir(pneumonia_path) if os.path.isfile(os.path.join(pneumonia_path, img))])

    # Shuffle the combined data
    random.shuffle(combined_normal)
    random.shuffle(combined_pneumonia)

    # Calculate new train size
    target_train_size = int(
        min(len(combined_normal), len(combined_pneumonia)) * 0.8
    )

    # Split data into new train and val sets
    new_train_normal = combined_normal[:target_train_size]
    new_val_normal = combined_normal[target_train_size:]

    new_train_pneumonia = combined_pneumonia[:target_train_size]
    new_val_pneumonia = combined_pneumonia[target_train_size:]

    # Define new paths
    new_train_normal_path = os.path.join(base_path, "train", "NORMAL")
    new_train_pneumonia_path = os.path.join(base_path, "train", "PNEUMONIA")
    new_val_normal_path = os.path.join(base_path, "val", "NORMAL")
    new_val_pneumonia_path = os.path.join(base_path, "val", "PNEUMONIA")

    # Temporarily move files to avoid overwriting
    temp_path = os.path.join(base_path, "temp")
    os.makedirs(temp_path, exist_ok=True)

    for img_path in combined_normal + combined_pneumonia:
        shutil.move(img_path, temp_path)

    # Clear and recreate target directories
    for path in [new_train_normal_path, new_train_pneumonia_path, new_val_normal_path, new_val_pneumonia_path]:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    # Move files into the new structure
    for img_path in new_train_normal:
        shutil.move(os.path.join(temp_path, os.path.basename(img_path)), new_train_normal_path)

    for img_path in new_train_pneumonia:
        shutil.move(os.path.join(temp_path, os.path.basename(img_path)), new_train_pneumonia_path)

    for img_path in new_val_normal:
        shutil.move(os.path.join(temp_path, os.path.basename(img_path)), new_val_normal_path)

    for img_path in new_val_pneumonia:
        shutil.move(os.path.join(temp_path, os.path.basename(img_path)), new_val_pneumonia_path)

    # Clean up temporary directory
    shutil.rmtree(temp_path)

    print("Dataset reorganization complete.")
    print(f"Train/NORMAL: {len(new_train_normal)}")
    print(f"Train/PNEUMONIA: {len(new_train_pneumonia)}")
    print(f"Val/NORMAL: {len(new_val_normal)}")
    print(f"Val/PNEUMONIA: {len(new_val_pneumonia)}")

# local dataset path
local_pt = 'datasets'

# Download first dataset
path_unet = kagglehub.dataset_download("iamtapendu/chest-x-ray-lungs-segmentation")

# Save first dataset locally
local_save_path_unet = f"./{local_pt}/chest_xray_lungs_segmentation"
os.makedirs(local_save_path_unet, exist_ok=True)
shutil.copytree(path_unet, local_save_path_unet, dirs_exist_ok=True)
print("First dataset saved locally at:", local_save_path_unet)

# Download second dataset
path_classifier = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# Save second dataset locally
local_save_path_classifier = f"./{local_pt}/chest_xray_pneumonia"
os.makedirs(local_save_path_classifier, exist_ok=True)
shutil.copytree(path_classifier, local_save_path_classifier, dirs_exist_ok=True)
print("Second dataset saved locally at:", local_save_path_classifier)

# Example usage
base_path_classifier = os.path.join(local_save_path_classifier, "chest_xray")
reorganize_dataset(base_path_classifier)
