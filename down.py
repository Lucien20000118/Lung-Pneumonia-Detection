import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

# Save dataset to local directory
local_save_path = "./local_datasets/chest_xray_pneumonia"
os.makedirs(local_save_path, exist_ok=True)

shutil.copytree(path, local_save_path, dirs_exist_ok=True)

print("Path to dataset files:", path)
print("Dataset also saved locally at:", local_save_path)
