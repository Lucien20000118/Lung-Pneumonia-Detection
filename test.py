import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from tools.model import UNet, LungDataset
import random

current_dir = os.getcwd()
destination_path = os.path.join(current_dir, "lung_segmentation_dataset/Chest-X-Ray")
image_dir = os.path.join(destination_path, "image")
mask_dir = os.path.join(destination_path, "mask")
weights_path = "output/best_unet.pth"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
train_dataset = LungDataset(image_dir, mask_dir, transform=transform)

# Load Model
unet_model = UNet().to(device)
checkpoint = torch.load(weights_path, map_location=device)
unet_model.load_state_dict(checkpoint['model_state_dict'])
unet_model.eval()

# Randomly Select an Image
random_index = random.randint(0, len(train_dataset) - 1)
image, ground_truth = train_dataset[random_index]

# Convert to Batch and Move to Device
image_batch = image.unsqueeze(0).to(device)

# Perform Inference
with torch.no_grad():
    probabilities = unet_model(image_batch)  # Model outputs logits

# Post-process the Output
predicted_mask = probabilities.squeeze(0).cpu().numpy()
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Binarize the output

# Convert Tensors to PIL Images
image_pil = transforms.ToPILImage()(image)
ground_truth_pil = transforms.ToPILImage()(ground_truth)
predicted_mask_pil = Image.fromarray((predicted_mask[0] * 255).astype(np.uint8))

# Create Composite Images
image_np = np.array(image_pil.convert("RGB"))
ground_truth_np = (np.array(ground_truth_pil) * 255).astype(np.uint8)  # Scale to 0-255 range
predicted_mask_np = np.array(predicted_mask_pil.convert("L"))

# Apply Ground Truth Mask (Red Overlay)
ground_truth_overlay = image_np.copy()
red_mask = np.zeros_like(image_np)
red_mask[:, :, 0] = ground_truth_np  # Red channel for ground truth

# Blend original image with red mask for ground truth overlay
ground_truth_overlay = (
    0.5 * image_np + 0.5 * red_mask
).astype(np.uint8)

# Apply Predicted Mask (Semi-transparent Purple Overlay)
predicted_overlay = image_np.copy()
purple_mask = np.zeros_like(image_np)
purple_mask[:, :, 0] = 128  # Red channel
purple_mask[:, :, 2] = 255  # Blue channel

predicted_overlay[predicted_mask_np > 0] = (
    0.5 * predicted_overlay[predicted_mask_np > 0] + 0.5 * purple_mask[predicted_mask_np > 0]
).astype(np.uint8)

# Display Results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image_np)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Ground Truth Overlay")
plt.imshow(ground_truth_overlay)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Predicted Mask Overlay")
plt.imshow(predicted_overlay)
plt.axis("off")

plt.tight_layout()
plt.show()
