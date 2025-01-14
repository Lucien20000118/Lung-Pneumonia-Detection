import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tools.model import UNet, LungDataset
from tools.record import CSVLogger, ModelSaver, plot_loss_curve 
from tools.eval_util import iou, dice_coefficient, pixel_accuracy, precision_recall, specificity

# Paths and Hyperparameters
current_dir = os.getcwd()
print(f"Current Path: {current_dir}")

# Dataset Paths
lung_data_path = os.path.join(current_dir, "chest-x-ray-lungs-segmentation/Chest-X-Ray/Chest-X-Ray")
image_dir = os.path.join(lung_data_path, "image")
mask_dir = os.path.join(lung_data_path, "mask")

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Normalize automatically
])

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Load and Split Dataset
all_images = os.listdir(image_dir)
train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

train_dataset = LungDataset(image_dir, mask_dir, transform=transform)
val_dataset = LungDataset(image_dir, mask_dir, transform=transform)
train_dataset.images = train_images
val_dataset.images = val_images

dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize U-Net Model
unet_model = UNet().to(device)
criterion_segmentation = nn.BCELoss()
optimizer_segmentation = optim.Adam(unet_model.parameters(), lr=0.0001)

# Logger and Saver
csv_logger = CSVLogger("./output/training_log.csv")
model_saver = ModelSaver("./output/best_unet.pth")

# Training Hyperparameters
epochs = 100
patience = 5
train_losses = []
val_losses = []
no_improve_epochs = 0
consecutive_increase_epochs = 0

# Training Loop
print("Starting U-Net Training...")
plt.figure(figsize=(10, 6))

for epoch in tqdm(range(epochs), desc="Epoch Progress", unit="epoch"):
    epoch_start_time = time.time()

    # Training Phase
    unet_model.train()
    running_loss = 0.0
    for images, masks in tqdm(dataloader_train, desc=f"Training Epoch {epoch+1}/{epochs}", unit="batch"):
        images, masks = images.to(device), masks.to(device)

        optimizer_segmentation.zero_grad()
        outputs = unet_model(images)
        loss = criterion_segmentation(outputs, masks)
        loss.backward()
        optimizer_segmentation.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader_train)
    train_losses.append(train_loss)

    # Validation Phase
    unet_model.eval()
    val_running_loss = 0.0
    all_metrics = {"iou": [], "dice": [], "pixel_accuracy": [], "precision": [], "recall": [], "specificity": []}
    with torch.no_grad():
        for images, masks in dataloader_val:
            images, masks = images.to(device), masks.to(device)
            outputs = unet_model(images)
            loss = criterion_segmentation(outputs, masks)
            val_running_loss += loss.item()

            preds = (outputs > 0.5).float()
            masks = (masks > 0.5).float()

            all_metrics["iou"].append(iou(preds, masks))
            all_metrics["dice"].append(dice_coefficient(preds, masks))
            all_metrics["pixel_accuracy"].append(pixel_accuracy(preds, masks))
            precision, recall = precision_recall(preds, masks)
            all_metrics["precision"].append(precision)
            all_metrics["recall"].append(recall)
            all_metrics["specificity"].append(specificity(preds, masks))

    val_loss = val_running_loss / len(dataloader_val)
    val_losses.append(val_loss)

    avg_metrics = {key: sum(values) / len(values) for key, values in all_metrics.items()}
    metrics_to_log = [avg_metrics["iou"], avg_metrics["dice"], avg_metrics["pixel_accuracy"],
                      avg_metrics["precision"], avg_metrics["recall"], avg_metrics["specificity"]]

    print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Metrics - IoU: {avg_metrics['iou']:.4f}, Dice: {avg_metrics['dice']:.4f}, Pixel Accuracy: {avg_metrics['pixel_accuracy']:.4f}, "
          f"Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}, Specificity: {avg_metrics['specificity']:.4f}")

    csv_logger.log(epoch + 1, train_loss, val_loss, metrics_to_log)

    if val_loss < model_saver.best_val_loss:
        model_saver.save(unet_model.state_dict(), val_loss)
        no_improve_epochs = 0
        consecutive_increase_epochs = 0
    else:
        no_improve_epochs += 1
        consecutive_increase_epochs += 1
        print(f"No improvement in validation loss for {no_improve_epochs} epoch(s).")

    if consecutive_increase_epochs >= 3:
        lr = optimizer_segmentation.param_groups[0]['lr'] * 0.5
        for param_group in optimizer_segmentation.param_groups:
            param_group['lr'] = lr
        print(f"Learning rate adjusted to {lr:.6f}")
        consecutive_increase_epochs = 0

    if no_improve_epochs >= patience:
        print("Early stopping triggered.")
        break

    plot_loss_curve(train_losses, val_losses)

    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    remaining_epochs = epochs - (epoch + 1)
    estimated_time_remaining = remaining_epochs * elapsed_time
    print(f"Epoch {epoch+1}/{epochs} completed. Estimated time remaining: {estimated_time_remaining:.2f} seconds")

plot_loss_curve(train_losses, val_losses)
print("Training complete. Best model saved.")
