import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tools.model import UNet, DepthwiseClassifier, LungDataset, ClassificationDataset
from tools.record import plot_loss_curve, CSVLogger, ModelSaver
from tools.eval_util import iou, dice_coefficient, pixel_accuracy, precision_recall, specificity

# Paths and Hyperparameters
current_dir = os.getcwd()
print(f"Current Path: {current_dir}")
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Common settings
img_size = 128
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

########################################
# Phase 1: Train U-Net
########################################
print("Starting Phase 1: Training U-Net")

# Paths for U-Net training data
lung_data_path = os.path.join(current_dir, "chest-x-ray-lungs-segmentation/Chest-X-Ray/Chest-X-Ray")
image_dir = os.path.join(lung_data_path, "image")
mask_dir = os.path.join(lung_data_path, "mask")

# Load and split datasets
all_images = os.listdir(image_dir)
train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

train_dataset = LungDataset(image_dir, mask_dir, transform=transform)
val_dataset = LungDataset(image_dir, mask_dir, transform=transform)
train_dataset.images = train_images
val_dataset.images = val_images

dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize U-Net
unet_model = UNet().to(device)
criterion_segmentation = nn.BCELoss()
optimizer_segmentation = optim.Adam(unet_model.parameters(), lr=0.0001)

# Model saver and logger
unet_saver = ModelSaver(os.path.join(output_dir, "best_unet.pth"))
unet_logger = CSVLogger(os.path.join(output_dir, "unet_training_log.csv"))

# Training loop for U-Net
train_losses = []
val_losses = []
unet_epochs = 100
patience = 5
no_improve_epochs = 0

for epoch in tqdm(range(unet_epochs), desc="U-Net Training", unit="epoch"):
    # Training phase
    unet_model.train()
    running_loss = 0.0
    for images, masks in dataloader_train:
        images, masks = images.to(device), masks.to(device)

        optimizer_segmentation.zero_grad()
        outputs = unet_model(images)
        loss = criterion_segmentation(outputs, masks)
        loss.backward()
        optimizer_segmentation.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader_train)
    train_losses.append(train_loss)

    # Validation phase
    unet_model.eval()
    val_running_loss = 0.0
    all_metrics = {"iou": [], "dice": [], "pixel_accuracy": [], "precision": [], "recall": [], "specificity": []}
    with torch.no_grad():
        for images, masks in dataloader_val:
            images, masks = images.to(device), masks.to(device)
            outputs = unet_model(images)
            loss = criterion_segmentation(outputs, masks)
            val_running_loss += loss.item()

            # Metrics calculation
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
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, IoU: {avg_metrics['iou']:.4f}")

    # Save model if validation loss improves
    if val_loss < unet_saver.best_val_loss:
        unet_saver.save(unet_model.state_dict(), val_loss)
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    # Early stopping
    if no_improve_epochs >= patience:
        print("Early stopping triggered for U-Net.")
        break

    # Log to CSV
    unet_logger.log(epoch + 1, train_loss, val_loss, [avg_metrics["iou"], avg_metrics["dice"], avg_metrics["pixel_accuracy"], avg_metrics["precision"], avg_metrics["recall"], avg_metrics["specificity"]])

# Plot U-Net loss curve
plot_loss_curve(train_losses, val_losses, save_path=os.path.join(output_dir, "unet_loss_curve.jpg"))
print("U-Net training complete.")

########################################
# Phase 2: Train Classifier
########################################
print("Starting Phase 2: Training Classifier")

# Paths for classifier data
cancer_data_path = os.path.join(current_dir, "cancer")
data_dirs = {"train": os.path.join(cancer_data_path, "train"),
             "val": os.path.join(cancer_data_path, "val")}

# Load datasets
datasets = {split: ClassificationDataset(data_dirs[split], transform=transform) for split in ["train", "val"]}
dataloaders = {split: DataLoader(datasets[split], batch_size=32, shuffle=(split == "train")) for split in ["train", "val"]}

cls_num = len(datasets['train'].class_names)

# Load pre-trained U-Net weights
unet_model.load_state_dict(torch.load(unet_saver.filename)["model_state_dict"])
unet_model.eval()
for param in unet_model.parameters():
    param.requires_grad = False

# Initialize Classifier
classifier = DepthwiseClassifier(input_channel=2, output_channel=32, img_size=img_size, classes_num=cls_num).to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(classifier.parameters(), lr=0.0005)

# Model saver and logger
cls_saver = ModelSaver(os.path.join(output_dir, "best_classifier.pth"))
cls_logger = CSVLogger(os.path.join(output_dir, "classifier_training_log.csv"))

# Training loop for classifier
train_losses = []
val_losses = []
best_val_loss = float("inf")
cls_epochs = 100

for epoch in tqdm(range(cls_epochs), desc="Classifier Training", unit="epoch"):
    # Training phase
    classifier.train()
    running_loss = 0.0
    for images, labels in dataloaders["train"]:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            segmented_images = unet_model(images)

        combined_input = torch.cat((images, segmented_images), dim=1)

        optimizer_cls.zero_grad()
        outputs = classifier(combined_input)
        loss = criterion_cls(outputs, labels)
        loss.backward()
        optimizer_cls.step()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloaders["train"])
    train_losses.append(train_loss)

    # Validation phase
    classifier.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloaders["val"]:
            images, labels = images.to(device), labels.to(device)

            segmented_images = unet_model(images)
            combined_input = torch.cat((images, segmented_images), dim=1)
            outputs = classifier(combined_input)
            loss = criterion_cls(outputs, labels)

            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss = running_val_loss / len(dataloaders["val"])
    val_losses.append(val_loss)
    val_accuracy = correct / total

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Save model if validation loss improves
    if val_loss < cls_saver.best_val_loss:
        cls_saver.save(classifier.state_dict(), val_loss)

    # Log to CSV
    cls_logger.log(epoch + 1, train_loss, val_loss, [val_accuracy])

# Plot classifier loss curve
plot_loss_curve(train_losses, val_losses, save_path=os.path.join(output_dir, "classifier_loss_curve.jpg"))
print("Classifier training complete.")
