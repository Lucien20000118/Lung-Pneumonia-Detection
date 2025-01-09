import os
import time 
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tools.model import UNet, DepthwiseClassifier, ClassificationDataset
from tools.record import plot_loss_curve, ModelSaver
from tqdm import tqdm
from PIL import Image
import csv
from sklearn.metrics import roc_auc_score, precision_score, recall_score

# Paths and Hyperparameters
current_dir = os.getcwd()
cancer_data_path = os.path.join(current_dir, "cancer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 128
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# Load datasets
data_dirs = {"train": os.path.join(cancer_data_path, "train"),
             "val": os.path.join(cancer_data_path, "val"),
             "test": os.path.join(cancer_data_path, "test")}

datasets = {split: ClassificationDataset(data_dirs[split], transform=transform) for split in ["train", "val", "test"]}
dataloaders = {split: DataLoader(datasets[split], batch_size=32, shuffle=(split == "train")) for split in ["train", "val"]}

cls_num = len(datasets['train'].class_names)
# Load pre-trained UNet weights
print("Loading pre-trained UNet model...")
unet_model = UNet().to(device)
checkpoint = torch.load("output/best_unet.pth", map_location=torch.device(device))
unet_model.eval()
for param in unet_model.parameters():
    param.requires_grad = False

# Initialize Classifier
classifier = DepthwiseClassifier(input_channel = 2, output_channel = 32, img_size = img_size, classes_num = cls_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.0005)

# Model saver
model_saver = ModelSaver("output/best_classifier_small.pth")

# Initialize CSV logger
csv_file = "output/training_log.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Validation Accuracy", "AUC", "Precision", "Recall"])

# Training parameters
epochs = 100
train_losses = []
val_losses = []
best_val_loss = float("inf")

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_start_time = time.time()

    # Training phase
    classifier.train()
    running_train_loss = 0.0
    with tqdm(dataloaders["train"], desc="Training", unit="batch") as t:
        for images, labels in t:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                segmented_images = unet_model(images)

            combined_input = torch.cat((images, segmented_images), dim=1)

            optimizer.zero_grad()
            outputs = classifier(combined_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

    train_loss = running_train_loss / len(dataloaders["train"])
    train_losses.append(train_loss)

    # Validation phase
    classifier.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in dataloaders["val"]:
            images, labels = images.to(device), labels.to(device)

            segmented_images = unet_model(images)
            combined_input = torch.cat((images, segmented_images), dim=1)
            outputs = classifier(combined_input)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

    val_loss = running_val_loss / len(dataloaders["val"])
    val_losses.append(val_loss)
    val_accuracy = correct / total
    auc = roc_auc_score(all_labels, all_probabilities)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)

    print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, AUC: {auc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model_saver.save(classifier.state_dict(), val_loss)

    # Log results to CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch + 1, train_loss, val_loss, val_accuracy, auc, precision, recall])

    # Update loss curve
    plot_loss_curve(train_losses, val_losses, save_path="output/cls_loss_curve.jpg")

    epoch_end_time = time.time()
    print(f"Epoch completed in {epoch_end_time - epoch_start_time:.2f} seconds")

print("Training complete: Best Classifier model saved.")
