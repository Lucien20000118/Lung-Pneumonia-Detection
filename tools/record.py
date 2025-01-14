import csv
import torch
import matplotlib.pyplot as plt


class CSVLogger:
    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Training Loss", "Validation Loss", 
                             "IoU", "Dice", "Pixel Accuracy", 
                             "Precision", "Recall", "Specificity"])

    def log(self, epoch, train_loss, val_loss, metrics):
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_loss] + metrics)


class ModelSaver:
    def __init__(self, filename):
        self.filename = filename
        self.best_val_loss = float("inf")

    def save(self, model_state_dict, val_loss):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save({
                'model_state_dict': model_state_dict,
                'best_val_loss': val_loss
            }, self.filename)
            print(f"Best model weights saved to {self.filename} with validation loss {val_loss:.4f}")


# Function to update and save loss curves
def plot_loss_curve(train_losses, val_losses, save_path="losses_record.jpg"):
    plt.clf()
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'r-', label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'b-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.pause(0.1)
    plt.savefig(save_path)
    plt.close()
    
