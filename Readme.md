# README: Lung Segmentation and Classification Pipeline

This repository contains a comprehensive pipeline for performing lung segmentation and disease classification using convolutional neural networks (CNNs). The project is divided into several stages, including preprocessing, U-Net segmentation training, classification training, and testing.

---

## **1. Preprocessing**
The preprocessing stage handles the preparation of the datasets for segmentation and classification tasks.

### **Datasets**
- **Lung Segmentation Dataset**: Used to train the U-Net model for lung segmentation.
- **Chest X-ray Pneumonia Dataset**: Used to train the classifier for detecting diseases (e.g., pneumonia).

### **Steps**
1. Download datasets using Kaggle API:
    - Lung segmentation dataset: `iamtapendu/chest-x-ray-lungs-segmentation`
    - Pneumonia classification dataset: `paultimothymooney/chest-xray-pneumonia`
2. Save datasets locally under the `datasets` directory.
3. Reorganize the classification dataset to split `NORMAL` and `PNEUMONIA` images into `train` and `val` directories using the script in `preprocess.py`.

### **Command**
Run preprocessing with:
```bash
python preprocess.py
```

---

## **2. U-Net Segmentation Training**
The U-Net model is trained to segment lungs from chest X-ray images.

### **Script**
`train_mask.py`

### **Key Features**
- Splits dataset into training and validation sets.
- Logs training and validation losses, IoU, Dice Coefficient, and other metrics.
- Implements early stopping and learning rate adjustment.
- Saves the best model weights to `output/best_unet.pth`.

### **Command**
Train the U-Net model with:
```bash
python train_mask.py
```

---

## **3. Classification Training**
The classification model uses the U-Net segmentation output and the original chest X-ray to classify images as `NORMAL` or `PNEUMONIA`.

### **Script**
`train_cls.py`

### **Key Features**
- Loads pre-trained U-Net weights for lung segmentation.
- Combines original images and segmentation masks as inputs to the classifier.
- Logs metrics like loss, accuracy, AUC, precision, and recall.
- Saves the best model weights to `output/best_classifier.pth`.

### **Command**
Train the classifier with:
```bash
python train_cls.py
```

---

## **4. End-to-End Training Pipeline**
To train both U-Net and classifier models in a single script, use:

### **Script**
`train.py`

### **Features**
- Phase 1: Trains U-Net for segmentation.
- Phase 2: Trains classifier using U-Net outputs.
- Combines all features of `train_mask.py` and `train_cls.py`.

### **Command**
Run the end-to-end training pipeline with:
```bash
python train.py
```

---

## **5. Testing**
The testing phase evaluates the performance of the U-Net model by overlaying predicted masks onto original images.

### **Script**
`test.py`

### **Key Features**
- Randomly selects an image from the dataset.
- Uses the trained U-Net model to generate segmentation masks.
- Overlays the predicted mask and ground truth onto the original image.
- Displays results using `matplotlib`.

### **Command**
Test the U-Net model with:
```bash
python test.py
```

---

## **6. Evaluation**
Evaluation metrics for segmentation and classification tasks are implemented in `eval_util.py`.

### **Metrics**
- **Segmentation**: IoU, Dice Coefficient, Pixel Accuracy, Precision, Recall, Specificity, Boundary IoU.
- **Classification**: Accuracy, AUC, Precision, Recall.

These metrics are logged during training and saved to CSV files for analysis.

---

## **File Structure**
```
.
├── datasets/                     # Preprocessed datasets
├── output/                       # Saved models and training logs
├── preprocess.py                 # Preprocessing script
├── train_mask.py                 # U-Net training script
├── train_cls.py                  # Classifier training script
├── train.py                      # Combined training pipeline
├── test.py                       # Testing script for U-Net
├── tools/
│   ├── model.py                  # Model definitions (U-Net, Classifier)
│   ├── eval_util.py              # Evaluation metrics
│   ├── record.py                 # Logging utilities
```

---

## **Dependencies**
Install required dependencies using:
```bash
pip install -r requirements.txt
```

---

## **Results**
1. **U-Net Segmentation**:
    - Visualize segmentation results with `test.py`.
2. **Classification**:
    - Analyze metrics like accuracy, AUC, and loss from logs in `output/training_log.csv`.

---

For further assistance, feel free to raise an issue or reach out to the maintainers!

