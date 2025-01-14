'''
Evaluation tools for segmentation
'''

import cv2
import numpy as np


def iou(pred, target, smooth=1e-6):
    """
    Compute Intersection over Union (IoU) for binary segmentation.
    """
    pred = (pred > 0.5).bool()  # Binarize predictions
    target = (target > 0.5).bool()  # Binarize ground truth

    intersection = (pred & target).sum().float()  # Intersection
    union = (pred | target).sum().float()  # Union

    return (intersection + smooth) / (union + smooth)

def mean_iou(pred, target, num_classes):
    """
    Compute Mean IoU for multi-class segmentation.
    """
    iou_per_class = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        iou_per_class.append((intersection + 1e-6) / (union + 1e-6))
    return sum(iou_per_class) / len(iou_per_class)

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Compute Dice Coefficient (F1 Score) for binary segmentation.
    """
    pred = (pred > 0.5).bool()  # Binarize predictions
    target = (target > 0.5).bool()  # Binarize ground truth

    intersection = (pred & target).sum().float()  # Intersection
    return (2.0 * intersection + smooth) / (pred.sum().float() + target.sum().float() + smooth)

def pixel_accuracy(pred, target):
    """
    Compute Pixel Accuracy for segmentation.
    """
    correct = (pred == target).sum().float()  # Correct pixels
    total = pred.numel()  # Total pixels
    return correct / total

def precision_recall(pred, target):
    """
    Compute Precision and Recall for binary segmentation.
    """
    pred = (pred > 0.5).bool()  # Binarize predictions
    target = (target > 0.5).bool()  # Binarize ground truth

    tp = (pred & target).sum().float()  # True Positives
    fp = (pred & ~target).sum().float()  # False Positives
    fn = (~pred & target).sum().float()  # False Negatives

    precision = tp / (tp + fp + 1e-6)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-6)  # Avoid division by zero
    return precision, recall

def specificity(pred, target):
    """
    Compute Specificity for binary segmentation.
    """
    pred = (pred > 0.5).bool()  # Binarize predictions
    target = (target > 0.5).bool()  # Binarize ground truth

    tn = (~pred & ~target).sum().float()  # True Negatives
    fp = (pred & ~target).sum().float()  # False Positives

    return tn / (tn + fp + 1e-6)  # Avoid division by zero

def boundary_iou(pred, target, dilation=1):
    """
    Compute Boundary IoU for segmentation.
    """
    pred = (pred > 0.5).cpu().numpy().astype(np.uint8)  # Binarize predictions
    target = (target > 0.5).cpu().numpy().astype(np.uint8)  # Binarize ground truth

    # Extract boundaries using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = cv2.dilate(pred, kernel, iterations=dilation) & ~pred
    target_boundary = cv2.dilate(target, kernel, iterations=dilation) & ~target

    # Convert back to boolean arrays
    pred_boundary = pred_boundary.astype(bool)
    target_boundary = target_boundary.astype(bool)

    # Calculate IoU for boundaries
    intersection = (pred_boundary & target_boundary).sum()
    union = (pred_boundary | target_boundary).sum()

    return (intersection + 1e-6) / (union + 1e-6)  # Avoid division by zero
