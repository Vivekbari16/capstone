"""
Final optimized ensemble model for DeepFake detection that combines multiple approaches
to achieve the highest possible accuracy
"""
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime

# Model classes that match the saved model architectures
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetModel, self).__init__()
        self.features = models.efficientnet_b0(weights=None).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CoAtNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CoAtNetModel, self).__init__()
        self.base = EfficientNetModel(num_classes)
    
    def forward(self, x):
        return self.base(x)

class SwinTransformerModel(nn.Module):
    def __init__(self, num_classes=2):
        super(SwinTransformerModel, self).__init__()
        self.base = models.resnet50(weights=None)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class YOLOModel(nn.Module):
    def __init__(self, num_classes=2):
        super(YOLOModel, self).__init__()
        self.base = models.densenet121(weights=None)
        num_ftrs = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_ftrs, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

class FinalEnsemble(nn.Module):
    def __init__(self, models, weights=None, temperature=1.0):
        super(FinalEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        
        # Initialize weights if not provided
        if weights is None:
            self.weights = torch.ones(len(models), dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()
        else:
            # Ensure weights is a float tensor on the correct device
            self.weights = torch.tensor(weights, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()
        self.weights = self.weights.to(next(self.models.parameters()).device)
        
        # Temperature for calibration
        self.temperature = temperature
        
        # Adaptive thresholds for different confidence ranges
        self.high_conf_threshold = 0.5899999999999999  # For high confidence predictions
        self.mid_conf_threshold = 0.5399999999999998   # For medium confidence predictions
        self.low_conf_threshold = 0.4899999999999998   # For low confidence predictions
        
        # Confidence boundaries
        self.high_conf_boundary = 0.7
        self.low_conf_boundary = 0.3
    
    def forward(self, x):
        all_outputs = []
        individual_outputs = []
        
        for i, model in enumerate(self.models):
            outputs = model(x)
            # Apply temperature scaling for better calibration
            outputs = outputs / self.temperature
            probs = torch.softmax(outputs, dim=1)
            # Ensure weight is a scalar float tensor
            weight = self.weights[i].item() if hasattr(self.weights[i], 'item') else float(self.weights[i])
            all_outputs.append(probs * weight)
            individual_outputs.append(probs)
        
        # Sum the weighted probabilities using torch.stack for safety
        ensemble_probs = torch.stack(all_outputs).sum(dim=0)
        return ensemble_probs, individual_outputs
    
    def predict(self, x):
        """Advanced prediction with adaptive thresholds and test-time augmentation"""
        # Apply test-time augmentation
        # 1. Original image
        outputs, _ = self.forward(x)  # Ensure we get the ensemble probs, not individual outputs
        
        # 2. Horizontally flipped image
        flipped_x = torch.flip(x, [3])
        flipped_outputs, _ = self.forward(flipped_x)  # Ensure we get the ensemble probs
        
        # Combine predictions with equal weights - ensure both are tensors
        combined_outputs = (outputs + flipped_outputs) / 2.0
        
        # Get probabilities directly (outputs from forward are already softmaxed)
        probabilities = combined_outputs
        
        # Get confidence
        confidence, predicted = torch.max(probabilities, dim=1)
        
        # Apply adaptive thresholding based on confidence level
        # Higher confidence predictions can use more aggressive thresholds
        thresholds = torch.ones_like(confidence)
        
        # Apply different thresholds based on confidence ranges
        high_conf_mask = confidence >= self.high_conf_boundary
        low_conf_mask = confidence <= self.low_conf_boundary
        mid_conf_mask = ~(high_conf_mask | low_conf_mask)
        
        # Assign thresholds based on confidence
        thresholds[high_conf_mask] = self.high_conf_threshold
        thresholds[mid_conf_mask] = self.mid_conf_threshold
        thresholds[low_conf_mask] = self.low_conf_threshold
        
        # Make final decision based on thresholds
        # If fake probability > threshold, predict fake (1), otherwise real (0)
        fake_probs = probabilities[:, 1]
        predictions = (fake_probs > thresholds).long()
        
        return predictions, probabilities
