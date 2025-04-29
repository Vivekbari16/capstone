"""
YOLOv10 backbone adapted for DeepFake binary classification

This module adapts the YOLOv10 architecture from Ultralytics for binary classification tasks.
It leverages the powerful feature extraction capabilities of YOLO while replacing the 
detection head with a classification head.
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Optional, Union, Tuple, Dict, Any

class YOLOv10Classifier(nn.Module):
    """
    YOLOv10 model adapted for binary classification
    
    This model uses a pretrained YOLOv10 backbone from Ultralytics
    and adds a custom classification head for binary classification.
    """
    def __init__(
        self, 
        num_classes: int = 2, 
        pretrained: bool = True,
        model_size: str = 'small',
        img_size: int = 224,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """
        Initialize the YOLOv10 classifier
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use pretrained weights
            model_size: Size of the model ('nano', 'small', 'medium', 'large', 'extra')
            img_size: Input image size
            dropout: Dropout rate for the classification head
            freeze_backbone: Whether to freeze the backbone weights
        """
        super(YOLOv10Classifier, self).__init__()
        
        # Map model size to Ultralytics model name
        model_name_map = {
            'nano': 'yolov10n',
            'small': 'yolov10s',
            'medium': 'yolov10m',
            'large': 'yolov10l',
            'extra': 'yolov10x'
        }
        
        if model_size not in model_name_map:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {list(model_name_map.keys())}")
        
        model_name = model_name_map[model_size]
        
        # Load pretrained YOLO model
        self.backbone = YOLO(model_name + '.pt' if pretrained else model_name + '.yaml')
        
        # Extract the backbone (feature extractor) from YOLO
        self.feature_extractor = self._extract_backbone()
        
        # Get feature dimension from the backbone
        # This is an approximation - actual dimension depends on the specific model
        feature_dims = {
            'nano': 1024,
            'small': 1024,
            'medium': 1280,
            'large': 1536,
            'extra': 2048
        }
        feature_dim = feature_dims[model_size]
        
        # Create classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
        # Initialize the classifier weights
        self._init_classifier_weights()
        
        print(f"Initialized YOLOv10 classifier with:")
        print(f"  Model name: {model_name}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Image size: {img_size}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Frozen backbone: {freeze_backbone}")
    
    def _extract_backbone(self) -> nn.Module:
        """
        Extract the backbone (feature extractor) from the YOLO model
        
        Returns:
            Feature extractor module
        """
        # This is a simplified implementation
        # In a real implementation, you would extract the backbone from the YOLO model
        # and return it as a nn.Module
        
        # For demonstration purposes, we'll use the model up to a certain layer
        # In practice, you would need to analyze the YOLO architecture and extract
        # the appropriate layers for feature extraction
        
        # Access the underlying PyTorch model
        yolo_model = self.backbone.model
        
        # Extract the backbone (up to the detection head)
        # This is a simplified approach - actual implementation would depend on
        # the specific YOLO architecture
        backbone = nn.Sequential(*list(yolo_model.model.children())[:-1])
        
        return backbone
    
    def _init_classifier_weights(self):
        """Initialize the weights of the classifier"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, img_size, img_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

def create_yolov10_classifier(
    num_classes: int = 2, 
    pretrained: bool = True,
    model_size: str = 'small',
    img_size: int = 224,
    dropout: float = 0.2,
    freeze_backbone: bool = False
) -> YOLOv10Classifier:
    """
    Create a YOLOv10 classifier with the specified parameters
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        model_size: Size of the model ('nano', 'small', 'medium', 'large', 'extra')
        img_size: Input image size
        dropout: Dropout rate for the classification head
        freeze_backbone: Whether to freeze the backbone weights
        
    Returns:
        Initialized YOLOv10Classifier model
    """
    return YOLOv10Classifier(
        num_classes=num_classes,
        pretrained=pretrained,
        model_size=model_size,
        img_size=img_size,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )

class YOLOFeatureExtractor(nn.Module):
    """
    Feature extractor for YOLOv10
    
    This is a more detailed implementation of the feature extractor
    that extracts features from specific layers of the YOLO model.
    """
    def __init__(
        self, 
        model_size: str = 'small',
        pretrained: bool = True
    ):
        """
        Initialize the YOLOv10 feature extractor
        
        Args:
            model_size: Size of the model ('nano', 'small', 'medium', 'large', 'extra')
            pretrained: Whether to use pretrained weights
        """
        super(YOLOFeatureExtractor, self).__init__()
        
        # Map model size to Ultralytics model name
        model_name_map = {
            'nano': 'yolov10n',
            'small': 'yolov10s',
            'medium': 'yolov10m',
            'large': 'yolov10l',
            'extra': 'yolov10x'
        }
        
        if model_size not in model_name_map:
            raise ValueError(f"Invalid model size: {model_size}. Must be one of {list(model_name_map.keys())}")
        
        model_name = model_name_map[model_size]
        
        # Load pretrained YOLO model
        self.yolo = YOLO(model_name + '.pt' if pretrained else model_name + '.yaml')
        
        # Get the underlying PyTorch model
        self.model = self.yolo.model
        
        # Define the feature extraction layers
        # These are the layers that output feature maps
        # The exact indices would depend on the specific YOLO architecture
        self.feature_layers = [6, 12, 18]  # Example indices - adjust based on actual architecture
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor
        """
        features = []
        
        # Extract features from specified layers
        for i, module in enumerate(self.model.model):
            x = module(x)
            if i in self.feature_layers:
                features.append(x)
        
        # Return the final feature map
        return features[-1]

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification
    
    This loss function is useful for handling class imbalance.
    It down-weights well-classified examples and focuses on hard examples.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for the rare class
            gamma: Focusing parameter
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            inputs: Predicted logits of shape (batch_size, num_classes)
            targets: Target labels of shape (batch_size,)
            
        Returns:
            Loss value
        """
        # Apply softmax to get probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # Get probability for the target class
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Calculate focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Calculate alpha weight
        alpha_weight = torch.ones_like(targets, device=inputs.device)
        alpha_weight[targets == 0] = 1 - self.alpha  # Weight for class 0
        alpha_weight[targets == 1] = self.alpha  # Weight for class 1
        
        # Calculate loss
        loss = -alpha_weight * focal_weight * torch.log(p_t + self.eps)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
