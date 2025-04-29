"""
EfficientNetV2 model for DeepFake detection

EfficientNetV2 is an improved version of EfficientNet that is faster and more
parameter efficient while achieving better accuracy.
"""
import torch
import torch.nn as nn
import timm
from typing import Optional, Union, List

class EfficientNetV2(nn.Module):
    """
    EfficientNetV2 model for DeepFake detection
    
    This model uses a pretrained EfficientNetV2 backbone from the timm library
    and adds a custom classification head for binary classification.
    """
    def __init__(
        self, 
        num_classes: int = 2, 
        pretrained: bool = True,
        model_name: str = 'efficientnetv2_s',
        img_size: int = 224,
        dropout: float = 0.3,
        drop_path_rate: float = 0.2,
        freeze_backbone: bool = False
    ):
        """
        Initialize the EfficientNetV2 model
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use pretrained weights
            model_name: Name of the EfficientNetV2 model from timm
            img_size: Input image size
            dropout: Dropout rate for the classification head
            drop_path_rate: Drop path rate for the backbone
            freeze_backbone: Whether to freeze the backbone weights
        """
        super(EfficientNetV2, self).__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_path_rate=drop_path_rate
        )
        
        # Get feature dimension from the backbone
        with torch.no_grad():
            # Create a dummy input to get the output dimension
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Create classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Initialize the classifier weights
        self._init_classifier_weights()
        
        print(f"Initialized EfficientNetV2 model with:")
        print(f"  Model name: {model_name}")
        print(f"  Pretrained: {pretrained}")
        print(f"  Image size: {img_size}")
        print(f"  Feature dimension: {feature_dim}")
        print(f"  Frozen backbone: {freeze_backbone}")
    
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
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def create_efficientnetv2(
    num_classes: int = 2, 
    pretrained: bool = True,
    model_size: str = 's',
    img_size: int = 224,
    dropout: float = 0.3,
    drop_path_rate: float = 0.2,
    freeze_backbone: bool = False
) -> EfficientNetV2:
    """
    Create an EfficientNetV2 model with the specified parameters
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        model_size: Size of the model ('s', 'm', 'l', 'xl')
        img_size: Input image size
        dropout: Dropout rate for the classification head
        drop_path_rate: Drop path rate for the backbone
        freeze_backbone: Whether to freeze the backbone weights
        
    Returns:
        Initialized EfficientNetV2 model
    """
    # Map model size to timm model name
    model_name_map = {
        's': 'efficientnetv2_s',
        'm': 'efficientnetv2_m',
        'l': 'efficientnetv2_l',
        'xl': 'efficientnetv2_xl'
    }
    
    if model_size not in model_name_map:
        raise ValueError(f"Invalid model size: {model_size}. Must be one of {list(model_name_map.keys())}")
    
    model_name = model_name_map[model_size]
    
    return EfficientNetV2(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name,
        img_size=img_size,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone
    )

class EfficientNetV2FocalLoss(nn.Module):
    """
    Focal Loss for EfficientNetV2
    
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
        super(EfficientNetV2FocalLoss, self).__init__()
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
        # Convert targets to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # Calculate focal loss
        pt = torch.sum(targets_one_hot * probs, dim=1)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + self.eps)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
