"""
Swin Transformer V2 model for DeepFake detection
"""
import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple

class SwinTransformerV2(nn.Module):
    """
    Swin Transformer V2 model for DeepFake detection
    
    This model uses a pretrained Swin Transformer V2 backbone from the timm library
    and adds a custom classification head for binary classification.
    """
    def __init__(
        self, 
        num_classes: int = 2, 
        pretrained: bool = True,
        model_name: str = 'swinv2_tiny_window16_256',
        img_size: int = 224,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        """
        Initialize the Swin Transformer V2 model
        
        Args:
            num_classes: Number of output classes (2 for binary classification)
            pretrained: Whether to use pretrained weights
            model_name: Name of the Swin Transformer V2 model from timm
            img_size: Input image size
            dropout: Dropout rate for the classification head
            freeze_backbone: Whether to freeze the backbone weights
        """
        super(SwinTransformerV2, self).__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            num_classes=0,  # Remove classification head
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
        
        print(f"Initialized Swin Transformer V2 model with:")
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
    
    def get_attention_maps(
        self, 
        x: torch.Tensor, 
        layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention maps from the model for visualization
        
        Args:
            x: Input tensor
            layer_idx: Index of the layer to get attention maps from (None for all layers)
            
        Returns:
            Tuple of (output, attention_maps)
        """
        # This is a simplified implementation - actual attention map extraction
        # would require modifications to the timm model
        features = self.backbone(x)
        output = self.classifier(features)
        
        # Placeholder for attention maps
        # In a real implementation, you would extract these from the model
        attention_maps = torch.zeros(
            (x.shape[0], 1, x.shape[2]//32, x.shape[3]//32), 
            device=x.device
        )
        
        return output, attention_maps

def create_swin_transformer_v2(
    num_classes: int = 2, 
    pretrained: bool = True,
    model_size: str = 'tiny',
    img_size: int = 224,
    dropout: float = 0.2,
    freeze_backbone: bool = False
) -> SwinTransformerV2:
    """
    Create a Swin Transformer V2 model with the specified parameters
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        model_size: Size of the model ('tiny', 'small', 'base', 'large')
        img_size: Input image size
        dropout: Dropout rate for the classification head
        freeze_backbone: Whether to freeze the backbone weights
        
    Returns:
        Initialized SwinTransformerV2 model
    """
    # Map model size to timm model name
    model_name_map = {
        'tiny': 'swinv2_tiny_window16_256',
        'small': 'swinv2_small_window16_256',
        'base': 'swinv2_base_window16_256',
        'large': 'swinv2_large_window16_256'
    }
    
    if model_size not in model_name_map:
        raise ValueError(f"Invalid model size: {model_size}. Must be one of {list(model_name_map.keys())}")
    
    model_name = model_name_map[model_size]
    
    return SwinTransformerV2(
        num_classes=num_classes,
        pretrained=pretrained,
        model_name=model_name,
        img_size=img_size,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
