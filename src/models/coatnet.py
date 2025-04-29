"""
CoAtNet model for DeepFake detection

CoAtNet combines convolutional layers and self-attention mechanisms to create
a powerful hybrid architecture that excels at image classification tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

class MBConv(nn.Module):
    """
    MBConv block: Mobile Inverted Bottleneck Convolution
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        expansion_ratio: int = 4,
        stride: int = 1,
        use_se: bool = True,
        drop_path_rate: float = 0.0
    ):
        super(MBConv, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expansion_ratio
        
        # Expansion
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        ) if expansion_ratio != 1 else nn.Identity()
        
        # Depthwise convolution
        self.dwconv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, 3, stride=stride, 
                     padding=1, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )
        
        # Squeeze and Excitation
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, expanded_channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(expanded_channels // 4, expanded_channels, 1),
            nn.Sigmoid()
        ) if use_se else nn.Identity()
        
        # Projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise convolution
        x = self.dwconv(x)
        
        # Squeeze and Excitation
        x = x * self.se(x)
        
        # Projection
        x = self.project_conv(x)
        
        # Skip connection
        if self.use_residual:
            x = self.drop_path(x) + residual
            
        return x

class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super(RelativeMultiHeadAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block with relative multi-head attention
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        mlp_ratio: float = 4.,
        qkv_bias: bool = False, 
        drop: float = 0., 
        attn_drop: float = 0.,
        drop_path: float = 0.
    ):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RelativeMultiHeadAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)
    """
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and activation
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1, 
        padding: int = 1,
        groups: int = 1
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class CoAtNet(nn.Module):
    """
    CoAtNet: Combining Convolution and Attention for image classification
    
    This implementation follows the architecture described in the paper:
    "CoAtNet: Marrying Convolution and Attention for All Data Sizes"
    """
    def __init__(
        self, 
        img_size: int = 224,
        in_channels: int = 3,
        num_classes: int = 2,
        depths: List[int] = [2, 2, 6, 14, 2],
        dims: List[int] = [64, 96, 192, 384, 768],
        drop_path_rate: float = 0.2,
        pretrained: bool = False,
        model_name: str = 'coatnet-0'
    ):
        """
        Initialize CoAtNet model
        
        Args:
            img_size: Input image size
            in_channels: Number of input channels
            num_classes: Number of output classes
            depths: Number of blocks in each stage
            dims: Channel dimensions in each stage
            drop_path_rate: Drop path rate
            pretrained: Whether to use pretrained weights
            model_name: Model name for loading pretrained weights
        """
        super(CoAtNet, self).__init__()
        self.depths = depths
        self.dims = dims
        
        # Stem: Initial convolution
        self.stem = nn.Sequential(
            ConvBlock(in_channels, dims[0], kernel_size=3, stride=2, padding=1),
            ConvBlock(dims[0], dims[0], kernel_size=3, stride=1, padding=1)
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        
        # Stage 1: MBConv blocks
        self.stages.append(self._make_stage(
            dims[0], dims[1], depths[0], 'mbconv', stride=2
        ))
        
        # Stage 2: MBConv blocks
        self.stages.append(self._make_stage(
            dims[1], dims[2], depths[1], 'mbconv', stride=2
        ))
        
        # Stage 3: Transformer blocks
        self.stages.append(self._make_stage(
            dims[2], dims[3], depths[2], 'transformer', stride=2
        ))
        
        # Stage 4: Transformer blocks
        self.stages.append(self._make_stage(
            dims[3], dims[4], depths[3], 'transformer', stride=2
        ))
        
        # Head
        self.norm = nn.LayerNorm(dims[4])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[4], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Load pretrained weights if available
        if pretrained:
            self._load_pretrained(model_name)
    
    def _make_stage(
        self, 
        in_dim: int, 
        out_dim: int, 
        depth: int, 
        block_type: str, 
        stride: int = 1
    ) -> nn.Sequential:
        """
        Create a stage of the CoAtNet model
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension
            depth: Number of blocks
            block_type: Type of block ('mbconv' or 'transformer')
            stride: Stride for the first block
            
        Returns:
            Sequential module containing the blocks
        """
        if block_type == 'mbconv':
            # MBConv stage
            blocks = []
            for i in range(depth):
                s = stride if i == 0 else 1
                blocks.append(MBConv(
                    in_channels=in_dim if i == 0 else out_dim,
                    out_channels=out_dim,
                    expansion_ratio=4,
                    stride=s,
                    use_se=True,
                    drop_path_rate=0.0
                ))
            return nn.Sequential(*blocks)
        
        elif block_type == 'transformer':
            # Transformer stage with downsampling
            blocks = []
            
            # Downsampling using convolution
            if stride > 1:
                blocks.append(nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=stride, stride=stride),
                    nn.BatchNorm2d(out_dim)
                ))
            else:
                blocks.append(nn.Identity())
            
            # Add transformer blocks
            for i in range(depth):
                blocks.append(TransformerBlock(
                    dim=out_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop=0.0,
                    attn_drop=0.0,
                    drop_path=0.0
                ))
            
            return nn.Sequential(*blocks)
        
        else:
            raise ValueError(f"Unknown block type: {block_type}")
    
    def _init_weights(self, m: nn.Module):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def _load_pretrained(self, model_name: str):
        """
        Load pretrained weights
        
        Note: This is a placeholder. In a real implementation, you would load
        pretrained weights from a file or a model zoo.
        """
        print(f"Pretrained weights for {model_name} would be loaded here.")
        print("This is a placeholder - implement actual weight loading for production use.")
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input"""
        # Stem
        x = self.stem(x)
        
        # Stages 1-2: Convolutional (MBConv)
        x = self.stages[0](x)
        x = self.stages[1](x)
        
        # Reshape for transformer blocks
        B, C, H, W = x.shape
        
        # Stages 3-4: Transformer
        # For transformer stages, we need to reshape the tensor
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        for block in self.stages[2]:
            if isinstance(block, TransformerBlock):
                x = block(x)
        
        for block in self.stages[3]:
            if isinstance(block, TransformerBlock):
                x = block(x)
        
        # Apply normalization
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.forward_features(x)
        x = self.head(x)
        return x

def create_coatnet(
    model_name: str = 'coatnet-0',
    num_classes: int = 2,
    pretrained: bool = True,
    img_size: int = 224
) -> CoAtNet:
    """
    Create a CoAtNet model with the specified parameters
    
    Args:
        model_name: Model name ('coatnet-0', 'coatnet-1', 'coatnet-2', 'coatnet-3', 'coatnet-4')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        img_size: Input image size
        
    Returns:
        Initialized CoAtNet model
    """
    # Model configurations
    model_configs = {
        'coatnet-0': {
            'depths': [2, 2, 3, 5, 2],
            'dims': [64, 96, 192, 384, 768]
        },
        'coatnet-1': {
            'depths': [2, 2, 6, 14, 2],
            'dims': [64, 96, 192, 384, 768]
        },
        'coatnet-2': {
            'depths': [2, 2, 6, 14, 2],
            'dims': [128, 128, 256, 512, 1024]
        },
        'coatnet-3': {
            'depths': [2, 2, 6, 14, 2],
            'dims': [192, 192, 384, 768, 1536]
        },
        'coatnet-4': {
            'depths': [2, 2, 12, 28, 2],
            'dims': [192, 192, 384, 768, 1536]
        }
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Invalid model name: {model_name}. Must be one of {list(model_configs.keys())}")
    
    config = model_configs[model_name]
    
    return CoAtNet(
        img_size=img_size,
        in_channels=3,
        num_classes=num_classes,
        depths=config['depths'],
        dims=config['dims'],
        drop_path_rate=0.2,
        pretrained=pretrained,
        model_name=model_name
    )
