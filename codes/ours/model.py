"""
Custom Deep Learning Model with Feature Fusion for COVID-19 Classification

This module defines:
1. `MyModel`: A DenseNet121-based model with a custom feature fusion mechanism for COVID-19 image classification.
2. `Fusion`: A multi-scale feature fusion module that integrates features from different network depths.
3. `weight_init`: A utility function for proper weight initialization to improve training stability.

The architecture is designed to effectively classify chest X-ray/CT images into different categories
(COVID-19, lung opacity, normal, viral pneumonia) by leveraging multi-scale feature representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.densenet import densenet121


class MyModel(nn.Module):
    """
    Custom model based on DenseNet121 with a multi-scale feature fusion module.
    
    This model extracts features at different scales from the DenseNet121 backbone,
    fuses them using a custom fusion module, and then classifies the fused features.
    The multi-scale approach helps capture both fine-grained details and global context
    in medical images, which is crucial for accurate diagnosis.

    Args:
        n_out (int): Number of output classes (4 for COVID-19 classification).
        backbone (str): Name of the backbone model (default: "densenet121").
    """
    
    def __init__(self, n_out, backbone="densenet121"):
        super(MyModel, self).__init__()
        self.backbone = densenet121(pretrained=True)
        self.backbone.classifier = nn.Linear(1024, n_out, bias=True)  # Adjusting the classifier
        self.fusion = Fusion(channels=[128, 256, 512, 1024])  # Multi-scale feature fusion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        The model extracts multi-scale features from the backbone network,
        fuses them using the fusion module, applies global average pooling,
        and finally classifies the pooled features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Output logits of shape (B, n_out) for classification.
        """
        x1, x2, x3, x4 = self.backbone(x)  # Extract multi-scale features from different network depths
        features = self.fusion(x1, x2, x3, x4)  # Fuse features from different scales
        out = F.relu(features, inplace=True)  # Apply ReLU activation
        out = F.adaptive_avg_pool2d(out, (1, 1))  # Global average pooling to reduce spatial dimensions
        out = torch.flatten(out, 1)  # Flatten to vector (B, C)
        out = self.backbone.classifier(out)  # Apply final classification layer
        return out


class Fusion(nn.Module):
    """
    Feature Fusion Module for multi-scale feature integration.

    This module takes features from different scales of the backbone network,
    transforms them to have the same number of channels, resizes them to the
    same spatial dimensions, and fuses them through element-wise addition.
    
    This approach allows the model to leverage both low-level details (from earlier layers)
    and high-level semantic information (from deeper layers) for better classification.
    
    Args:
        channels (list): List of input channels at different scales [128, 256, 512, 1024].
    """

    def __init__(self, channels: list):
        super(Fusion, self).__init__()

        # 1x1 Conv layers to unify channel dimensions across all feature maps
        self.linear1 = self._make_layer(channels[0], channels[3])  # Transform scale 1 features
        self.linear2 = self._make_layer(channels[1], channels[3])  # Transform scale 2 features
        self.linear3 = self._make_layer(channels[2], channels[3])  # Transform scale 3 features

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for feature fusion.

        1. Transform each feature map to have the same number of channels
        2. Resize all feature maps to match the spatial dimensions of x4
        3. Fuse all feature maps through element-wise addition

        Args:
            x1 (torch.Tensor): Feature map from first scale (shallowest, highest resolution)
            x2 (torch.Tensor): Feature map from second scale
            x3 (torch.Tensor): Feature map from third scale
            x4 (torch.Tensor): Feature map from fourth scale (deepest, lowest resolution)

        Returns:
            torch.Tensor: Fused feature map with enriched multi-scale information
        """
        # Transform channel dimensions
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x3 = self.linear3(x3)

        # Resize all features to match the spatial size of x4 (deepest features)
        x1 = F.interpolate(x1, size=x4.shape[2:], mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=x4.shape[2:], mode="bilinear", align_corners=False)
        x3 = F.interpolate(x3, size=x4.shape[2:], mode="bilinear", align_corners=False)

        # Element-wise sum fusion - simple but effective approach to combine features
        return x1 + x2 + x3 + x4

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int) -> nn.Sequential:
        """
        Creates a 1x1 convolutional layer with batch normalization and ReLU activation.
        
        This is used to transform feature maps to have the same number of channels
        before fusion, while preserving spatial information.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Returns:
            nn.Sequential: A sequential layer for feature transformation
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),  # 1x1 conv to change channels
            nn.BatchNorm2d(out_channels),  # Normalize features
            nn.ReLU(inplace=True)  # Non-linearity
        )

    def initialize(self):
        """
        Initializes the weights of the Fusion module using the weight_init function.
        Proper initialization helps with faster convergence during training.
        """
        weight_init(self)


def weight_init(module: nn.Module):
    """
    Initializes the weights of different layers in the model using appropriate
    initialization methods for each layer type.
    
    Proper weight initialization is crucial for stable and efficient training
    of deep neural networks.

    Args:
        module (nn.Module): The module to initialize
    """
    for name, m in module.named_children():
        print(f"Initializing: {name}")

        if isinstance(m, nn.Conv2d):
            # Kaiming initialization for convolutional layers - well-suited for ReLU activations
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Initialize biases to zero

        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            # Standard initialization for normalization layers
            if m.weight is not None:
                nn.init.ones_(m.weight)  # Scale initialized to 1
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Bias initialized to 0

        elif isinstance(m, nn.Linear):
            # Kaiming initialization for linear layers
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # Initialize biases to zero

        elif isinstance(m, nn.Sequential):
            weight_init(m)  # Recursively initialize layers inside Sequential containers

        elif isinstance(m, (nn.ReLU, nn.PReLU)):  # Activation functions don't need initialization
            pass

        else:
            # For custom layers that have an initialize method
            if hasattr(m, 'initialize'):
                m.initialize()
