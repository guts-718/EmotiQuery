# src/cv/model_resnet.py

import torch
import torch.nn as nn
from torchvision import models

class EmotionResNet18(nn.Module):
    """
    ResNet18-based model adapted for grayscale emotion recognition.

    - Uses ImageNet-pretrained weights
    - First conv layer modified to accept 1-channel input
    - Final fully-connected layer outputs `num_classes` logits
    """

    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()

        # Load a pretrained ResNet18
        # For torchvision >= 0.13, the 'weights' argument is used
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.backbone = models.resnet18(pretrained=pretrained)

        # Modify first conv layer for 1-channel grayscale input
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Initialize new conv1 weights by averaging over the RGB channels
        with torch.no_grad():
            if hasattr(old_conv.weight, "data"):
                self.backbone.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)

        # Replace the final FC layer to match num_classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    # Sanity check
    model = EmotionResNet18(num_classes=7, pretrained=False)
    dummy = torch.randn(4, 1, 48, 48)
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (4, 7)
