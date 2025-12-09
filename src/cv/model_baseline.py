# src/cv/model_baseline.py
# Input: 1 × 48 × 48  (grayscale face image)
# Output: 7 logits (scores → softmax)

import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    """
    Simple CNN for facial emotion recognition on grayscale images.

    Input shape: (batch_size, 1, H, W)
    Output: logits of shape (batch_size, num_classes)
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 1 -> 32 channels
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/2, W/2

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/4, W/4

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # H/8, W/8
        )

        # Adaptive pooling to avoid manual shape calculations
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # output: (128, 4, 4)

        # Fully-connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    
    model = EmotionCNN(num_classes=7)
    dummy = torch.randn(4, 1, 48, 48)  
    out = model(dummy)
    print("Output shape:", out.shape)  # expect (4, 7)
