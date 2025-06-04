import torch
from torch import nn


class AlexNet(nn.Module):
    """
    AlexNet
    """

    def __init__(self, num_classes=200):  # Tiny ImageNet has 200 classes
        super(AlexNet, self).__init__()

        # Define the layers in AlexNet
        self.features = nn.Sequential(
            # Input size: (batch_size, 3, 224, 224)
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output (batch_size, 256, 6, 6)
        )

        # Fully connected layers (adjusted for 224x224 input size)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # 256 * 6 * 6 for 224x224 input size
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # num_classes=200 for Tiny ImageNet
        )

    def forward(self, x):
        x = self.features(x)  # Pass input through feature extraction layers
        x = torch.flatten(x, 1)  # Flatten the output (adjusted for 224x224)
        x = self.classifier(x)  # Pass through fully connected layers
        return x
