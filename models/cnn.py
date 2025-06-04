from torch import nn


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        # Define layers of the neural network
        self.conv_layers = nn.Sequential(
            # Feature extraction
            nn.Conv2d(
                3, 64, kernel_size=3, padding=1
            ),  # (3, 224, 224) -> (64, 224, 224)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 224, 224) -> (64, 112, 112)
            nn.Conv2d(
                64, 128, kernel_size=3, padding=1
            ),  # (64, 112, 112) -> (128, 112, 112)
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 112, 112) -> (128, 56, 56)
            nn.Conv2d(
                128, 256, kernel_size=3, padding=1
            ),  # (128, 56, 56) -> (256, 56, 56)
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 56, 56) -> (256, 28, 28)
            nn.Conv2d(
                256, 512, kernel_size=3, padding=1
            ),  # (256, 28, 28) -> (512, 28, 28)
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (512, 28, 28) -> (512, 14, 14)
        )

        # Global Average Pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # Reduce each 512x14x14 to 512x1x1

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 1024),  # Flattened from 512x1x1 to 512
            nn.ReLU(),
            nn.Dropout(0.5),  # prevents overfitting
            nn.Linear(1024, 200),  # 200 is the number of classes in TinyImageNet
        )

    def forward(self, x):
        # Feature extraction
        x = self.conv_layers(x)

        # Global Average Pooling
        x = self.global_pool(x)

        # Flatten the output from the pooling layer
        x = x.view(x.size(0), -1)

        # Fully connected layers
        logits = self.fc_layers(x)
        return logits
