import torch
from torch import nn
import torch.nn.functional as F

class CustomNet(nn.Module):
    def __init__(self, num_classes=200):
        super(CustomNet, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 224 -> 112

        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 112 -> 56

        # Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 56 -> 28

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        # Fully connected layer - Updated input size
        #self.fc1 = nn.Linear(256 * 28 * 28, num_classes) # Input size = 200704
        self.fc1 = nn.Linear(256, num_classes)
    def forward(self, x):
        # Input: B x 3 x 224 x 224
        x = self.pool1(F.relu(self.bn1(self.conv1(x)))) # B x 64 x 112 x 112
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) # B x 128 x 56 x 56
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))) # B x 256 x 28 x 28

        # --- Applicazione delle modifiche ---

        x = self.gap(x)       # B x 256 x 1 x 1
        x = self.flatten(x)   # B x 256
        x = self.dropout(x)   # Applica il dropout prima del layer finale
        x = self.fc1(x)       # B x num_classes
        return x