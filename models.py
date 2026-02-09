import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """Standard CNN without flip feature support for baseline experiments (CIFAR-10)."""
    
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # First convolutional block (3 channels for RGB)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 32 → 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 → 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block (added for CIFAR-10 complexity)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128 → 256
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (32x32 -> 16x16 -> 8x8 -> 4x4 after 3 pools)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # 256 → 512
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # x shape: (batch, 3, 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)  # (batch, 64, 16, 16)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)  # (batch, 128, 8, 8)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)  # (batch, 256, 4, 4)
        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FlipCNN_LateFusion(nn.Module):
    """CNN with flip feature concatenated after convolutional layers (late fusion, CIFAR-10)."""
    
    def __init__(self):
        super(FlipCNN_LateFusion, self).__init__()
        # First convolutional block (3 channels for RGB)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 32 → 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 → 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128 → 256
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers (256*4*4 + 1 for flip feature)
        self.fc1 = nn.Linear(256 * 4 * 4 + 1, 512)  # 256 → 512
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x, flip):
        # x shape: (batch, 3, 32, 32)
        # flip shape: (batch,)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)  # (batch, 64, 16, 16)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)  # (batch, 128, 8, 8)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)  # (batch, 256, 4, 4)
        
        # Flatten and concatenate flip feature
        x = x.view(-1, 256 * 4 * 4)
        flip = flip.unsqueeze(1).float()  # (batch, 1)
        x = torch.cat([x, flip], dim=1)  # (batch, 256*4*4 + 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FlipCNN_EarlyFusion(nn.Module):
    """CNN with flip feature as input channel from the start (early fusion, CIFAR-10)."""
    
    def __init__(self):
        super(FlipCNN_EarlyFusion, self).__init__()
        # First convolutional block (4 channels: 3 RGB + 1 flip)
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # 32 → 64
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Second convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 → 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Third convolutional block
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128 → 256
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # 256 → 512
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x, flip):
        # x shape: (batch, 3, 32, 32)
        # flip shape: (batch,)
        # Expand flip to match image spatial dimensions: (batch, 1, 32, 32)
        flip_channel = flip.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, 1, 32, 32).float()
        # Concatenate: (batch, 4, 32, 32)
        x = torch.cat([x, flip_channel], dim=1)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)  # (batch, 64, 16, 16)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)  # (batch, 128, 8, 8)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)  # (batch, 256, 4, 4)
        
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

