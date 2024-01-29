import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ------construct CNN with residual learning structure----------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Handle dimension change for residual connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomCNN(nn.Module):
    def __init__(self, input_channels, num_classes, dropout_prob=0.4):
        super(CustomCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        self.fc1 = nn.Linear(409600, 1400)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(1400, 512)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(512, 128)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x, return_feature_maps=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if return_feature_maps:
            # Return the feature maps after the last convolutional layer
            return x
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        
        return x
    class InterpretableResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, stride=1):
        super(InterpretableResidualBlock, self).__init__(in_channels, out_channels, stride)
        # Additional layers for interpretability
        self.interp_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.interp_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Standard residual block forward
        out = super(InterpretableResidualBlock, self).forward(x)
        # Interpretability forward
        interp_out = F.relu(self.interp_bn(self.interp_conv(out)))
        return out, interp_out

class InterpretableCNN(CustomCNN):
    def __init__(self, input_channels, num_classes, dropout_prob=0.4):
        super(InterpretableCNN, self).__init__(input_channels, num_classes, dropout_prob)
        # Replace standard residual blocks with interpretable ones
        self.layer2 = InterpretableResidualBlock(64, 128, stride=2)
        self.layer3 = InterpretableResidualBlock(128, 256, stride=2)

    def forward(self, x):
        x = self.layer1(x)
        x, interp2 = self.layer2(x)
        x, interp3 = self.layer3(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  
        
        # Standard forward
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)

        # Return both standard and interpretability outputs
        return x, (interp2, interp3)

# Example use
model = InterpretableCNN(input_channels=3, num_classes=10)
output, interpretability_maps = model(input_tensor)
