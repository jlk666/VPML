import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CustomMLP(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob=0.6):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1400)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(1400, 512)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(512, 128)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x
