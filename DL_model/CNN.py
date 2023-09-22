# -------CNN based predictor--------
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# ------customer dataset----------
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = torch.tensor(self.features[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.int64)
        return feature, label

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
        )
        
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        self.fc1 = nn.Linear(69632, 1400)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(1400, 512)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(512, 128)
        self.dropout3 = nn.Dropout(dropout_prob)
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
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

# Genome image constructure (aka "QR code" of micrbial genome)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python DLScript.py <filename>")
    else:
        #Check GPU availability first 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        filename = sys.argv[1]
        data_frame = pd.read_csv(filename)
        data_frame = data_frame.set_index('genome_ID')
        features = data_frame.iloc[:, :-1]
        label_mapping = {'Clinical': 1, 'Non_clinical': 0}
        data_frame['Label_numerical'] = data_frame['Label'].map(label_mapping)
        labels = data_frame.iloc[:, -1] 

        features_array = features.values
        labels_array = labels.values

        # Number of columns to be padded with zeros
        padding_columns = 4224 - features_array.shape[1]

        # Padding
        features_array_padded = np.pad(features_array, ((0, 0), (0, padding_columns)), mode='constant', constant_values=0)

        # Sample data (replace this with your actual data)
        num_samples = 1980
        num_features = 4224
        image_width = 64
        image_height = 66  # You can adjust this as needed

        # Create a random feature array for demonstration purposes
        features_array = np.random.rand(num_samples, num_features)

        # Reshape each row into a (64, 66) matrix
        image_matrices = []

        for i in range(num_samples):
            sample_image = features_array_padded[i].reshape(image_width, image_height)
            image_matrices.append(sample_image)

        # Convert the list of image matrices back to a NumPy array if needed

        image_matrices = np.array(image_matrices)
        image_matrices = image_matrices[:, np.newaxis, :, :]
        image_tensor = torch.tensor(image_matrices, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_array, dtype=torch.long)  # Convert labels to a torch tensor of type long
        
        print(image_tensor.shape)
        print(labels_tensor.shape)

    # Lists to store results of each fold
        precision_kfold = []
        recall_kfold = []
        f1_kfold = []
        accuracy_kfold = []
        
    # Define KFold cross-validation
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(features, labels)):
            print(f'Fold {fold + 1}/{k_folds}')
            features_train, features_val = image_tensor[train_idx], image_tensor[val_idx]
            labels_train, labels_val = labels_tensor[train_idx], labels_tensor[val_idx]

             # Create datasets for this fold
            train_dataset = CustomDataset(features_train, labels_train)
            val_dataset = CustomDataset(features_val, labels_val)

            # Instantiate the model
            model = CustomCNN(input_channels=1, num_classes=2)
            model = model.to(device)  # move the model to GPU

             # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
