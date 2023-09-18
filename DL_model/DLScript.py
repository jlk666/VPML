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

def process_genome_matrix(filename):
    data_frame = pd.read_csv(filename)
    data_frame = data_frame.set_index('genome_ID')
    features = data_frame.iloc[:, :-1]
    label_mapping = {'Clinical': 1, 'Non_clinical': 0}
    data_frame['Label_numerical'] = data_frame['Label'].map(label_mapping)
    labels = data_frame.iloc[:, -1] 

    features_array = features.values
    labels_array = labels.values

    #Dont need to do data split as we choose to do cross validation 
    #X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)

    print("In this pangenome matrix, you have", data_frame.shape[0], "samples and each having", data_frame.shape[1], "features.")

    return features_array, labels_array

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

def figurePlot(epoch, train_loss_values, train_acc_values, test_acc_values):
  epochs = range(1, epoch+1)

  plt.figure(figsize=(12, 4))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, train_loss_values, label='Training Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training Loss Curve')
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, train_acc_values, label='Training Accuracy')
  plt.plot(epochs, test_acc_values, label='Test Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.title('Training and Test Accuracy Curves')
  plt.legend()
  plt.ylim(0, 100)

  plt.tight_layout()
  plt.show()


def ModelEvaluator(model, trainloader, testloader, criterion, optimizer, num_epochs=50):
    train_loss_values = []
    train_acc_values = []
    test_acc_values = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        epoch_train_acc = (100 * correct_train) / total_train
        epoch_train_loss = running_loss / len(trainloader)

        # Testing set
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_test_acc = 100 * correct / total

        train_loss_values.append(epoch_train_loss)
        train_acc_values.append(epoch_train_acc)
        test_acc_values.append(epoch_test_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {epoch_train_loss:.4f}, '
              f'Training Accuracy: {epoch_train_acc:.2f}%, '
              f'Test Accuracy: {epoch_test_acc:.2f}%')

    print('Finished Training')

    return train_loss_values, train_acc_values, test_acc_values

class CustomMLP(nn.Module):
    def __init__(self, dropout_prob=0.4):
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


# Define hyperparameters
input_size = X_train.shape[1]
output_size = 2
learning_rate = 0.001
momentum = 0.9
num_epochs = 100

# Instantiate the model
model = CustomMLP()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python DLScript.py <filename> <model selection>")
    else:
        filename = sys.argv[1]
        model_selection = sys.argv[2]
        model_selection = model_selection.upper()

    # Create instances of your custom dataset for train and test
        train_dataset = CustomDataset(X_train, y_train)
        test_dataset = CustomDataset(X_test, y_test)

    # Define batch size for train and test data
        batch_size = 64  

    # Create DataLoaders for train and test data
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        features = data_frame.iloc[:, :-2].values  # Remove the last two columns for features
        labels = data_frame.iloc[:, -1].values    # Use the last column as labels

        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create instances of your custom dataset for train and test
        train_dataset = CustomDataset(features_train, labels_train)
        test_dataset = CustomDataset(features_test, labels_test)

    # Define batch size for train and test data
        batch_size = 64  # Adjust this value according to your preference

    # Create DataLoaders for train and test data
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Split data into features and labels
        features = data_frame.iloc[:, :-2].values  
        labels = data_frame.iloc[:, -1].values    

    # Define hyperparameters
        input_size = features.shape[1]
        output_size = 2
        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 100

        # Define KFold cross-validation
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Lists to store results of each fold
        all_train_loss_values = []
        all_train_acc_values = []
        all_val_acc_values = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(features, labels)):
            print(f'Fold {fold + 1}/{k_folds}')

        # Split data
        features_train, features_val = features[train_idx], features[val_idx]
        labels_train, labels_val = labels[train_idx], labels[val_idx]

    # Create datasets for this fold
        train_dataset = CustomDataset(features_train, labels_train)
        val_dataset = CustomDataset(features_val, labels_val)

    # Create DataLoaders for this fold
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
        model = CustomMLP()

    # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Evaluate for this fold
        train_loss_values, train_acc_values, val_acc_values = ModelEvaluator(model, trainloader, valloader, criterion, optimizer, num_epochs)

        all_train_loss_values.append(train_loss_values)
        all_train_acc_values.append(train_acc_values)
        all_val_acc_values.append(val_acc_values)

# Average results
        avg_train_loss = np.mean(all_train_loss_values, axis=0)
        avg_train_acc = np.mean(all_train_acc_values, axis=0)
        avg_val_acc = np.mean(all_val_acc_values, axis=0)

# Plot average results
        figurePlot(num_epochs, avg_train_loss, avg_train_acc, avg_val_acc)



