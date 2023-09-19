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

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
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

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)

    print(f'Final Evaluation: '
          f'Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, '
          f'F1 Score: {f1:.4f}, '
          f'Accuracy: {accuracy:.4f}')
    return train_loss_values, train_acc_values, test_acc_values


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python DLScript.py <filename>")
    else:
        filename = sys.argv[1]
        data_frame = pd.read_csv(filename)
        data_frame = data_frame.set_index('genome_ID')
        features = data_frame.iloc[:, :-1]
        label_mapping = {'Clinical': 1, 'Non_clinical': 0}
        data_frame['Label_numerical'] = data_frame['Label'].map(label_mapping)
        features = data_frame.iloc[:, :-2].values  
        labels = data_frame.iloc[:, -1].values    

        # Define hyperparameters
        input_size = features.shape[1]
        output_size = 2
        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 100
        batch_size = 64  # Adjust this value according to your preference

# Define KFold cross-validation
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Lists to store results of each fold
        all_train_loss_values = []
        all_train_acc_values = []
        all_test_acc_values = []

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
            train_loss_values, train_acc_values, test_acc_values = ModelEvaluator(model, trainloader, valloader, criterion, optimizer, num_epochs)

            all_train_loss_values.append(train_loss_values[-1])
            all_train_acc_values.append(train_acc_values[-1])
            all_test_acc_values.append(test_acc_values[-1])

# Average results
        avg_train_loss = np.mean(all_train_loss_values, axis=0)
        avg_train_acc = np.mean(all_train_acc_values, axis=0)
        avg_test_acc = np.mean(all_test_acc_values, axis=0)

        print(f"Average Training Loss: {avg_train_loss}")
        print(f"Average Training Accuracy: {avg_train_acc}%")
        print(f"Average Validation Accuracy: {avg_test_acc}%")
