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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        self.fc1 = nn.Linear(16384, 1400)
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
        

        
def ModelEvaluator(model, trainloader, testloader, criterion, optimizer, num_epochs=100):
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
            inputs, labels = inputs.to(device), labels.to(device)

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
                images, labels = images.to(device), labels.to(device)

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
    return precision, recall, f1, accuracy

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

        num_sample = data_frame.shape[0]
        side_length = np.sqrt(data_frame.shape[1])
        side_length = int(np.ceil(side_length))
        genome_image_shape = side_length * side_length
        genome_image_shape  

        # Number of columns to be padded with zeros
        padding_columns = genome_image_shape - features_array.shape[1]

        # Padding
        features_array_padded = np.pad(features_array, ((0, 0), (0, padding_columns)), mode='constant', constant_values=0)



        # Create a random feature array for demonstration purposes
        features_array = np.random.rand(side_length, side_length)

        # Reshape each row into a (64, 66) matrix
        image_matrices = []

        for i in range(num_sample):
            sample_image = features_array_padded[i].reshape(side_length, side_length)
            image_matrices.append(sample_image)

        image_matrices = np.array(image_matrices)
        print(image_matrices.shape)

        # Convert the list of image matrices back to a NumPy array if needed

        image_matrices = np.array(image_matrices)
        image_matrices = image_matrices[:, np.newaxis, :, :]
        image_tensor = torch.tensor(image_matrices, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_array, dtype=torch.long)  # Convert labels to a torch tensor of type long
        
        print(image_tensor.shape)
        print(labels_tensor.shape)

    # Define hyperparameters
        input_size = features.shape[1]
        output_size = 2
        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 100
        batch_size = 64  # Adjust this value according to your preference

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

            # Create DataLoaders for this fold
            trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
             # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

            precision, recall, f1, accuracy = ModelEvaluator(model, trainloader, valloader, criterion, optimizer, num_epochs)

            precision_kfold.append(precision)
            recall_kfold.append(recall)
            f1_kfold.append(f1)
            accuracy_kfold.append(accuracy)

# Average results
        avg_train_precision = np.mean(precision_kfold, axis=0)
        avg_train_recall = np.mean(recall_kfold, axis=0)
        avg_test_f1 = np.mean(f1_kfold, axis=0)
        avg_test_accuracy = np.mean(accuracy_kfold, axis=0)

        std_train_precision = np.std(precision_kfold, axis=0)
        std_train_recall = np.std(recall_kfold, axis=0)
        std_test_f1 = np.std(f1_kfold, axis=0)
        std_test_accuracy = np.std(accuracy_kfold, axis=0)

        print(f"Average precision: {avg_train_precision}")
        print(f"Standard deviation precision: {std_train_precision}")
        print(f"Average recall: {avg_train_recall}%")
        print(f"Standard deviation recall: {std_train_recall}%")
        print(f"Average f1: {avg_test_f1}%")
        print(f"Standard deviation f1: {std_test_f1}%")
        print(f"Average accuracy: {avg_test_accuracy}%")
        print(f"Standard deviation accuracy: {std_test_accuracy}%")
