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
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class CustomMLP(nn.Module):
    def __init__(self, dropout_prob=0.6):
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
    print("In this pangenome matrix, you have", data_frame.shape[0], "samples and each having", data_frame.shape[1], "features.")
    return features_array, labels_array

def ModelEvaluator(model, trainloader, testloader, valloader, criterion, optimizer, device, num_epochs=100):
    train_loss_values = []
    train_acc_values = []
    test_acc_values = []

    best_valid_acc = 0.0
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

        # Validation set
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        epoch_valid_acc = 100 * correct / total

        train_loss_values.append(epoch_train_loss)
        train_acc_values.append(epoch_train_acc)
        test_acc_values.append(epoch_valid_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Training Loss: {epoch_train_loss:.4f}, '
              f'Training Accuracy: {epoch_train_acc:.2f}%, '
              f'Validation Accuracy: {epoch_valid_acc:.2f}%')
        
        if epoch_valid_acc > best_valid_acc:
            best_valid_acc = epoch_valid_acc
            torch.save(model.state_dict(), 'best_model.pth')

    print('Finished Training')

    # Reload the best model's parameters
    best_model = CustomMLP()  
    best_model.load_state_dict(torch.load('best_model.pth'))
    best_model.to(device)  


    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    test_probs = []
    test_labels = []

    
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            test_probs.append(outputs.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)

    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)
    fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
    auc_score = roc_auc_score(test_labels, test_probs[:, 1])

    print(f'Final Evaluation: '
          f'Precision: {precision:.4f}, '
          f'Recall: {recall:.4f}, '
          f'F1 Score: {f1:.4f}, '
          f'Accuracy: {accuracy:.4f}')
    
    return precision, recall, f1, accuracy, fpr, tpr, auc_score



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python DLScript.py <filename>")
    else:
        #Check GPU availability first 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        filename = sys.argv[1]
        features, labels = process_genome_matrix(filename)
        print(labels)

        # Define hyperparameters
        input_size = features.shape[1]
        output_size = 2
        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 100
        batch_size = 256  

    # Lists to store results of each fold
        precision_kfold = []
        recall_kfold = []
        f1_kfold = []
        accuracy_kfold = []

        fpr_kfold = []
        tpr_kfold = []
        auc_score_kfold = []

    # Define KFold cross-validation
        k_folds = 10
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=669)

        for fold, (train_valid_idx, test_idx) in enumerate(kf.split(features, labels)):
            print(f'Fold {fold + 1}/{k_folds}')

            features_train_valid, labels_train_valid = features[train_valid_idx], labels[train_valid_idx]
            features_test, labels_test = features[test_idx], labels[test_idx]

            X_train, X_valid, y_train, y_valid = train_test_split(features_train_valid, labels_train_valid, test_size= 1/9, random_state=42)
            
            print(X_train.shape)

            # Create datasets for this fold
            train_dataset = CustomDataset(X_train, y_train)
            val_dataset = CustomDataset(X_valid, y_valid)
            test_dataset = CustomDataset(features_test, labels_test)

            trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # Instantiate the model
            model = CustomMLP()
            model = model.to(device)  # move the model to GPU

    # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Evaluate for this fold
            precision, recall, f1, accuracy, fpr, tpr, auc_score  = ModelEvaluator(model, trainloader, testloader, valloader, criterion, optimizer, device, num_epochs)

            precision_kfold.append(precision)
            recall_kfold.append(recall)
            f1_kfold.append(f1)
            accuracy_kfold.append(accuracy)
            
            fpr_kfold.append(fpr)
            tpr_kfold.append(tpr)
            auc_score_kfold.append(auc_score)


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

        max_auc_index = np.argmax(auc_score_kfold)  # Index of the highest AUC score
        best_fpr = fpr_kfold[max_auc_index]
        best_tpr = tpr_kfold[max_auc_index]
        best_auc = auc_score_kfold[max_auc_index]

        # Plotting the ROC curve for the best fold
        plt.figure()
        plt.plot(best_fpr, best_tpr, color='darkorange', lw=2, label=f'ROC curve (area = {best_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Best Fold')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve_best_fold.png', dpi=300)  # Save as PNG with high resolution

