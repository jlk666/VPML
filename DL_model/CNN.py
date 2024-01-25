# -------CNN based predictor--------
import sys
import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from PanGeo_image import load_and_process_data

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt



class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        feature = self.features[index].clone().detach()
        feature = feature.to(dtype=torch.float32)
    
        label = self.labels[index].clone().detach()
        label = label.to(dtype=torch.int64)
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
        
        self.fc1 = nn.Linear(409600, 1400)
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
    best_model = CustomCNN(input_channels=1, num_classes=2)  
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
    
    wandb.log({"Final Precision ": precision, 
           "Final Recall": recall, 
           "Final F1 Score": f1, 
           "Final Accuracy": accuracy})
    
    return precision, recall, f1, accuracy, fpr, tpr, auc_score


# Genome image constructure (aka "QR code" of micrbial genome)
if __name__ == "__main__":
    wandb.init(project='VPML', name='CNN_full_genome_matrix', entity='zsliu')
    

    if len(sys.argv) != 2:
        print("Usage: python DLScript.py <filename>")
        
    else:
        #Check GPU availability first 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file = sys.argv[1]
        image_matrices, labels_array = load_and_process_data(file)

        # Convert the list of image matrices back to a NumPy array if needed

        image_matrices = np.array(image_matrices)
        image_matrices = image_matrices[:, np.newaxis, :, :]
        image_tensor = torch.tensor(image_matrices, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_array, dtype=torch.long)  # Convert labels to a torch tensor of type long

    # Define hyperparameters
        output_size = 2
        learning_rate = 0.001
        momentum = 0.9
        num_epochs = 100
        batch_size = 256 

        wandb.config = {"learning_rate": learning_rate, "epochs": num_epochs, "batch_size": batch_size}

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
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=39)

        for fold, (train_valid_idx, test_idx) in enumerate(kf.split(image_tensor, labels_tensor)):
            print(f'Fold {fold + 1}/{k_folds}')

            features_train_valid, labels_train_valid = image_tensor[train_valid_idx], labels_tensor[train_valid_idx]
            features_test, labels_test = image_tensor[test_idx], labels_tensor[test_idx]

            X_train, X_valid, y_train, y_valid = train_test_split(features_train_valid, labels_train_valid, test_size= 1/9, random_state=39)
            
            # Create datasets for this fold
            train_dataset = CustomDataset(X_train, y_train)
            val_dataset = CustomDataset(X_valid, y_valid)
            test_dataset = CustomDataset(features_test, labels_test)

            # Instantiate the model
            model = CustomCNN(input_channels=1, num_classes=2)
            model = model.to(device)  # move the model to GPU

            # Create DataLoaders for this fold
            trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

             # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

            precision, recall, f1, accuracy, fpr, tpr, auc_score = ModelEvaluator(model, trainloader, testloader, valloader, criterion, optimizer, device, num_epochs)

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

        wandb.log({"Average precision ": avg_train_precision, 
                   "Std precision ": std_train_precision,
           "Average recall": avg_train_recall, 
           "Std recall ": std_train_recall,
           "Average f1": avg_test_f1, 
           "Std f1 ": std_test_f1,
           "Average accuracy:": avg_test_accuracy,
           "Std accuracy:": std_test_accuracy,
           })

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
        plt.savefig('roc_curve_best_fold_cnn.png', dpi=300)  # Save as PNG with high resolution

        wandb.log({
           "Best AUC Score": best_auc})

        with open('roc_parameters_CNN.txt', 'w') as file:
            file.write("Best False Positive Rate (FPR):\n")
            file.write(str(best_fpr))
            file.write("\n\nBest True Positive Rate (TPR):\n")
            file.write(str(best_tpr))
            file.write("\n\nBest AUC Scores\n")
            file.write(str(best_auc))

