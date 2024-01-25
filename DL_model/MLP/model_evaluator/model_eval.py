import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.metrics import roc_curve, roc_auc_score
from model.MLP_model import CustomMLP


def ModelEvaluator(model, trainloader, testloader, valloader, criterion, optimizer, device, input_size, output_size, num_epochs=100):
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
    best_model = CustomMLP(input_size, output_size)  
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
