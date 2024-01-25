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

from data_process.PanGeo_image import load_and_process_data
from data_process.custom_dataset import CustomDataset
from model.CNN_model import ResidualBlock, CustomCNN
from model_evaluator.model_eval import ModelEvaluator

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

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
        rs = 39
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=rs)
        wandb.config = {"k-fold rs": rs}

        for fold, (train_valid_idx, test_idx) in enumerate(kf.split(image_tensor, labels_tensor)):
            print(f'Fold {fold + 1}/{k_folds}')

            features_train_valid, labels_train_valid = image_tensor[train_valid_idx], labels_tensor[train_valid_idx]
            features_test, labels_test = image_tensor[test_idx], labels_tensor[test_idx]

            rs2 = 39
            X_train, X_valid, y_train, y_valid = train_test_split(features_train_valid, labels_train_valid, test_size= 1/9, random_state=rs2)
            wandb.config = {"train_valid rs": rs}
            
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

