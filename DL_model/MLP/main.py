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

from data_process.PanGeo_preprocess import process_genome_matrix
from data_process.custom_dataset import CustomDataset
from model.MLP_model import CustomMLP
from model_evaluator.model_eval import ModelEvaluator

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

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
            model = CustomMLP(input_size, output_size)
            model = model.to(device)  # move the model to GPU

    # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Evaluate for this fold
            precision, recall, f1, accuracy, fpr, tpr, auc_score  = ModelEvaluator(model, trainloader, testloader, valloader, criterion, optimizer, device, input_size, output_size, num_epochs)

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
        plt.savefig('roc_curve_best_fold.png', dpi=300)  

        with open('roc_parameters_MLP.txt', 'w') as file:
            file.write("Best False Positive Rate (FPR):\n")
            file.write(str(best_fpr))
            file.write("\n\nBest True Positive Rate (TPR):\n")
            file.write(str(best_tpr))
            file.write("\n\nAUC Scores for Each Fold:\n")
            file.write(str(auc_score_kfold))