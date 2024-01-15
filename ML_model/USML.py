import sys
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt

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


def k_mean(X, Y, save_results=True):
    n_neighbors = 5
    weights = 'uniform'
    algorithm = 'auto'
    num_folds = 10

    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)

    # Define the scoring metrics
    scoring_metrics = ['accuracy', 'f1', 'precision', 'recall']
    results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(knn_classifier, X, Y, cv=num_folds, scoring=metric)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results[metric.capitalize()] = f"{mean_score:.2f} ± {std_score:.2f}"

    print("Results of KNN:")

    for metric, value in results.items():
        print(f"{metric}: {value}")

    if save_results:
        with open("knn_performance_results.txt", "w") as results_file:
            for metric, value in results.items():
                results_file.write(f"{metric}: {value}\n")

    predicted_probabilities = cross_val_predict(knn_classifier, X, Y, cv=num_folds, method='predict_proba')
    fpr, tpr, _ = roc_curve(Y, predicted_probabilities[:, 1])
    auc_score = roc_auc_score(Y, predicted_probabilities[:, 1])

    return results, auc_score, fpr, tpr

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python matrix.py <filename> <model selection>")
    else:
        filename = sys.argv[1]
        model_selection = sys.argv[2]
        model_selection = model_selection.upper()# Make sure capital issue resolved here
        X,Y = process_genome_matrix(filename)

        if model_selection == 'kmean':
            _, kmean_auc, fpr_Kmean, tpr_Kmean = k_mean(X, Y)
            plt.plot(kmean_auc, label="SVM")