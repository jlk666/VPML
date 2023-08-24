import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score 


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

def SVM(X_train, y_train):
    C = 0.88  # Regularization parameter
    kernel = 'rbf'  # You can experiment with different kernels (linear, rbf, poly, etc.)
    gamma = 0.005
    svm_classifier = SVC(C=C, kernel=kernel)
    y_pred_cv = cross_val_predict(svm_classifier, X_train, y_train, cv=5)
    accuracy = accuracy_score(y_train, y_pred_cv)
    classification_rep = classification_report(y_train, y_pred_cv)
    confusion_mat = confusion_matrix(y_train, y_pred_cv)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)

def RF(X_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=150, random_state=42)
    cv_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)

    print("Cross-Validation Scores:", cv_scores)
    print("Mean Cross-Validation Score:", cv_scores.mean())

    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_train)

    accuracy = accuracy_score(y_train, y_pred)
    classification_rep = classification_report(y_train, y_pred)
    confusion_mat = confusion_matrix(y_train, y_pred)

    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)
    

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python matrix.py <filename> <model selection>")
    else:
        filename = sys.argv[1]
        model_selection = sys.argv[2]

        X,Y = process_genome_matrix(filename)

        if model_selection == 'SVM':
            SVM(X,Y)
        elif model_selection == 'RF':
            RF(X,Y)

