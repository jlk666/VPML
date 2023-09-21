import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC # SVM 
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.ensemble import RandomForestClassifier #RF

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

def SVM(X,Y):
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np
    C = 0.88  
    kernel = 'rbf'  
    gamma = 0.005
    num_folds = 5  

    svm_classifier = SVC(C=C, kernel=kernel, gamma=gamma)

    accuracy_scores = cross_val_score(svm_classifier, X, Y, cv=num_folds, scoring='accuracy')
    f1_scores = cross_val_score(svm_classifier, X, Y, cv=num_folds, scoring='f1')
    precision_scores = cross_val_score(svm_classifier, X, Y, cv=num_folds, scoring='precision')
    recall_scores = cross_val_score(svm_classifier, X, Y, cv=num_folds, scoring='recall')

    accuracy_std = np.std(accuracy_scores)
    f1_std = np.std(f1_scores)
    precision_std = np.std(precision_scores)
    recall_std = np.std(recall_scores)

    print("Accuracy Mean:", np.mean(accuracy_scores))
    print("Accuracy Std:", accuracy_std)
    print("F1 Score Mean:", np.mean(f1_scores))
    print("F1 Score Std:", f1_std)
    print("Precision Mean:", np.mean(precision_scores))
    print("Precision Std:", precision_std)
    print("Recall Mean:", np.mean(recall_scores))
    print("Recall Std:", recall_std)


def RF(X, Y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np


    num_folds = 5  

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Example parameters, adjust as needed

    accuracy_scores = cross_val_score(rf_classifier, X, Y, cv=num_folds, scoring='accuracy')
    f1_scores = cross_val_score(rf_classifier, X, Y, cv=num_folds, scoring='f1_macro')  # Note: 'f1_macro' for multiclass tasks
    precision_scores = cross_val_score(rf_classifier, X, Y, cv=num_folds, scoring='precision_macro')  # 'precision_macro' for multiclass
    recall_scores = cross_val_score(rf_classifier, X, Y, cv=num_folds, scoring='recall_macro')  # 'recall_macro' for multiclass

    accuracy_std = np.std(accuracy_scores)
    f1_std = np.std(f1_scores)
    precision_std = np.std(precision_scores)
    recall_std = np.std(recall_scores)

    print("Accuracy Mean:", np.mean(accuracy_scores))
    print("Accuracy Std:", accuracy_std)
    print("F1 Score Mean:", np.mean(f1_scores))
    print("F1 Score Std:", f1_std)
    print("Precision Mean:", np.mean(precision_scores))
    print("Precision Std:", precision_std)
    print("Recall Mean:", np.mean(recall_scores))
    print("Recall Std:", recall_std)

def KNN(X, Y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    import numpy as np

    # Create a K-Nearest Neighbors classifier with your desired parameters
    knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust n_neighbors as needed

    # Define the number of cross-validation folds
    num_folds = 5  

    # Perform cross-validation to obtain performance metrics for each fold
    accuracy_scores = cross_val_score(knn_classifier, X, Y, cv=num_folds, scoring='accuracy')
    f1_scores = cross_val_score(knn_classifier, X, Y, cv=num_folds, scoring='f1_macro')  # 'f1_macro' for multiclass
    precision_scores = cross_val_score(knn_classifier, X, Y, cv=num_folds, scoring='precision_macro')  # 'precision_macro' for multiclass
    recall_scores = cross_val_score(knn_classifier, X, Y, cv=num_folds, scoring='recall_macro')  # 'recall_macro' for multiclass

    accuracy_std = np.std(accuracy_scores)
    f1_std = np.std(f1_scores)
    precision_std = np.std(precision_scores)
    recall_std = np.std(recall_scores)

    print("Accuracy Mean:", np.mean(accuracy_scores))
    print("Accuracy Std:", accuracy_std)
    print("F1 Score Mean:", np.mean(f1_scores))
    print("F1 Score Std:", f1_std)
    print("Precision Mean:", np.mean(precision_scores))
    print("Precision Std:", precision_std)
    print("Recall Mean:", np.mean(recall_scores))
    print("Recall Std:", recall_std)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python matrix.py <filename> <model selection>")
    else:
        filename = sys.argv[1]
        model_selection = sys.argv[2]
        model_selection = model_selection.upper()# Make sure capital issue resolved here
        X,Y = process_genome_matrix(filename)

        if model_selection == 'SVM':
            SVM(X,Y)
        elif model_selection == 'RF':
            RF(X,Y)
        elif model_selection == 'KNN':
            KNN(X,Y)
        elif model_selection == 'ALL':
            SVM(X,Y)
            RF(X,Y)
            KNN(X,Y)

