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

def SVM(X, Y):
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    import numpy as np

    C = 0.88
    kernel = 'rbf'
    gamma = 0.005
    num_folds = 5

    svm_classifier = SVC(C=C, kernel=kernel, gamma=gamma)

    # Define the scoring metrics
    scoring_metrics = ['accuracy', 'f1', 'precision', 'recall']
    results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(svm_classifier, X, Y, cv=num_folds, scoring=metric)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results[metric.capitalize()] = f"{mean_score:.2f} ± {std_score:.2f}"

    print("Results of SVM:")

    for metric, value in results.items():
        print(f"{metric}: {value}")



def RF(X, Y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np

    num_folds = 5

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Define the scoring metrics
    scoring_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(rf_classifier, X, Y, cv=num_folds, scoring=metric)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results[metric.capitalize()] = f"{mean_score:.2f} ± {std_score:.2f}"

    print("Results of Random Forest Classifier:")
    for metric, value in results.items():
        print(f"{metric}: {value}")

# Example usage:
# RF(X, Y)


def KNN(X, Y):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    import numpy as np

    # Create a K-Nearest Neighbors classifier with your desired parameters
    knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust n_neighbors as needed

    # Define the number of cross-validation folds
    num_folds = 5  

    # Define the scoring metrics
    scoring_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(knn_classifier, X, Y, cv=num_folds, scoring=metric)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        results[metric.capitalize()] = f"{mean_score:.2f} ± {std_score:.2f}"

    print("Results of K-Nearest Neighbors Classifier:")
    for metric, value in results.items():
        print(f"{metric}: {value}")



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

