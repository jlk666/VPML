import sys
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

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


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def k_means_clustering(X, save_results=True):
    # Create KMeans instance with desired number of clusters
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the model
    kmeans.fit(X)

    # Get cluster assignments
    labels = kmeans.labels_

    # Calculate silhouette score as a measure of cluster quality
    silhouette_avg = silhouette_score(X, labels)

    print(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}")

    if save_results:
        with open("kmeans_performance_results.txt", "w") as results_file:
            results_file.write(f"Silhouette Score for {n_clusters} clusters: {silhouette_avg:.2f}\n")

    return labels, silhouette_avg


def GMM(X, save_results=True):
    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction
    num_components = 2
    pca = PCA(n_components=num_components)
    principal_components = pca.fit_transform(scaled_features)

    # Fit the Gaussian Mixture Model
    n_clusters = 2
    gmm = GaussianMixture(n_components=n_clusters)
    gmm.fit(principal_components)

    # Obtain the labels for each data point
    gmm_labels = gmm.predict(principal_components)

    # Evaluate the model using silhouette score
    silhouette_avg = silhouette_score(principal_components, gmm_labels)
    print(f"Silhouette Score: {silhouette_avg:.2f}")

    # Optionally save the results
    if save_results:
        with open("gmm_performance_results.txt", "w") as results_file:
            results_file.write(f"Silhouette Score: {silhouette_avg:.2f}\n")

    return gmm_labels, silhouette_avg


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python matrix.py <filename> <model selection>")
    else:
        filename = sys.argv[1]
        model_selection = sys.argv[2]
        model_selection = model_selection.upper()# Make sure capital issue resolved here
        X,Y = process_genome_matrix(filename)

        if model_selection == 'ALL':
            labels, silhouette_avg = k_means_clustering(X, 2)
            labels, silhouette_avg = GMM(X)


