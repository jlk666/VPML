import sys
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

    print("In this pangenome matrix, you have", data_frame.shape[0], "samples and each having", data_frame.shape[1], "features.")

    return features_array, labels_array


from sklearn.cluster import KMeans

def k_means_clustering(X, Y, filename, save_results=True):
    # Create KMeans instance with desired number of clusters
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit the model
    kmeans.fit(X)

    # Get cluster assignments
    kmean_labels = kmeans.labels_
    # Assuming true_labels contain the ground truth labels
    accuracy = accuracy_score(Y, kmean_labels)
    precision = precision_score(Y, kmean_labels, average='weighted')  # Use weighted for multiclass
    recall = recall_score(Y, kmean_labels, average='weighted')  # Use weighted for multiclass
    f1 = f1_score(Y, kmean_labels, average='weighted')  # Use weighted for multiclass

    # Optionally save the results
    output_filename = filename.split('.')[0] + '_kmeaneval.txt'
    if save_results:
        with open(output_filename, "w") as results_file:
            results_file.write(f"Accuracy: {accuracy:.2f}\n")
            results_file.write(f"Precision: {precision:.2f}\n")
            results_file.write(f"Recall: {recall:.2f}\n")
            results_file.write(f"F1 Score: {f1:.2f}\n")

    return kmean_labels


def GMM(X, Y, filename, save_results=True):
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

    # Assuming true_labels contain the ground truth labels
    accuracy = accuracy_score(Y, gmm_labels)
    precision = precision_score(Y, gmm_labels, average='weighted')  # Use weighted for multiclass
    recall = recall_score(Y, gmm_labels, average='weighted')  # Use weighted for multiclass
    f1 = f1_score(Y, gmm_labels, average='weighted')  # Use weighted for multiclass

    # Optionally save the results
    if save_results:
        output_filename = filename.split('.')[0] + '_gmmeval.txt'
        with open(output_filename, "w") as results_file:
            results_file.write(f"Accuracy: {accuracy:.2f}\n")
            results_file.write(f"Precision: {precision:.2f}\n")
            results_file.write(f"Recall: {recall:.2f}\n")
            results_file.write(f"F1 Score: {f1:.2f}\n")

    return principal_components,gmm_labels

def draw_fig(principal_components, labels_kmean, labels_gmm, filename):
    filename = sys.argv[1]
    data_frame = pd.read_csv(filename)
    data_frame = data_frame.set_index('genome_ID')
    label_mapping = {'Clinical': 1, 'Non_clinical': 0}
    data_frame['Label_numerical'] = data_frame['Label'].map(label_mapping)
    labels = data_frame.iloc[:, -1] 

    # Define the dot size
    dot_size = 2

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))  # 1 row, 3 columns, and set a figure size

    # First plot: Unsupervised PCA pathogenicity classification
    scatter1 = axs[0].scatter(principal_components[:, 0], principal_components[:, 1], c=data_frame['Label_numerical'], cmap='viridis', s=dot_size)
    axs[0].set_title('Unsupervised PCA pathogenicity classification')
    axs[0].set_xlabel('Principal Component 1')
    axs[0].set_ylabel('Principal Component 2')
    cbar1 = fig.colorbar(scatter1, ax=axs[0], ticks=[0, 1])
    cbar1.set_label('Pathogenicity')
    cbar1.set_ticklabels(['Clinical', 'Non-clinical'])

    # Second plot: Kmeans clustering on PCA components
    scatter2 = axs[1].scatter(principal_components[:, 0], principal_components[:, 1], c=labels_kmean, cmap='viridis', s=dot_size)
    axs[1].set_title('Kmeans clustering on PCA components')
    axs[1].set_xlabel('Principal Component 1')
    axs[1].set_ylabel('Principal Component 2')
    cbar2 = fig.colorbar(scatter2, ax=axs[1], ticks=[0, 1])
    cbar2.set_label('Cluster ID')
    cbar2.set_ticklabels(['Cluster 1', 'Cluster 2'])

    # Third plot: GMM clustering on PCA components
    scatter3 = axs[2].scatter(principal_components[:, 0], principal_components[:, 1], c=labels_gmm, cmap='viridis', s=dot_size)
    axs[2].set_title('GMM clustering on PCA components')
    axs[2].set_xlabel('Principal Component 1')
    axs[2].set_ylabel('Principal Component 2')
    cbar3 = fig.colorbar(scatter3, ax=axs[2], ticks=[0, 1])
    cbar3.set_label('Cluster ID')
    cbar3.set_ticklabels(['Cluster 1', 'Cluster 2'])

    output_filename = filename.split('.')[0] + '_plot.png'
    # Save the plot to a file
    fig.savefig(output_filename, dpi=600)  # Adjust the filename and DPI as needed

# Display the plot
plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python matrix.py <filename> <model selection>")
    else:
        filename = sys.argv[1]
        model_selection = sys.argv[2]
        model_selection = model_selection.upper()# Make sure capital issue resolved here
        X,Y = process_genome_matrix(filename)

        if model_selection == 'ALL':
            principal_components,gmm_labels = GMM(X, Y, filename)
            kmean_labels = k_means_clustering(X, Y, filename)
            draw_fig(principal_components, kmean_labels, gmm_labels, filename)


