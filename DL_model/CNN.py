# -------CNN based predictor--------
import sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
# Genome image constructure (aka "QR code" of micrbial genome)
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python DLScript.py <filename>")
    else:
        #Check GPU availability first 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        filename = sys.argv[1]
        data_frame = pd.read_csv(filename)
        data_frame = data_frame.set_index('genome_ID')
        features = data_frame.iloc[:, :-1]
        label_mapping = {'Clinical': 1, 'Non_clinical': 0}
        data_frame['Label_numerical'] = data_frame['Label'].map(label_mapping)
        labels = data_frame.iloc[:, -1] 

        features_array = features.values
        labels_array = labels.values

        # Number of columns to be padded with zeros
        padding_columns = 4224 - features_array.shape[1]

        # Padding
        features_array_padded = np.pad(features_array, ((0, 0), (0, padding_columns)), mode='constant', constant_values=0)

        # Sample data (replace this with your actual data)
        num_samples = 1980
        num_features = 4224
        image_width = 64
        image_height = 66  # You can adjust this as needed

        # Create a random feature array for demonstration purposes
        features_array = np.random.rand(num_samples, num_features)

        # Reshape each row into a (64, 66) matrix
        image_matrices = []

        for i in range(num_samples):
            sample_image = features_array_padded[i].reshape(image_width, image_height)
            image_matrices.append(sample_image)

        # Convert the list of image matrices back to a NumPy array if needed

        image_matrices = np.array(image_matrices)
        image_matrices = image_matrices[:, np.newaxis, :, :]
        image_tensor = torch.tensor(image_matrices, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_array, dtype=torch.long)  # Convert labels to a torch tensor of type long
        print(image_tensor.shape)
        print(labels_tensor.shape)

        # Instantiate the model
        model = CustomMLP()
        model = model.to(device)  # move the model to GPU
# ------construct CNN with residual learning structure----------
