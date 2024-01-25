import pandas as pd
import numpy as np

def load_and_process_data(filename):
    # Load dataframe
    data_frame = pd.read_csv(filename)
    data_frame = data_frame.set_index('genome_ID')
    
    # Extract features
    features = data_frame.iloc[:, :-1]
    label_mapping = {'Clinical': 1, 'Non_clinical': 0}
    data_frame['Label_numerical'] = data_frame['Label'].map(label_mapping)
    labels = data_frame.iloc[:, -1] 
    labels_array = labels.values
    
    # Calculate image shape details
    num_sample = data_frame.shape[0]
    side_length = int(np.ceil(np.sqrt(data_frame.shape[1])))
    genome_image_shape = side_length * side_length

    # Padding
    padding_columns = genome_image_shape - features.values.shape[1]
    features_array_padded = np.pad(features.values, ((0, 0), (0, padding_columns)), mode='constant', constant_values=0)

    # Reshape each row into appropriate matrix shape
    image_matrices = np.array([features_array_padded[i].reshape(side_length, side_length) for i in range(num_sample)])

    return image_matrices, labels_array