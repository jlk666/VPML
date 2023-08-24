import sys
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python matrix.py <filename>")
    else:
        filename = sys.argv[1]
        process_genome_matrix(filename)
