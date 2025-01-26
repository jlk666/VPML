import sys
import numpy as np
import pandas as pd
import os 
import csv


def process_genome_matrix(filename):
    data_frame = pd.read_csv(filename)
    genome_id_col = data_frame['genome_ID']
    data_frame = data_frame.set_index('genome_ID')

    features = data_frame.iloc[:, :-1]
    label_mapping = {'Clinical': 1, 'Non_clinical': 0}
    data_frame['Label_numerical'] = data_frame['Label'].map(label_mapping)
    labels = data_frame.iloc[:, -1] 

    features_array = features.values
    labels_array = labels.values
    #create a dictionary for later Random Forest Explainer
    column_names = data_frame.columns.tolist()
    column_dict = {i+1: column_name for i, column_name in enumerate(column_names)}
    #Dont need to do data split as we choose to do cross validation 
    #X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.2, random_state=42)

    print("In this pangenome matrix, you have", data_frame.shape[0], "samples and each having", features.shape[1], "features.")

    return features_array, labels_array, column_dict, genome_id_col


