import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "genome_matrix_full.csv"
data_frame = pd.read_csv(filename)
data_frame = data_frame.set_index('genome_ID')
features = data_frame.iloc[:, :-1]
label_mapping = {'Clinical': 1, 'Non_clinical': 0}
data_frame['Label_numerical'] = data_frame['Label'].map(label_mapping)
labels = data_frame.iloc[:, -1] 

print(labels)