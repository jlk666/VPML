import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
grad_cam = pd.read_csv('grad_cam.csv')

# Read the CSV file into a pandas DataFrame
rf = pd.read_csv('RF_explain.csv')

grad_cam_dict = {}

# Iterate over each row in the DataFrame
for index, row in grad_cam.iterrows():
    # Extract the values from the row
    gene_name = row['GeneName']
    value = row['Value']
    
    # Store the values in the dictionary
    grad_cam_dict[gene_name] = value

# Print the dictionary
grad_cam_dict = dict(sorted(grad_cam_dict.items(), key=lambda item: item[1]))

max_value_gram_cam = max(grad_cam_dict.values())
normalized_gradcam = {key: value / max_value_gram_cam for key, value in grad_cam_dict.items()}



rf_dict = {}

# Iterate over each row in the DataFrame
for index, row in rf.iterrows():
    # Extract the values from the row
    gene_name = row['GeneName']
    value = row['Value']
    
    # Store the values in the dictionary
    rf_dict[gene_name] = value

# Print the dictionary
rf_dict = dict(sorted(rf_dict.items(), key=lambda item: item[1]))

max_value_rf = max(rf_dict.values())
normalized_rf = {key: value / max_value_gram_cam for key, value in rf_dict.items()}

common_keys = set(normalized_gradcam.keys()).intersection(normalized_rf.keys())

# Extract the corresponding values for the common keys
values_dict1 = [normalized_gradcam[key] for key in common_keys]
values_dict2 = [normalized_rf[key] for key in common_keys]

# Plotting the coefficients
plt.scatter(values_dict1, values_dict2, s =3)
plt.xlabel('Gene feauture used in gradcam')
plt.ylabel('Gene feauture used in RF')
plt.title('Coefficient Plot between gene features used in different models')
plt.grid(True)

# Save the figure
plt.savefig('plot.png', dpi=500)  # Saves the plot as 'plot.png' with 300 dpi resolution


filtered_rf = {key: value for key, value in normalized_rf.items() if value > 0}

common_keys = set(normalized_gradcam.keys()).intersection(filtered_rf.keys())

# Extract the corresponding values for the common keys
values_dict1 = [normalized_gradcam[key] for key in common_keys]
values_dict2 = [filtered_rf[key] for key in common_keys]

# Plotting the coefficients
plt.scatter(values_dict1, values_dict2, s =4)
plt.xlabel('Gene feauture used in gradcam')
plt.ylabel('Gene feauture used in RF')
plt.title('Coefficient Plot between gene features used in different models')
plt.grid(True)

# Save the figure
plt.savefig('plot2.png', dpi=500)  # Saves the plot as 'plot.png' with 300 dpi resolution

plt.show()