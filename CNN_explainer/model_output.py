import sys
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from data_process.PanGeo_image_gradcam import load_and_process_data
from CNN_feature_extractor import GradCAM, CustomCNN
from torch.utils.data import TensorDataset, DataLoader

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python DLScript.py <filename>")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file = sys.argv[1]
        image_matrices, labels_array, vp_genome_name_list = load_and_process_data(file)
        #original_image_shape = (image_matrices.shape[1], image_matrices.shape[2])

        

        image_matrices = np.array(image_matrices)
        image_matrices = image_matrices[:, np.newaxis, :, :]
        image_tensor = torch.tensor(image_matrices, dtype=torch.float32).to(device)
        labels_tensor = torch.tensor(labels_array, dtype=torch.long).to(device)  # Ensure labels are on the same device

        dataset = TensorDataset(image_tensor, labels_tensor)
        batch_size = 64  # Adjust based on your GPU memory
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


        model = CustomCNN(input_channels=1, num_classes=2)
        model.load_state_dict(torch.load('../DL_model/CNN/best_model.pth'))
        model.to(device)
        model.eval()

        #starting get output from the model
        correct_indices = [] 
        correct = 0
        total = 0
        current_index = 0 

        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                matches = (predicted == labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Identify correct indices in this batch
                correct_batch_indices = (matches.nonzero(as_tuple=False) + current_index).view(-1).tolist()
                correct_indices.extend(correct_batch_indices)
                current_index += labels.size(0)  # Update global index


        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy}%')

        with open('correct_prediction_indices.txt', 'w') as f:
            for index in correct_indices:
                f.write(f"{index}\n")

        print(f"Saved correct prediction indices to 'correct_prediction_indices.txt'.")

