import sys
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt


# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from data_process.PanGeo_image_gradcam import load_and_process_data
from CNN_feature_extractor import GradCAM, CustomCNN


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


        model = CustomCNN(input_channels=1, num_classes=2)
        model.load_state_dict(torch.load('../DL_model/CNN/best_model.pth'))
        model.to(device)
        model.eval()

        #starting get output from the model
        correct = 0
        total = 0

        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        # Calculate accuracy
        correct = (predicted == labels_tensor).sum().item()
        total = labels_tensor.size(0)
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the dataset: {accuracy:.2f}%')


