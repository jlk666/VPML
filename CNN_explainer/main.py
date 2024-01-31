import sys
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
from DL_model.CNN.data_process.PanGeo_image import load_and_process_data
from CNN_feature_extractor import GradCAM, CustomCNN

# ... [rest of your imports and CustomCNN definition]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python DLScript.py <filename>")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file = sys.argv[1]
        image_matrices, labels_array = load_and_process_data(file)
        image_matrices = np.array(image_matrices)
        image_matrices = image_matrices[:, np.newaxis, :, :]
        image_tensor = torch.tensor(image_matrices, dtype=torch.float32).to(device)

        model = CustomCNN(input_channels=1, num_classes=2)
        model.load_state_dict(torch.load('../DL_model/CNN/best_model.pth'))
        model.to(device)
        model.eval()

        grad_cam = GradCAM(model, model.layer3)
        num_images = image_tensor.size(0)
        # Assuming you want to process the first image in the batch
        
        input_image = image_tensor[0].unsqueeze(0)  # Add batch dimension
        input_image = input_image.to(device)
        
        
        target_class = 1  # or model(input_image).argmax().item() for the predicted class

        # Generate CAM
        cam = grad_cam.generate_cam(input_image, target_class)

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]  # convert BGR to RGB

        # Plot the heatmap
        plt.imshow(heatmap)
        plt.axis('off')  # Turn off axis numbers and labels
        # Add a color bar
        cbar = plt.colorbar()
        cbar.set_label('Level of Activation', rotation=270, labelpad=15)
        
        plt.savefig('draft.png', bbox_inches='tight', pad_inches=0)
        plt.close() 

