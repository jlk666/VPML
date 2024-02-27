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
        original_image_shape = (image_matrices.shape[1], image_matrices.shape[2])

        

        image_matrices = np.array(image_matrices)
        image_matrices = image_matrices[:, np.newaxis, :, :]
        image_tensor = torch.tensor(image_matrices, dtype=torch.float32).to(device)

        model = CustomCNN(input_channels=1, num_classes=2)
        model.load_state_dict(torch.load('../DL_model/CNN/best_model.pth'))
        model.to(device)
        model.eval()

        grad_cam = GradCAM(model, model.layer3)
        
        
        
        target_class = 1  
        virulence_strain_index = np.where(labels_array == target_class)[0]
        output_dir = 'gradcam_class_1'

        vp_genome_name = 0
        for i in virulence_strain_index:
            input_image = image_tensor[i].unsqueeze(0)  # Add batch dimension
            input_image = input_image.to(device)

            # Generate CAM
            cam = grad_cam.generate_cam(input_image, target_class)
            cam = cv2.resize(cam, (original_image_shape[1], original_image_shape[0]))


            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]  # convert BGR to RGB

            # Plot the heatmap
            plt.imshow(heatmap)
            plt.axis('off')  # Turn off axis numbers and labels
            # Add a color bar
            cbar = plt.colorbar()
            cbar.set_label('Level of Activation', rotation=270, labelpad=15)

            png_filename = vp_genome_name_list[vp_genome_name]+'.png'
            vp_genome_name += 1
            output_path = os.path.join(output_dir, png_filename)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close() 

