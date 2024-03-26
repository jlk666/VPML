from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the directory containing the images
image_dir = '/home/zhuosl/VPML/CNN_explainer/gradcam_clinical'
# Directory to save the new PNG files
output_dir = '/home/zhuosl/VPML/CNN_explainer/clinical_pixel_distribution'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def plot_activation_intensity(gray_raw_img, filename):
    plt.figure(figsize=(10, 5))
    plt.hist(gray_raw_img.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of Grayscale Intensities')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{filename}_intensity_distribution.png"))  # Save the plot
    plt.close()  # Close the plot to release memory

# Iterate over each file in the directory
for filename in os.listdir(image_dir):
    # Check if the file is an image file
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Construct the full path to the image
        image_path = os.path.join(image_dir, filename)
        
        # Load the image
        raw_img = plt.imread(image_path)
        
        # Convert to grayscale by taking the mean across the color channels
        gray_raw_img = raw_img.mean(axis=2)
        
        # Plot pixel distribution frequency and save
        plot_activation_intensity(gray_raw_img, filename)

# Find where the grayscale value is higher than 200
#high_activation_indices_raw = np.where(gray_raw_img >  0.6630) # Normalize because np.where expects values [0,1]

# Combine these into readable pairs of indices
#high_activation_pixel_indices_raw = list(zip(high_activation_indices_raw[0], high_activation_indices_raw[1]))

#print(high_activation_pixel_indices_raw)  # Display the first 10 for brevity


