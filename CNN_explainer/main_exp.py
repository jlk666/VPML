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
    
def pixel_index_finder(gray_raw_img):
    high_activation_indices_raw = np.where(gray_raw_img >  0.6630)
    high_activation_pixel_indices_raw = list(zip(high_activation_indices_raw[0], high_activation_indices_raw[1]))
    if(len(high_activation_pixel_indices_raw) != 0):
        return high_activation_pixel_indices_raw
    return None

# Iterate over each file in the directory
total_index = []
for filename in os.listdir(image_dir):
    # Check if the file is an image file
    if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
        # Construct the full path to the image
        image_path = os.path.join(image_dir, filename)
        
        # Load the image
        raw_img = plt.imread(image_path)
        
        # Convert to grayscale by taking the mean across the color channels
        gray_raw_img = raw_img.mean(axis=2)
        
        each_genome_index = index_mapping(gray_raw_img)
        if (each_genome_index != None):
            total_index.append(each_genome_index)

sets = [set(lst) for lst in total_index]
overlap = set.intersection(*sets)

print("Overlap among the lists:", total_index)



