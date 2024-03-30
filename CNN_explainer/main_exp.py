from PIL import Image
from collections import defaultdict
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

def majority_vote(list_of_lists):
    # Create a defaultdict to store the counts of each tuple
    count_dict = defaultdict(int)

    # Iterate over each list
    for inner_list in list_of_lists:
        # Iterate over each tuple in the inner list
        for tpl in inner_list:
            # Increment the count for this tuple
            count_dict[tpl] += 1

    # Find the tuple with the maximum count
    majority_vote = max(count_dict, key=count_dict.get)
    
    return majority_vote, count_dict

def plot_entry_frequencies(count_dict):
    # Extract entries and frequencies from the count dictionary
    entries = [str(entry) for entry in count_dict.keys()]
    frequencies = list(count_dict.values())

    # Create a bar graph
    plt.figure(figsize=(10, 6))
    plt.bar(entries, frequencies, color='skyblue')
    plt.xlabel('Entry')
    plt.ylabel('Frequency')
    plt.title('Entry Frequencies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    plt.savefig("majority_voting.png")

def convert_to_1d_index(x, y, width):
    return y * width + x + 1

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
        print(gray_raw_img.shape)
        each_genome_index = pixel_index_finder(gray_raw_img)
        if (each_genome_index != None):
            total_index.append(each_genome_index)

result, count_dict  = majority_vote(total_index)



