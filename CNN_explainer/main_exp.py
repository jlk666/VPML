from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Path to your image
image_path = 'gradcam_class_1/GCF_000477475.1_ASM47747v1.png'

# Open the image
img = Image.open(image_path)

# Convert the image to a numpy array
img_array = np.array(img)

# Plot the heatmap
plt.imshow(img_array, cmap='hot', interpolation='nearest')
plt.colorbar()  # Add color bar for reference
plt.title('Pixel Heatmap')
plt.xlabel('Width')
plt.ylabel('Height')

# Save the heatmap
plt.savefig('heatmap.png')

# Show the plot (optional)
plt.show()
