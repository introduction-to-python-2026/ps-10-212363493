from image_utils import load_image, edge_detection
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median, threshold_otsu
from skimage.morphology import disk
from scipy.signal import convolve2d

# Load original image
my_img = load_image("/content/my_picture.jpg")

# Convert to grayscale
gray_image = np.mean(my_img, axis=2).astype(np.float64)

# Noise suppression with median filter
clean_image = median(gray_image, disk(3))

# Sobel kernels
kernelY = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
kernelX = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

# Edge detection
edgeY = convolve2d(clean_image, kernelY, mode='same', boundary='fill', fillvalue=0)
edgeX = convolve2d(clean_image, kernelX, mode='same', boundary='fill', fillvalue=0)

# Edge magnitude
edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
edgeMAG = edgeMAG / edgeMAG.max()

# Gamma correction to emphasize edges
edgeMAG = edgeMAG ** 0.5

# Threshold to binary
thresh = threshold_otsu(edgeMAG)
edge_binary = edgeMAG > thresh

# Display images
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(my_img)
plt.axis('off')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(edge_binary.astype(np.uint8)*255, cmap='gray')
plt.axis('off')
plt.title("Edge Detected Image")
plt.show()

# Save edge-detected image
edge_image_to_save = (edge_binary.astype(np.uint8) * 255)
edge_image_pil = Image.fromarray(edge_image_to_save)
edge_image_
