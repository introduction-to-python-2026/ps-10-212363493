from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median, threshold_otsu
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def load_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

my_img = load_image("/content/my_picture.jpg")
plt.imshow(my_img)
print(my_img.dtype)     
print(my_img.shape)


def edge_detection_final(my_img_array, median_size=3, gamma=0.5, show_intermediate=False):
    gray_image = np.mean(my_img_array, axis=2)
    
    if show_intermediate:
        plt.imshow(gray_image, cmap='gray')
        plt.axis('off')
        plt.show()
    
    clean_image = median(gray_image, disk(median_size))
    
    if show_intermediate:
        plt.imshow(clean_image, cmap='gray')
        plt.axis('off')
        plt.show()
    
    kernelY = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernelX = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    edgeY = convolve2d(clean_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(clean_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    edgeMAG = edgeMAG / edgeMAG.max()
    edgeMAG = edgeMAG ** gamma
    
    thresh = threshold_otsu(edgeMAG)
    edge_binary = edgeMAG > thresh
    
    plt.figure(figsize=(8,6))
    plt.imshow(edge_binary, cmap='gray')
    plt.axis('off')
    plt.show()
    
    return edge_binary

my_img = load_image("/content/my_picture.jpg")
edges = edge_detection_final(my_img)
