import numpy as np
from PIL import Image
from skimage.filters import median, threshold_otsu
from skimage.morphology import disk
from scipy.signal import convolve2d

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # מוודא 3 ערוצים בלבד
    return np.array(image)

def edge_detection(my_img_array, median_size=3, gamma=0.5):
    gray_image = np.mean(my_img_array, axis=2).astype(np.float32) / 255.0  # ערכים 0-1
    clean_image = median(gray_image, footprint=disk(median_size))
    
    kernelY = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernelX = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    edgeY = convolve2d(clean_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(clean_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    edgeMAG = edgeMAG / edgeMAG.max()
    edgeMAG = edgeMAG ** gamma
    
    thresh = threshold_otsu(edgeMAG)
    edge_binary = edgeMAG > thresh
    
    return edge_binary
