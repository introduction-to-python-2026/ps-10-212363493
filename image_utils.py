import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

def edge_detection(image_array):
    gray = np.mean(image_array, axis=2).astype(np.float64)
    
    kernelX = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]])
    
    kernelY = np.array([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]])
    
    edgeX = convolve2d(gray, kernelX, mode='same', boundary='fill', fillvalue=0)
    edgeY = convolve2d(gray, kernelY, mode='same', boundary='fill', fillvalue=0)
    
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
