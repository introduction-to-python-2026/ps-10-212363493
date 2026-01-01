# image_utils.py
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.array(image)

def edge_detection(my_img_array, gamma=1.0):
    gray_image = np.mean(my_img_array, axis=2).astype(np.float32)

    kernelY = np.array([[1,0,-1],
                        [2,0,-2],
                        [1,0,-1]])
    kernelX = np.array([[-1,-2,-1],
                        [0,0,0],
                        [1,2,1]])

    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    edgeMAG = edgeMAG / edgeMAG.max() * 255
    edgeMAG = edgeMAG ** gamma

    return edgeMAG
