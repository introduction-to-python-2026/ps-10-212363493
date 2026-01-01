from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d

def load_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

my_img = load_image("/content/my_picture.jpg")
plt.imshow(my_img)
print(my_img.dtype)     
print(my_img.shape)


def edge_detection(my_img_array):
    gray_image = np.mean(my_img_array, axis=2)
    plt.imshow(gray_image, cmap='gray')
    plt.show()
    print(gray_image.shape)
    print(gray_image.dtype)

edge_detection(my_img)
