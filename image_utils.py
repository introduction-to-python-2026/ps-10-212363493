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
    from scipy.signal import convolve2d
    filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) 
    filtered_image = convolve2d(gray_image, filter, mode='same')
    plt.imshow(filtered_image, cmap='gray')

    kernelY = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    kernelX = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ])

   
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 4️⃣ Compute edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    # 5️⃣ Normalize to 0-1
    edgeMAG = edgeMAG / edgeMAG.max()

    # 6️⃣ Gamma correction to emphasize edges
    edgeMAG = edgeMAG ** 0.5  # adjust exponent if needed

    # 7️⃣ Display
    plt.imshow(edgeMAG, cmap='gray')
    plt.axis('off')
    plt.show()

    return edgeMAG

edge_detection(my_img)  
