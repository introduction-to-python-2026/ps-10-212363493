from image_utils import load_image, edge_detection
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

my_img = load_image("/content/my_picture.jpg")
edges = edge_detection(my_img, median_size=3, gamma=0.5)

# המרה ל-uint8 0-255
edges_uint8 = (edges.astype(np.uint8) * 255)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(my_img)
plt.axis('off')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(edges_uint8, cmap='gray')
plt.axis('off')
plt.title("Edge Detected Image")

plt.show()

edge_image_pil = Image.fromarray(edges_uint8)
edge_image_pil.save("my_edges.png")
