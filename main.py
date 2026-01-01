# main.py
from image_utils import load_image, edge_detection
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball

# 1️⃣ טען את התמונה המקורית
my_img = load_image("/content/my_picture.jpg")

# 2️⃣ הסרת רעש עם median filter
clean_image = median(my_img, ball(3))

# 3️⃣ חישוב הקצוות
edges = edge_detection(clean_image)

# 4️⃣ נורמליזציה + gamma correction
edges = edges / edges.max()
edges = edges ** 0.5  # אפשר לשנות את 0.5 כדי להדגיש קצוות

# 5️⃣ הצגת התמונות
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(my_img)
plt.axis('off')
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.title("Edge Detected Image")

plt.show()

# 6️⃣ שמירת התמונה הסופית
edge_image_to_save = (edges * 255).astype(np.uint8)
Image.fromarray(edge_image_to_save).save("my_edges.png")
