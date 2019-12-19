from PIL import Image
import numpy as np
from math import log2
from matplotlib import pyplot as plt

window_size = 7

img = np.array(Image.open('SRA Khopdi Baba Grayscale.jpg'), np.float64)

temp = np.ones((window_size, window_size)) / (window_size ** 2)

img_conv = img.copy()

for i in range((window_size // 2), np.shape(img)[0] - (window_size // 2)):
    for j in range((window_size // 2), np.shape(img)[1] - (window_size // 2)):
        img_conv[i, j] = np.sum(
            temp * img[i-(window_size // 2):i+(window_size // 2)+1, j-(window_size // 2):j+(window_size // 2)+1])

plt.imshow(img_conv, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
