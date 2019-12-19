from PIL import Image
import numpy as np
from math import log2
from matplotlib import pyplot as plt

img = np.array(Image.open('SRA Khopdi Baba Grayscale.jpg'), np.uint8)

window_size = 3

img_conv = img.copy()

for i in range((window_size // 2), np.shape(img)[0] - (window_size // 2)):
    for j in range((window_size // 2), np.shape(img)[1] - (window_size // 2)):
        img_conv[i, j] = np.median(img[i-(window_size // 2):i+(window_size // 2)+1, j-(window_size // 2):j+(window_size // 2)+1])

plt.imshow(img_conv, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
