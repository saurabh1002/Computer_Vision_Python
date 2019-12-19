from PIL import Image
import numpy as np
from math import log2
from matplotlib import pyplot as plt

window_size = 3
sigma = 0.1
temp = np.zeros((window_size, window_size))

sum = 0
center = window_size // 2
for i in range (0, window_size):
    for j in range (0, window_size):
        temp[i, j] = np.exp(-1 * (((i - center) ** 2) + ((j - center) ** 2)) / (sigma ** 2 * 2))
        sum += temp[i, j]

temp = temp / sum

img = np.array(Image.open('SRA Khopdi Baba Grayscale.jpg'), np.float64)

img_conv = img.copy()

for i in range((window_size // 2), np.shape(img)[0] - (window_size // 2)):
    for j in range((window_size // 2), np.shape(img)[1] - (window_size // 2)):
        img_conv[i, j] = np.sum(
            temp * img[i-(window_size // 2):i+(window_size // 2)+1, j-(window_size // 2):j+(window_size // 2)+1])

plt.imshow(img_conv, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
