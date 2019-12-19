from PIL import Image
import numpy as np
from math import log2
from matplotlib import pyplot as plt

img = np.array(Image.open('SRA Khopdi Baba Grayscale.jpg'), np.float64)

threshold = 180

img = img > threshold

plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
