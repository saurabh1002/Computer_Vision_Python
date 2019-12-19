from PIL import Image
import numpy as np
from math import log2
from matplotlib import pyplot as plt

img = np.array(Image.open('SRA Khopdi Baba Grayscale.jpg'), np.float64)

img = 20 * (np.log2(100 * img))

plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
# plt.show()
