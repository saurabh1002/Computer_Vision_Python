from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


img = np.array(Image.open('SRA Khopdi Baba.jpg'))


img_R = img.copy()
img_R[:,:,(1,2)] = 0

img_G = img.copy()
img_G[:, :, (0, 2)] = 0

img_B = img.copy()
img_B[:, :, (0, 1)] = 0

img_RGB = np.hstack((img_R, img_G, img_B))

print(img.dtype, img.shape, img.ndim)

plt.imshow(img_RGB, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
