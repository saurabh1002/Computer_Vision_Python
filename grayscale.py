from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

grey_img = np.zeros((256, 256), np.uint8)
img = np.array(Image.open('saltnpepper.png'))

grey_img = np.mean(img[:, :, :], 2)

# print(grey_img.dtype, grey_img.shape, grey_img.ndim)

plt.imshow(grey_img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()

Image.fromarray(grey_img).convert("L").save('saltnpepper_Grayscale.png')
