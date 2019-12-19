from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


img = np.array(Image.open('SRA Khopdi Baba.jpg').resize((256, 256)))

img_i = 255 - img

print(img.dtype, img.shape, img.ndim)

plt.imshow(img_i, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
# plt.show()

Image.fromarray(img_i).save('SRA Khopdi Baba Inverse.jpg')