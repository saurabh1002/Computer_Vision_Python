from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

slope = 2
bright_level = 100

img = np.array(Image.open('SRA Khopdi Baba Grayscale.jpg'))

img = (img % bright_level) * slope

plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
# plt.show()
