from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

img = np.array(Image.open('Images/1.jpg'))
img_1 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('Images/2.jpg'))
img_2 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('Images/3.jpg'))
img_3 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('Images/4.jpg'))
img_4 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('Images/5.jpg'))
img_5 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('Images/6.jpg'))
img_6 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('Images/7.jpg'))
img_7 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('Images/8.jpg'))
img_8 = np.mean(img[:, :, :], 2)

temp_avg = (img_1 + img_2 + img_3 + img_4 + img_5 + img_6 + img_7 + img_8) / 8
temp_avg = np.transpose(temp_avg)

plt.imshow(temp_avg, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
