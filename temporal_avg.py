from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

img = np.array(Image.open('1.jpg'))
img_1 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('2.jpg'))
img_2 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('3.jpg'))
img_3 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('4.jpg'))
img_4 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('5.jpg'))
img_5 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('6.jpg'))
img_6 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('7.jpg'))
img_7 = np.mean(img[:, :, :], 2)
img = np.array(Image.open('8.jpg'))
img_8 = np.mean(img[:, :, :], 2)

temp_avg = (img_1 + img_2 + img_3 + img_4 + img_5 + img_6 + img_7 + img_8) / 8
temp_avg = np.transpose(temp_avg)

plt.imshow(temp_avg, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
