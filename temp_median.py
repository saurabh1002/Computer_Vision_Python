from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt

img_1 = GrayScale(np.array(Image.open('1.jpg')))
img_2 = GrayScale(np.array(Image.open('2.jpg')))
img_3 = GrayScale(np.array(Image.open('3.jpg')))
img_4 = GrayScale(np.array(Image.open('4.jpg')))
img_5 = GrayScale(np.array(Image.open('5.jpg')))
img_6 = GrayScale(np.array(Image.open('6.jpg')))
img_7 = GrayScale(np.array(Image.open('7.jpg')))
img_8 = GrayScale(np.array(Image.open('8.jpg')))


a = np.zeros((np.shape(img_1)[0], np.shape(img_1)[1], 8))

a[:, :, 0] = img_1
a[:, :, 1] = img_2
a[:, :, 2] = img_3
a[:, :, 3] = img_4
a[:, :, 4] = img_5
a[:, :, 5] = img_6
a[:, :, 6] = img_7
a[:, :, 7] = img_8

temp_median = np.transpose(np.median(a, 2))

img = np.transpose(img_5) - temp_median

res = np.hstack((np.transpose(img_1), np.transpose(img_4), np.transpose(img_7), temp_median))

plt.imshow(res, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
