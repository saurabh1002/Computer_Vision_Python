from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

img = np.array(Image.open('1.jpg'))
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

a = np.zeros((np.shape(img)[0], np.shape(img)[1], 8))

a[:, :, 0] = img_1
a[:, :, 1] = img_2
a[:, :, 2] = img_3
a[:, :, 3] = img_4
a[:, :, 4] = img_5
a[:, :, 5] = img_6
a[:, :, 6] = img_7
a[:, :, 7] = img_8

temp_median = np.transpose(np.median(a, 2))

plt.imshow(temp_median, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.tight_layout()
plt.show()
