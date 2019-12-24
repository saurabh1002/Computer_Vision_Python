from PIL import Image
import numpy as np
from math import log2
from GrayScale import GrayScale
from matplotlib import pyplot as plt

def GaussianBlur(img, window_size = 3, sigma = 0.5):
    """
    Performs a Blurring operation on the input grayscale image using the Normalized Gaussian Kernel.
    Input:  numpy array of grayscale image
            Gaussian Kernel window size (default = 3)
            Standard Deviation of the Gaussian Kernel (default = 0.5)
    Output: numpy array of blurred image
    """
    temp = np.zeros((window_size, window_size))

    sum = 0
    center = window_size // 2

    for i in range (0, window_size):
        for j in range (0, window_size):
            temp[i, j] = np.exp(-1 * (((i - center) ** 2) + ((j - center) ** 2)) / (sigma ** 2 * 2))
            sum += temp[i, j]

    temp = temp / sum

    img_conv = GrayScale(img).copy()

    for i in range((window_size // 2), np.shape(img)[0] - (window_size // 2)):
        for j in range((window_size // 2), np.shape(img)[1] - (window_size // 2)):
            img_conv[i, j] = np.sum(
                temp * img[i-(window_size // 2):i+(window_size // 2)+1, j-(window_size // 2):j+(window_size // 2)+1])

    return img_conv


if __name__ == "__main__":

    img = np.array(Image.open(
        'Images/SRA Khopdi Baba Grayscale.jpg'), np.float64)
    img_conv = GaussianBlur(img, 5, 2)

    plt.imshow(img_conv, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(img_conv)[1]))
    plt.ylabel('{} pixels'.format(np.shape(img_conv)[0]))
    plt.title("Gaussian Filtered Image")
    plt.show()

