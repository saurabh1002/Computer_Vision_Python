from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from GrayScale import GrayScale


def MedianFilter(img, window_size = 3):
    """
    Performs a Filtering operation on the input grayscale image using the Median Kernel.
    Used to remove Salt and Pepper noise.
    Input:  numpy array of grayscale image
            Median Kernel window size (default = 3)
    Output: numpy array of filtered image
    """
    img = GrayScale(img)
    med_img = img.copy()
    w = window_size // 2
    for i in range(w, np.shape(img)[0] - w):
        for j in range(w, np.shape(img)[1] - w):
            med_img[i, j] = np.median(img[i - w:i + w + 1, j - w:j + w + 1])
    return med_img


if __name__ == "__main__":

    img = np.array(Image.open(
        'Images/saltnpepper_Grayscale.png'), np.float64)
    med_img = MedianFilter(img, 5)

    plt.imshow(med_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(med_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(med_img)[0]))
    plt.title("Median Filtered image")
    plt.show()
