from PIL import Image
import numpy as np
from math import log2
from GrayScale import GrayScale
from matplotlib import pyplot as plt

def LogCompress(img, scale_1 = 100, scale_2 = 20):
    """
    Compresses the pixel values of the grayscale input image using the natural logarithmic operator.
    Input:  numpy array of grayscale image
            scale_1 --> scales the input image's pixels
            scale_2 --> scales the compressed image's pixels
    Output: numpy array of compressed pixel values image
    """
    log_img = scale_2 * np.log2(scale_1 * GrayScale(img))

    return log_img

if __name__ == "__main__":

    img = np.array(Image.open(
        'Images/SRA Khopdi Baba Grayscale.jpg'), np.float64)
    log_img = LogCompress(img)

    plt.imshow(log_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(log_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(log_img)[0]))
    plt.title("Logarithmic Compression")
    plt.show()
