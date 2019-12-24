from PIL import Image
import numpy as np
from math import log2
from matplotlib import pyplot as plt
from GrayScale import GrayScale

def Thresholding(img, threshold = 180):
    """
    Converts the input grayscale image to a binary image using a threshold value.
    Input:  numpy array of grayscale image
            thresholding pixel value (default = 180)
    Output: numpy array of thresholded image
    """
    th_img = GrayScale(img) > threshold

    return th_img


if __name__ == "__main__":
    
    img = np.array(Image.open(
        'Images/SRA Khopdi Baba Grayscale.jpg'), np.float64)
    th_img = Thresholding(img)

    plt.imshow(th_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(th_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(th_img)[0]))
    plt.title("Binary Thresholded image")
    plt.show()
