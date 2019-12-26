from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt


def ExpExpand(img, scale_1 = 0.01, scale_2 = 20):
    """
    Expands the pixel values of the grayscale input image using the exponential operator.
    Input:  numpy array of grayscale image
            scale_1 --> scales the input image's pixels
            scale_2 --> scales the expanded image's pixels
    Output: numpy array of expanded pixel values image
    """
    exp_img = scale_2 * (np.exp(GrayScale(img) * scale_1))

    return exp_img


if __name__ == "__main__":

    img = np.array(Image.open(
        'Images/SRA Khopdi Baba Grayscale.jpg'), np.float64)
    exp_img = ExpExpand(img)

    plt.imshow(exp_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(exp_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(exp_img)[0]))
    plt.title("Exponential Expansion")
    plt.show()

