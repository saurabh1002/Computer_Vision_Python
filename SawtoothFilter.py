from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt

def SawtoothFilter(img, width = 100, slope = 2):
    """
    Rescales the pixel values of an input grayscale image using a sawtooth waveform.
    Input:  numpy array of grayscale image
            width of the sawtooth wave (default = 100)
            slope of the sawtooth wave (default = 2)
    Output: numpy array of filtered image
    """
    saw_img = (GrayScale(img) % width) * slope

    return saw_img


if __name__ == "__main__":

    img = np.array(Image.open('Images/SRA Khopdi Baba Grayscale.jpg'))
    saw_img = SawtoothFilter(img)

    plt.imshow(saw_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(saw_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(saw_img)[0]))
    plt.title("Sawtooth Filter")
    plt.show()
