from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt

def AverageBlur(img, window_size = 3):
    """
    Performs a Blurring operation on the input grayscale image using the Averaging Kernel.
    Input:  numpy array of grayscale image
            Averaging Kernel window size (default = 3)
    Output: numpy array of blurred image
    """
    img = GrayScale(img)
    avg_img = img.copy()
    temp = np.ones((window_size, window_size)) / (window_size ** 2)
    w = window_size // 2

    for i in range(w, np.shape(img)[0] - w):
        for j in range(w, np.shape(img)[1] - w):
            avg_img[i, j] = np.sum(
                temp * img[i - w:i + w + 1, j - w:j + w + 1])

    return avg_img


if __name__ == "__main__":
    
    img = np.array(Image.open(
        'Images/SRA Khopdi Baba Grayscale.jpg'), np.float64)
    avg_img = AverageBlur(img)

    plt.imshow(avg_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(avg_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(avg_img)[0]))
    plt.title("Average Kernel Blur")
    plt.show()
