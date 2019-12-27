from PIL import Image
import numpy as np
from GrayScale import GrayScale
from math import factorial
from matplotlib import pyplot as plt

def Pascal(n, k):
    """
    Generates a number at position (row: n, element: k) in the Pascal triangle.
    Input:  n --> row number
            k --> element number
    Output: number at the input position
    """
    if(k < 0 or k > n):
        return 0
    else:
        return factorial(n) / (factorial(n - k) * factorial(k))

def SobelKernel(window_size):
    """
    Creates a Sobel Kernel of the given window size.
    Input: Size of the Sobel Kernel
    Output: sobel_x --> numpy array of the Sobel Kernel along x direction
            sobel_y --> numpy array of the Sobel Kernel along y direction
    """
    a = np.zeros((window_size, 1))
    b = np.zeros((window_size, 1))
    for i in range(0, window_size):
        a[i] = Pascal(window_size - 1, i)
        b[i] = Pascal(window_size - 2, i) - Pascal(window_size - 2, i - 1)

    sobel_x = np.matmul(a, np.transpose(b))
    sobel_y = np.transpose(sobel_x)

    return sobel_x, sobel_y

def SobelEdgeDetector(img, window_size = 5, dir_x = 1, dir_y = 1):
    """
    Performs edge detection using the Sobel Kernel.
        Input:  
            numpy array of input grayscale image
            Size of the Sobel Kernel (default = 5)
            direction of the Kernel along x-axis (default = 1; set -1 to invert)
            direction of the Kernel along y-axis (default = 1; set -1 to invert)
        Output: 
            sob_mag --> numpy array of edge intensity magnitudes
            sob_dir --> numpy array of edge directions
    """

    kernel_x, kernel_y = SobelKernel(window_size)

    gray_img = GrayScale(img)
    sob_x = gray_img.copy()
    sob_y = gray_img.copy()

    w = window_size // 2

    for i in range(w, np.shape(gray_img)[0] - w):
        for j in range(w, np.shape(gray_img)[1] - w):
            sob_x[i, j] = abs(np.sum(
                dir_x * kernel_x * gray_img[i - w:i + w + 1, j - w:j + w + 1]))
            sob_y[i, j] = abs(np.sum(
                dir_y * kernel_y * gray_img[i - w:i + w + 1, j - w:j + w + 1]))

    sob_mag = np.sqrt(sob_x ** 2 + sob_y ** 2)
    sob_mag = sob_mag * 255 / np.amax(sob_mag)
    sob_dir = np.arctan2(sob_y, sob_x)

    return sob_mag, sob_dir


if __name__ == "__main__":
    img = np.array(Image.open(
        'Images/capture_1.png'), np.float64)
    sob_mag = SobelEdgeDetector(img)

    f = plt.figure()
    plt.title("Sobel Edge Detector")
    f.add_subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(sob_mag, cmap='gray', interpolation='nearest')
    plt.show(block=True)
