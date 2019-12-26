from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt


def PrewittsOperator(img):
    """
    Performs edge detection using the Prewitts Operator. It is an extension to Improved First Order Difference method.
    Input:  numpy array of grayscale image
    Output: prew_x --> numpy array of Vertical edges
            prew_y --> numpy array of Horizontal edges
            edge_mag --> numpy array of edge intensity magnitudes
    """
    kernel_x = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    kernel_y = np.transpose(kernel_x)

    gray_img = GrayScale(img)
    prew_x = gray_img.copy()
    prew_y = gray_img.copy()
    for i in range(1, np.shape(gray_img)[0] - 1):
        for j in range(1, np.shape(gray_img)[1] - 1):
            prew_x[i, j] = abs(np.sum(kernel_x * gray_img[i - 1:i + 2, j - 1:j + 2]))
            prew_y[i, j] = abs(np.sum(kernel_y * gray_img[i - 1:i + 2, j - 1:j + 2]))

    edge_mag = np.sqrt(prew_x ** 2 + prew_y ** 2)
    return prew_x, prew_y, edge_mag


if __name__ == "__main__":
    img = np.array(Image.open(
        'Images/capture_1.png'), np.float64)
    prew_x, prew_y, prew_m = PrewittsOperator(img)

    f = plt.figure()
    plt.title("Prewitts edge detection operator")
    f.add_subplot(1, 3, 1)
    plt.title("Vertical Egdes")
    plt.imshow(prew_x, cmap='gray', interpolation='nearest')
    f.add_subplot(1, 3, 2)
    plt.title("Horizontal Edges")
    plt.imshow(prew_y, cmap='gray', interpolation='nearest')
    f.add_subplot(1, 3, 3)
    plt.title("Magnitude of edge vectors")
    plt.imshow(prew_m, cmap='gray', interpolation='nearest')
    plt.show(block=True)
