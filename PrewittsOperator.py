from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt


def PrewittsOperator(img):
    """
    Performs edge detection using the Prewitts Operator. It is an extension to Improved First Order Difference method.
    Input:  numpy array of grayscale image
    Output: edge_mag --> numpy array of edge intensity magnitudes
            edge_dir --> numpy array of edge directions
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
    edge_mag = edge_mag * 255 / np.amax(edge_mag)
    edge_dir = np.arctan2(prew_y, prew_x)

    return edge_mag, edge_dir


if __name__ == "__main__":
    img = np.array(Image.open(
        'Images/capture_1.png'), np.float64)
    prew_m, prew_d = PrewittsOperator(img)

    f = plt.figure()
    plt.title("Prewitts edge detection operator")
    f.add_subplot(1, 1, 1)
    plt.imshow(prew_m, cmap='gray', interpolation='nearest')
    plt.show(block=True)
