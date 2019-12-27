from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt


def LaplacianEdgeDetection(img):
    """
    Performs Second order edge detection using the Laplacian Kernel.
    Input:  numpy array of grayscale image
    Output: numpy array of image with detected edges
    """
    kernel = np.array([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])

    gray_img = GrayScale(img)
    laplacian = gray_img.copy()

    for i in range(1, np.shape(gray_img)[0] - 1):
        for j in range(1, np.shape(gray_img)[1] - 1):
            laplacian[i, j] = abs(np.sum(kernel * gray_img[i - 1:i + 2, j - 1:j + 2]))

    return laplacian


if __name__ == "__main__":
    img = np.array(Image.open(
        'Images/capture_1.png'), np.float64)
    laplacian = LaplacianEdgeDetection(img)

    f = plt.figure()
    plt.title("Laplacian edge detection operator")
    f.add_subplot(1, 1, 1)
    plt.imshow(laplacian, cmap='gray', interpolation='nearest')
    plt.show(block=True)
