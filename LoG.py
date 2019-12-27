from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt

def LoG(img, window_size = 5, sigma = 0.5):
    """
    Performs Second Order Difference using the Laplacian of Gaussian Operator (Marr-Hildreth Edge Detection)
    Input:  numpy array of input grayscale image
            kernel size (default = 5)
            variance for the Gaussian operator (default = 0.5) 
    Output: numpy array of edges detected by the LoG kernel and Marr-Hildreth method
    """
    kernel = np.zeros((window_size, window_size))
    sum = 0
    center = window_size // 2

    for i in range(0, window_size):
        for j in range(0, window_size):
            kernel[i, j] = (sigma ** -2) * (((((i - center) ** 2) + ((j - center) ** 2)) / (sigma ** 2)) - 2) * np.exp(-1 * (((i - center) ** 2) + ((j - center) ** 2)) / (sigma ** 2 * 2))
            sum += kernel[i, j]
    kernel = kernel / sum

    img = GrayScale(img)
    out = img.copy()

    for i in range(center, np.shape(img)[0] - center):
        for j in range(center, np.shape(img)[1] - center):
            out[i, j] = np.sum(
                kernel * img[i - center:i + center + 1, j - center:j + center + 1])

    lap_o_gaus = np.zeros(np.shape(out))

    # Marr Hildreth method
    for i in range(1, np.shape(out)[0] - 1):
        for j in range(1, np.shape(out)[1] - 1):
            quad_1 = out[i, j] + out[i + 1, j] + out[i, j + 1] + out[i + 1, j + 1]
            quad_2 = out[i, j] + out[i - 1, j] + out[i, j - 1] + out[i - 1, j - 1]
            quad_3 = out[i, j] + out[i - 1, j] + out[i, j + 1] + out[i - 1, j + 1]
            quad_4 = out[i, j] + out[i + 1, j] + out[i, j - 1] + out[i + 1, j - 1]
            quad = [quad_1, quad_2, quad_3, quad_4]
            if max(quad) > 0 and min(quad) < 0:
                lap_o_gaus[i, j] = 255

    return lap_o_gaus


if __name__ == "__main__":
    img = np.array(Image.open(
        'Images/capture_1.png'), np.float64)
    log = LoG(img, 11, 1.5)

    f = plt.figure()
    plt.title("Laplacian of Gaussian edge detection operator")
    f.add_subplot(1, 1, 1)
    plt.imshow(log, cmap='gray', interpolation='nearest')
    plt.show(block=True)
