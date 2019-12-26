from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt


def FirstOrderDifference(img):
    """
    Performs a first order difference on the grayscale image pixels to detect edges.
    Input:  numpy array of grayscale image
    Output: numpy array of image with the filtered edges
    """
    kernel = np.array([[2.0, -1.0], [-1.0, 0.0]])
    gray_img = GrayScale(img)
    conv_img = gray_img.copy()
    for i in range(0, np.shape(gray_img)[0] - 1):
        for j in range(0, np.shape(gray_img)[1] - 1):
            sum = np.sum(kernel * gray_img[i:i + 2, j:j + 2])
            conv_img[i, j] = abs(sum)
    return conv_img


def Improved_FirstOrderDifference(img, direction = 1):
    """
    Imporved version of the First Order Difference method to detect edges.
    Input:  numpy array of grayscale image
            direction of edge detection (direction = 1 for horizontal edge; direction = 0 for vertical edges)
    Output: numpy array of image with the filtered edges
    """
    kernel = np.array([1.0, 0.0, -1.0])
    gray_img = GrayScale(img)
    if direction:
        conv_img = gray_img.copy()
        for i in range(1, np.shape(gray_img)[0] - 1):
            for j in range(0, np.shape(gray_img)[1]):
                sum = np.sum(kernel * gray_img[i - 1:i + 2, j])
                conv_img[i, j] = abs(sum)
    else:
        gray_img = np.transpose(gray_img)
        conv_img = gray_img.copy()
        for i in range(1, np.shape(gray_img)[0] - 1):
            for j in range(0, np.shape(gray_img)[1]):
                sum = np.sum(kernel * gray_img[i - 1:i + 2, j])
                conv_img[i, j] = abs(sum)
        conv_img = np.transpose(conv_img)
    return conv_img

def RobertsCrossDiff(img):
    """
    Performs edge detection using the Robert's Cross Difference kernel.
    Input:  numpy array of grayscale image
    Output: numpy array of image with the filtered edges
    """
    kernel_p = np.array([[1, 0], [0, -1]])
    kernel_n = np.array([[0, 1], [-1, 0]])

    gray_img = GrayScale(img)
    conv_img = gray_img.copy()
    for i in range(0, np.shape(gray_img)[0] - 1):
        for j in range(0, np.shape(gray_img)[1] - 1):
            sum_p = abs(np.sum(kernel_p * gray_img[i:i + 2, j:j + 2]))
            sum_n = abs(np.sum(kernel_n * gray_img[i:i + 2, j:j + 2]))
            conv_img[i, j] = max([sum_p, sum_n])
    return conv_img

if __name__ == "__main__":
    img = np.array(Image.open(
        'Images/capture_1.png'), np.float64)
    img_1 = FirstOrderDifference(img)
    img_2_h = Improved_FirstOrderDifference(img, 1)
    img_2_v = Improved_FirstOrderDifference(img, 0)
    img_3 = RobertsCrossDiff(img)

    f = plt.figure()
    plt.title("Comparison of First order Difference methods")
    f.add_subplot(1, 3, 1)
    plt.title("First Order Difference")
    plt.imshow(img_1, cmap='gray', interpolation='nearest')
    f.add_subplot(1, 3, 2)
    plt.title("Imporved First Order Difference")
    plt.imshow((img_2_h + img_2_v) / 2.0, cmap='gray', interpolation='nearest')
    f.add_subplot(1, 3, 3)
    plt.title("Roberts Cross Difference")
    plt.imshow(img_3, cmap='gray', interpolation='nearest')
    plt.show(block=True)
