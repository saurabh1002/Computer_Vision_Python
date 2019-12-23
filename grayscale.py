from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def GrayScale(img):
    """
    Converts an input color image to grayscale.

    Input:  numpy array of image
    Output: numpy array of image conerted to Grayscale 
    """
    if np.ndim(img) > 2:
        gray_img = np.zeros((np.shape(img)[0], np.shape(img)[1]), np.uint8)
        gray_img = np.mean(img, 2)
        return gray_img
    else:
        return img

def SaveImage(img, name = 'Default.jpg'):
    """
    Saves the input image to a file.

    Input:  numpy array of image to be saved
            string with the name and extension of the file (Default name: 'Default.jpg')
    """
    Image.fromarray(img).convert("L").save(name)


if __name__ == "__main__":

    img = np.array(Image.open('saltnpepper.png'))
    gray_img = GrayScale(img)

    plt.imshow(gray_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(gray_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(gray_img)[0]))
    plt.title("Grayscale Image")
    plt.show()

