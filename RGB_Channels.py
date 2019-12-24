from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def RGB_Channels(img):
    """
    Segregates the individual RGB color channels of an input RGB image.
    Input:  numpy array of an RGB image
    Output: numpy arrays containing individual R, G, B color channels
    """
    img_R = img.copy()
    img_R[:, :, (1, 2)] = 0

    img_G = img.copy()
    img_G[:, :, (0, 2)] = 0

    img_B = img.copy()
    img_B[:, :, (0, 1)] = 0

    return img_R, img_G, img_B


if __name__ == "__main__":
    
    img = np.array(Image.open('Images/SRA Khopdi Baba.jpg'))
    img_r, img_g, img_b = RGB_Channels(img)

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.title("Red Channel", color='red')
    plt.imshow(img_r, cmap='gray', interpolation='nearest')
    f.add_subplot(1, 3, 2)
    plt.title("Green Channel", color='green')
    plt.imshow(img_g, cmap='gray', interpolation='nearest')
    f.add_subplot(1, 3, 3)
    plt.title("Blue Channel", color='blue')
    plt.imshow(img_b, cmap='gray', interpolation='nearest')
    plt.show(block = True)
