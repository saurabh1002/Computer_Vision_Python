from PIL import Image
import numpy as np
from GaussianBlur import GaussianBlur
from SobelEdgeDetection import SobelEdgeDetector
from matplotlib import pyplot as plt

def NonMaxSuppression(edge_mag, edge_dir):
    """
    Performs Non-Maximal Supression in a direction perpendicular to the edge direction obtained from Sobel Edge detection.
    Input:  numpy array of edge magnitudes obtained from the Sobel operator
            numpy array of the edge directions (0 - 360 degrees) obtained from the Sobel operator
    Output: Non-Maximal suppressed image numpy array
    """
    out = np.zeros(np.shape(edge_mag))
    edge_dir = edge_dir * 180. / np.pi
    for i in range(1, np.shape(out)[0] - 1):
        for j in range(1, np.shape(out)[1] - 1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= edge_dir[i, j] < 22.5) or (157.5 <= edge_dir[i, j] <= 180):
                    q = edge_mag[i, j+1]
                    r = edge_mag[i, j-1]
                #angle 45
                elif (22.5 <= edge_dir[i, j] < 67.5):
                    q = edge_mag[i+1, j-1]
                    r = edge_mag[i-1, j+1]
                #angle 90
                elif (67.5 <= edge_dir[i, j] < 112.5):
                    q = edge_mag[i+1, j]
                    r = edge_mag[i-1, j]
                #angle 135
                elif (112.5 <= edge_dir[i, j] < 157.5):
                    q = edge_mag[i-1, j-1]
                    r = edge_mag[i+1, j+1]
                
                if (edge_mag[i, j] >= q) and (edge_mag[i, j] >= r):
                    out[i, j] = edge_mag[i, j]
                else:
                    out[i, j] = 0

            except IndexError:
                pass
    return out

def HysteresisThresholding(img, lower_level = 10, upper_level = 40, weak = 25, strong = 255):
    """
    Performs Hysteresis Thresholding on the input suppressed image.
    Input:  numpy array of non-maximal suppressed image
            lower level of thresholding (default = 10; Pixels below this value are set to 0)
            upper level of thresholding (default = 40; Pixels above this value are set to 'strong')
            value of pixels between lower and upper thresholds (default = 25)
            value of pixels above upper threshold (default = 255)
    Output: numpy array of hysteresis thresholded image
    """
    thresh = img.copy()
    for i in range(0, np.shape(img)[0]):
        for j in range(0, np.shape(img)[1]):
            if img[i, j] >= upper_level:
                thresh[i, j] = strong
            elif img[i, j] < lower_level:
                thresh[i, j] = 0.0
            else:
                thresh[i, j] = weak

    return thresh

def EdgeTracking(img, weak = 25, strong = 255):
    """
    Interpolates the edges by setting weak pixels with a neighbouring strong pixel as a strong pixel.
    Input:  numpy array of hysteresis thresholded image
            value of weak pixels (default = 25; must be same as that set in thresholding operation)
            value of the strong pixels (default = 255)
    """
    edge_track = img.copy()
    for i in range(1, np.shape(img)[0] - 1):
        for j in range(1, np.shape(img)[1] - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        edge_track[i, j] = strong
                    else:
                        edge_track[i, j] = 0
                except IndexError:
                    pass

    return edge_track

def CannyEdgeDetector(img, window_gaussian = 5, window_sobel = 5):
    """
    Performs Canny Edge Detection
    Input:  numpy array of input Grayscale image
            window size for Gaussian smoothing (default = 5)
            window size for Sobel edge detection (default = 5)
    Output: numpy array of Canny Edge Detected images
    """
    blur_img = GaussianBlur(img, window_gaussian)                   # Gaussian Smoothing
    sob_mag, sob_dir = SobelEdgeDetector(blur_img, window_sobel)    # Sobel Edge Detection
    non_supress = NonMaxSuppression(sob_mag, sob_dir)               # Non Maximal Suppressing
    thresholded = HysteresisThresholding(non_supress)               # Hysteresis Thresholding
    canny_edges = EdgeTracking(thresholded)                         # Edge Extrapolation

    return blur_img, sob_mag, non_supress, thresholded, canny_edges

if __name__ == "__main__":
    img = np.array(Image.open(
        'Images/SRA Khopdi Baba Grayscale.jpg'), np.float64)
    blur_img, sob_mag, non_max, hyst_thresh, canny_edges = CannyEdgeDetector(img)

    f = plt.figure()
    plt.title("Canny Edge Detector")
    f.add_subplot(2, 2, 1)
    plt.title("Sobel Edge Detector")
    plt.axis('off')
    plt.imshow(sob_mag, cmap='gray', interpolation='nearest')
    f.add_subplot(2, 2, 2)
    plt.title("Non Maximal Suppressed Image")
    plt.axis('off')
    plt.imshow(non_max, cmap='gray', interpolation='nearest')
    f.add_subplot(2, 2, 3)
    plt.title("Hysteresis Thresholded Image")
    plt.axis('off')
    plt.imshow(hyst_thresh, cmap='gray', interpolation='nearest')
    f.add_subplot(2, 2, 4)
    plt.title("Canny Edges")
    plt.axis('off')
    plt.imshow(canny_edges, cmap='gray', interpolation='nearest')
    plt.show(block=True)
