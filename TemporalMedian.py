from PIL import Image
import numpy as np
from GrayScale import GrayScale
from matplotlib import pyplot as plt
from GrayScale import GrayScale
from CaptureVideo import CaptureVideo
import cv2

def TemporalMedian(video_handle):
    """
    Finds the median pixel values for set of image frames in a video. 
    Can be used to extract the stationary background from the video.
    Input:  handle to the video file
    Output: numpy array of the temporal median of the video frames
    """
    a = np.zeros((int(video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video_handle.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_handle.get(cv2.CAP_PROP_FRAME_COUNT))))

    i = 0
    while video_handle.isOpened():
        ret, frame = video_handle.read()
        if not ret:
            break
        a[:, :, i] = GrayScale(frame)   # Converts input RGB frames to Grayscale and append it to the 3rd dimension of np array 'a'
        i += 1
    temp_median = np.median(a, 2)      # Find the median of each pixel in the video i.e. median along the 3rd dimension of 'a'

    return temp_median

if __name__ == "__main__":
    # Comment one of the two lines below
    # file_name = CaptureVideo('median_test')                     # Capture a new video
    file_name = 'median_test'                                   # Use an existing video

    cap = cv2.VideoCapture('Videos/{}.mov'.format(file_name))   # Get the handle to the video frames

    back_img = TemporalMedian(cap)

    plt.imshow(back_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(back_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(back_img)[0]))
    plt.title("Extracted Background Image")
    plt.show()
