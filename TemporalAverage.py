from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from GrayScale import GrayScale
from CaptureVideo import CaptureVideo
import cv2

def TemporalAverage(video_handle):
    """
    Finds the average pixel values for set of image frames in a video. 
    Input:  Handle to the video file
    Output: numpy array with temporal average of the video frames
    """
    temp_avg = np.zeros((int(video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video_handle.get(
        cv2.CAP_PROP_FRAME_WIDTH))))

    while video_handle.isOpened():
        ret, frame = video_handle.read()
        if not ret:
            break
        # Converts input RGB frames to Grayscale and adds the pixel values of successive frames
        temp_avg += GrayScale(frame)
    # Find the average of each pixel in the video
    temp_avg = temp_avg / video_handle.get(cv2.CAP_PROP_FRAME_COUNT)

    return temp_avg


if __name__ == "__main__":
    # Comment one of the two lines below
    file_name = CaptureVideo('average_test')        # Capture a new video
    # file_name = 'average_test'                      # Use an existing video

    # Get the handle to the video frames
    cap = cv2.VideoCapture('Videos/{}.mov'.format(file_name))

    avg_img = TemporalAverage(cap)

    plt.imshow(avg_img, cmap='gray', interpolation='nearest')
    plt.axis('on')
    plt.xlabel('{} pixels'.format(np.shape(avg_img)[1]))
    plt.ylabel('{} pixels'.format(np.shape(avg_img)[0]))
    plt.title("Average pixel values of Video frames")
    plt.show()
