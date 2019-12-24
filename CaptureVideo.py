# This script is used to capture a video
# Press 'q' to end recording

import cv2

def CaptureVideo(file_name, port = 0):
    """
    Captures video from the webcam and saves it as a .mov file in the /Videos folder.
    Input:  File name
            webcam port (default = 0)
    Output: String of file name
    """
    cap = cv2.VideoCapture(port)
    cv2.namedWindow("video_capture")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('Videos/{}.mov'.format(file_name), fourcc, 20.0, (640, 480), True)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if True:
            # write the flipped frame
            out.write(frame)

            cv2.imshow('Press q to save video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return file_name

if __name__ == "__main__":
    CaptureVideo('test_video')
