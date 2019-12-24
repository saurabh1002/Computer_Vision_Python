# Press 'Spacebar' to capture an image.
# Press 'Esc' to close.

import cv2

def CaptureImage(file_name, port = 0):
    """
    Captures image from the webcam and saves it as a .png file in the /Images folder.
    Input:  File name
            webcam port (default = 0)
    """
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("image_capture")
    img_counter = 0
    
    while True:
        ret, frame = cam.read()
        cv2.imshow("image_capture", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        elif k % 256 == 32:
            # SPACE pressed
            cv2.imwrite(
                "Images/{}{}.png".format(file_name, img_counter), frame)
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    CaptureImage('test_image')