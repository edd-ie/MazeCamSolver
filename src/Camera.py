import cv2
import numpy as np
import glob


class Camera:
    camObj = cv2.VideoCapture(0)

    def __init__(self):
        self.camObj.set(3, 640)

    def openCam(self):
        # Check if the camera opened successfully
        if not self.camObj.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            # Capture frame-by-frame
            ret, frame = self.camObj.read()

            # If frame is read correctly, ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            cv2.imshow('Camera', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture
        self.__del__()


    def calibrate(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('*.jpg')

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()



    def __del__(self):
        self.camObj.release()
        cv2.destroyAllWindows()
