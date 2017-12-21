import imutils
import argparse
import time
import cv2
import numpy as np
import sys
import glob, os
import datetime
import face_recognizer
import person_recognizer
import tensorflow as tf
import threading

class faceScanner:
    def __init__(self):
        self.detector = face_recognizer.face_model()
        # self.detector2 = person_recognizer.identify_model(sess)

    def pyramid(self, image, scale=1.5, minSize=(64, 64)):
        # yield the original image
        yield image

        # keep looping over the pyramid
        while True:
            # compute the new dimensions of the image and resize it
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)

            # if the resized image does not meet the supplied minimum
            # size, then stop constructing the pyramid
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break

            # yield the next image in the pyramid
            yield image


    def sliding_window(self, image, stepSize, windowSize):
        # slide a window across the image
        for y in xrange(0, image.shape[0], stepSize):
            for x in xrange(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


    def scan_image(self, image, scale_to_skip = 4):

        (winW, winH) = (64, 64)
        steps_to_skip = 0
        facesFound = []

        # debugFaceCounter = 0
        # loop over the image pyramid
        for resized in self.pyramid(image, scale=1.2):
            if scale_to_skip > 0:
                scale_to_skip -= 1
                continue
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in self.sliding_window(resized, stepSize=4, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                if steps_to_skip > 0:
                    steps_to_skip -= 4
                    continue

                window_crop = resized[y:y+winH,x:x+winW]
                dataProb = self.detector.detect_face(window_crop)
                #global_variables since we do not have a classifier, we'll just draw the window
                if dataProb[0][1] >= 0.95:
                    # cv2.imshow('face{}'.format(debugFaceCounter), window_crop)
                    processArray = window_crop.flatten().astype(np.float32)
                    facesFound.append(processArray)
                    # debugFaceCounter += 1
                    steps_to_skip += 32

        print len(facesFound)
        return np.array(facesFound)

# for filename in imgList:
#     image = cv2.imread(filename)
#     nextIndex = scan.scan_image(image, 'wenxin', startindex=index)
#     index = nextIndex + 1
cap = cv2.VideoCapture(0)
process = faceScanner()
recognizer = person_recognizer.identify_model()

# dirname = '/home/deepsee/PycharmProjects/tf_examples/face_collection/detect'
# os.chdir(dirname)
# while(True):
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frameq
    cv2.imshow('color',frame)
    # cv2.imshow('gray',gray)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    k = cv2.waitKey(30)

    if k & 0xFF == ord('q') or k %256 == 27:
        # ESC pressed
        print("Escape/q hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        # while (True):
        #     ret, frame = cap.read()
        #     cv2.imshow('scanning', frame)
        #     k2 = cv2.waitKey(30)
        #
        #     if k2 & 0xFF == ord('q') or k2 %256 == 27:
        #         # ESC pressed
        #         cv2.destroyWindow('scanning')
        #         break
        faces = process.scan_image(frame)
        if faces.size > 0:
            data, probs = recognizer.recognize_face(faces)
            print data
            print probs


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

