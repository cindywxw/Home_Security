import imutils
import argparse
import time
import cv2
import numpy as np
import sys
import glob, os
import datetime
import face_recognizer
import tensorflow as tf
import threading


class faceScanner:
    def __init__(self):
        self.detector = face_recognizer.face_model()

    def pyramid(self, image, scale=1.5, minSize=(128, 128)):
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


    def scan_image(self, image, folder, scale_to_skip = 4, threadnumber = 0):

        (winW, winH) = (64, 64)
        steps_to_skip = 0
        cropcounter = 0
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
                # since we do not have a classifier, we'll just draw the window
                if dataProb[0][1] >= 0.99:
                    # clone = resized.copy()
                    # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 2)
                    # cv2.imshow("Window", clone)
                    cv2.imwrite("/home/deepsee/PycharmProjects/tf_examples/face_collection/face-crops/{}/{}_{}.jpg".format(
                        folder, datetime.datetime.now(), threadnumber), window_crop)
                    cropcounter += 1
                    steps_to_skip += 32
                    # cv2.waitKey(1)
                # else:
                    # clone = resized.copy()
                    # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                    # cv2.imshow("Window", clone)
                    # cv2.waitKey(1)

        print "{} faces found".format(cropcounter)
        return cropcounter

imgList = []
os.chdir("/home/deepsee/PycharmProjects/tf_examples/face_collection/lauren")
for file in glob.glob("*.png"):
    imgList.append(file)

scan = faceScanner()
index = 0
# for filename in imgList:
#     image = cv2.imread(filename)
#     nextIndex = scan.scan_image(image, 'wenxin', startindex=index)
#     index = nextIndex + 1

size = len(imgList)
num_thread = 7
seg_length = size/num_thread
list = [imgList[x:x+seg_length] for x in xrange(0,len(imgList),seg_length)]

class myThread (threading.Thread):
    def __init__(self, split_list, threadIndex):
        threading.Thread.__init__(self)
        self.split_list = split_list
        self.threadIndex = threadIndex
    def run(self):
        for filename in self.split_list:
            image = cv2.imread(filename)
            nextIndex = scan.scan_image(image, 'lauren', threadnumber=self.threadIndex)

threads = []
for i in range(num_thread):
    threads.append([])
    threads[i] = myThread(list[i], i)
    threads[i].start()

if (size % num_thread != 0):
    for filename in list[num_thread]:
        image = cv2.imread(filename)
        nextIndex = scan.scan_image(image, 'lauren')
        index = nextIndex + 1