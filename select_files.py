import glob, os
import random
from PIL import Image
import threading
import numpy

imgList = []


os.chdir("/home/deepsee/Downloads/caffe-master/FaceDetection_CNN-master/aflw/crop_images/face")
for file in glob.glob("*.jpg"):
    imgList.insert(0, file)


index = numpy.arange(len(imgList))
numpy.random.shuffle(index)
dataArray = numpy.array(imgList)
imgList = dataArray[index].tolist()

size = len(imgList)
num_thread = 7
seg_length = size/num_thread
list = [imgList[x:x+seg_length] for x in xrange(0,len(imgList),seg_length)]

# list = list(split_by(imgList, seg_length))
# print size
# print seg_length
# print (size % num_thread)
# print (len(list[num_thread]))

# samplefiles = []
# for i in range (0, size/10):
#     samplefiles.insert(0, imgList.pop(random.randrange(0, len(imgList) - 1)))
#
# for filename in imgList:
#     img = Image.open("/home/deepsee/Downloads/caffe-master/FaceDetection_CNN-master/aflw/crop_images/non-face/{}".format(filename))
#     new_img = img.resize((64, 64))
#     new_img.save("/home/deepsee/PycharmProjects/tf_examples/crop_faces/non-face-full/{}".format(filename), "JPEG", optimize=True)
#     print filename


class myThread (threading.Thread):
    def __init__(self, split_list):
        threading.Thread.__init__(self)
        self.split_list = split_list
    def run(self):
        for filename in self.split_list:
            img = Image.open(filename)
            new_img = img.resize((64, 64))
            new_img.save("/home/deepsee/PycharmProjects/tf_examples/crop_faces/face-full-thread/{}".format(filename),
                         "JPEG", optimize=True)
            print filename

threads = []
for i in range(num_thread):
    threads.append([])
    threads[i] = myThread(list[i])
    threads[i].start()

if (size % num_thread != 0):
    for filename in list[num_thread]:
        img = Image.open(filename)
        new_img = img.resize((64, 64))
        new_img.save("/home/deepsee/PycharmProjects/tf_examples/crop_faces/face-full-thread/{}".format(filename),
                     "JPEG", optimize=True)
        print filename