# import subprocess as sbp
# import os
#
# os.chdir("/home/deepsee/PycharmProjects/tf_examples/crop_faces/256_ObjectCategories")
# path=os.getcwd()
# fol = os.listdir(path)
# p2 = os.path.join(path,'Merged')
#
# sbp.Popen(['mkdir','Merged'])
#
# for i in fol:
#     if os.path.isdir(i)==True:
#         if i!='Merged':
#             p1 = os.path.join(path,i)
#             p3 = 'cp -r "' + p1 +'"/* ' + p2
#             sbp.Popen(p3,shell=True)

import glob, os
import random
from PIL import Image
import threading
import numpy

imgList = []

os.chdir("/home/deepsee/PycharmProjects/tf_examples/crop_faces/Merged_caltech256")
for file in glob.glob("*.jpg"):
    imgList.insert(0, file)

index = numpy.arange(len(imgList))
numpy.random.shuffle(index)
dataArray = numpy.array(imgList)
imgList = dataArray[index].tolist()

samplefiles = []
size = len(imgList)

def split_by(sequence, length):
    iterable = iter(sequence)
    def yield_length():
        for i in xrange(length):
             yield iterable.next()
    while True:
        res = list(yield_length())
        if not res:
            return
        yield res
num_thread = 4
seg_length = size/num_thread
# [imgList[x:x+seg_length] for x in range(0,len(imgList),seg_length)]

list = list(split_by(imgList, seg_length))
print (len(list[0]))


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
            new_img.save("/home/deepsee/PycharmProjects/tf_examples/crop_faces/caltech_resized/{}".format(filename),
                         "JPEG", optimize=True)
            print filename

threads = []
for i in range(num_thread - 1):
    threads.append([])
    threads[i] = myThread(list[i])
    threads[i].start()

for filename in list[4]:
    img = Image.open(filename)
    new_img = img.resize((64, 64))
    new_img.save("/home/deepsee/PycharmProjects/tf_examples/crop_faces/caltech_resized/{}".format(filename),
                 "JPEG", optimize=True)
    print filename