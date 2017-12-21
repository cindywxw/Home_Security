import glob, os
import numpy
from scipy import misc
import random

face_not_person = False

class imageFormatter:
    train_data = []
    test_data = []
    readPos = 0

    def readImage(self, directory, file, label, data_set):
        try:
            img = misc.imread("{}/{}".format(directory, file), False, 'RGB')
            imgarray = img.flatten().astype(numpy.float32)
            imgarray = numpy.multiply(imgarray, 1.0 / 255.0)
            # mean = numpy.mean(imgarray)
            # imgarray -= mean
            defarray = numpy.array(label).astype(numpy.float32)
            dataarray = numpy.array([imgarray, defarray])
            data_set.append(dataarray)
            return True
        except:
            return False


    def load_from_dir(self, directory, label, train_count, test_count):
        if face_not_person:
            dir = "/home/deepsee/PycharmProjects/tf_examples/crop_faces/{}".format(directory)
        else:
            dir = "/home/deepsee/PycharmProjects/tf_examples/face_collection/face-crops/{}".format(directory)
        os.chdir(dir)
        list = []
        for file in glob.glob("*.jpg"):
            list.append(file)

        for i in range(0, train_count):
            file = list.pop(random.randrange(0, len(list)-1))
            self.readImage(dir, file, label, self.train_data)
            if (i+1) % 1000 == 0:
                print '{} training files loaded'.format(i+1)

        for i in range(0, test_count):
            file = list.pop(random.randrange(0, len(list) - 1))
            self.readImage(dir, file, label, self.test_data)
            if (i+1) % 500 == 0:
                print '{} testing files loaded'.format(i+1)


    def load_images(self, train_count = 2500, test_count = 250):
        if face_not_person:
            self.load_from_dir("face-full", numpy.array([0, 1]), train_count, test_count)
            self.load_from_dir("non-face", numpy.array([1, 0]), 3*train_count, test_count)
            self.load_from_dir("caltech_resized", numpy.array([1, 0]), 2*train_count, test_count)
        else:
            self.load_from_dir("dad", numpy.array([1, 0, 0, 0, 0, 0, 0]), train_count, test_count)
            self.load_from_dir("pete", numpy.array([0, 1, 0, 0, 0, 0, 0]), train_count, test_count)
            self.load_from_dir("mark", numpy.array([0, 0, 1, 0, 0, 0, 0]), train_count, test_count)
            self.load_from_dir("lauren", numpy.array([0, 0, 0, 1, 0, 0, 0]), train_count, test_count)
            self.load_from_dir("mom", numpy.array([0, 0, 0, 0, 1, 0, 0]), train_count, test_count)
            self.load_from_dir("grace", numpy.array([0, 0, 0, 0, 0, 1, 0]), train_count, test_count)
            self.load_from_dir("wenxin", numpy.array([0, 0, 0, 0, 0, 0, 1]), train_count, test_count)
            # self.load_from_dir("face-full-thread", numpy.array([0, 0, 0, 0, 0, 0, 1]), 2*train_count, 2*test_count)
        print "files loaded"
        trainIndex = numpy.arange(len(self.train_data))
        numpy.random.shuffle(trainIndex)
        trainDataArray = numpy.array(self.train_data)
        self.train_data = trainDataArray[trainIndex].tolist()

        testIndex = numpy.arange(len(self.test_data))
        numpy.random.shuffle(testIndex)
        testDataArray = numpy.array(self.test_data)
        self.test_data = testDataArray[testIndex].tolist()

    # def load_test_images(self):
    #
    #     self.load_from_dir("face-full", numpy.array([0, 1]), 10000)
    #     self.load_from_dir("non-face-full", numpy.array([1, 0]), 10000)
    #     print "files loaded"
    #     index = numpy.arange(len(self.dataformat))
    #     numpy.random.shuffle(index)
    #     dataArray = numpy.array(self.dataformat)
    #     self.dataformat = dataArray[index].tolist()

    def next_train_batch(self, batchsize):
        img = []
        label = []
        for i in range(self.readPos, self.readPos + batchsize):
            self.readPos = i
            if i >= len(self.train_data):
                self.readPos = i - len(self.train_data) - 1

            img.append(self.train_data[self.readPos][0])
            label.append(self.train_data[self.readPos][1])

        imgArray = numpy.array(img)
        labelArray = numpy.array(label)
        self.readPos += 1
        return imgArray,labelArray

    def batch_all_test(self):
        img = []
        label = []
        for i in range(0, len(self.test_data)):

            img.append(self.test_data[i][0])
            label.append(self.test_data[i][1])

        imgArray = numpy.array(img)
        labelArray = numpy.array(label)

        return imgArray, labelArray