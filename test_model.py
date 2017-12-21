import cv2
import numpy
import file_to_numpy
import face_recognizer
import person_recognizer
import tensorflow as tf

# image = cv2.imread("/home/deepsee/PycharmProjects/tf_examples/crop_faces/face/image52856_163612.jpg")
#
# detector = face_recognizer.face_model()
# dataProb = detector.detect_face(image)
#
# print dataProb
# index = numpy.arange(5)
# numpy.random.shuffle(index)
# data = numpy.array([[0,1],[2,3],[4,5],[6,7],[8,9]])
# print data[index]
# print data
# i_f = file_to_numpy.imageFormatter()
# i_f.readImage("/home/deepsee/PycharmProjects/tf_examples/face_collection",
#               "face_pete_0.jpg", numpy.array([0,1]), i_f.train_data)
# print i_f.train_data

# sess = tf.Session()
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]), 'w1')
# b = tf.Variable(tf.zeros([10]), 'b1')
# y = tf.matmul(x, W) + b
# y_ = tf.placeholder(tf.float32, [None, 10])

# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
# saver.restore(sess, "/home/deepsee/test_checkpoint/1/1.ckpt")
#
# x2 = tf.placeholder(tf.float32, [None, 784])
# W2 = tf.Variable(tf.zeros([784, 10]), 'w1')
# b2 = tf.Variable(tf.zeros([10]), 'b1')
# y2 = tf.matmul(x2, W2) + b2
# y2_ = tf.placeholder(tf.float32, [None, 10])
#
# saver2 = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
# saver2.restore(sess, "/home/deepsee/test_checkpoint/2/2.ckpt")


# detector = face_recognizer.face_model(sess)
# detector2 = person_recognizer.identify_model(sess)
# face = face_recognizer.face_model()
# person = person_recognizer.identify_model()
