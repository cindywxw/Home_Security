import tensorflow as tf
import numpy

class identify_model:
    def __init__(self):
        # with tf.variable_scope("person_scope"):
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        self.personGraph = tf.Graph()
        self.psess = tf.Session(graph=self.personGraph)

        with self.personGraph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, 64 * 64 * 3])

            self.x_image = tf.reshape(self.x, [-1, 64, 64, 3])

            self.W_conv1 = weight_variable([5, 5, 3, 8], "p_W_conv1")
            self.b_conv1 = bias_variable([8], "p_b_conv1")

            self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)

            self.W_conv2 = weight_variable([5, 5, 8, 16], "p_W_conv2")
            self.b_conv2 = bias_variable([16], "p_b_conv2")

            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = max_pool_2x2(self.h_conv2)

            self.W_conv3 = weight_variable([5, 5, 16, 32], "p_W_conv3")
            self.b_conv3 = bias_variable([32], "p_b_conv3")

            self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)
            self.h_pool3 = max_pool_2x2(self.h_conv3)

            self.W_fc1 = weight_variable([8 * 8 * 32, 1024], "p_W_fc1")
            self.b_fc1 = bias_variable([1024], "p_b_fc1")

            self.h_pool3_flat = tf.reshape(self.h_pool3, [-1, 8 * 8 * 32])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool3_flat, self.W_fc1) + self.b_fc1)

            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, 1)

            self.W_fc2 = weight_variable([1024, 7], "p_W_fc2")
            self.b_fc2 = bias_variable([7], "p_b_fc2")

            self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

            self.softmax = tf.nn.softmax(self.y_conv)

            self.allmax = tf.arg_max(self.softmax, 1)
            psaver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith("p_")])
            self.psess.run(tf.global_variables_initializer())
            psaver.restore(self.psess, "/home/deepsee/personmodel_checkpoint/personRecog.ckpt")


    def recognize_face(self, imagesArray):
        processArray = numpy.multiply(imagesArray, 1.0 / 255.0)[numpy.newaxis]
        softs = self.softmax.eval(feed_dict={self.x:processArray[0]}, session=self.psess)
        data = self.allmax.eval(feed_dict={self.x:processArray[0]}, session=self.psess)
        arrayPosition = 0
        probabilities = []
        for index in data:
            probabilities.append(softs[arrayPosition][index])
            arrayPosition += 1
        return data, probabilities