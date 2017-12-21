import tensorflow as tf
import numpy

class face_model:
    def __init__(self):
        # with tf.variable_scope("face_scope"):
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

        self.faceGraph = tf.Graph()
        self.sess = tf.Session(graph=self.faceGraph)

        with self.faceGraph.as_default():
            self.x = tf.placeholder(tf.float32, shape=[None, 64 * 64 * 3])

            self.x_image = tf.reshape(self.x, [-1, 64, 64, 3])

            self.W_conv1 = weight_variable([5, 5, 3, 8], "f_W_conv1")
            self.b_conv1 = bias_variable([8], "f_b_conv1")

            self.h_conv1 = tf.nn.relu(conv2d(self.x_image, self.W_conv1) + self.b_conv1)
            self.h_pool1 = max_pool_2x2(self.h_conv1)

            self.W_conv2 = weight_variable([5, 5, 8, 16], "f_W_conv2")
            self.b_conv2 = bias_variable([16], "f_b_conv2")

            self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)
            self.h_pool2 = max_pool_2x2(self.h_conv2)

            self.W_conv3 = weight_variable([5, 5, 16, 16], "f_W_conv3")
            self.b_conv3 = bias_variable([16], "f_b_conv3")

            self.h_conv3 = tf.nn.relu(conv2d(self.h_pool2, self.W_conv3) + self.b_conv3)
            self.h_pool3 = max_pool_2x2(self.h_conv3)

            self.W_fc1 = weight_variable([8 * 8 * 16, 256], "f_W_fc1")
            self.b_fc1 = bias_variable([256], "f_b_fc1")

            self.h_pool3_flat = tf.reshape(self.h_pool3, [-1, 8 * 8 * 16])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool3_flat, self.W_fc1) + self.b_fc1)

            self.W_fc2 = weight_variable([256, 2], "f_W_fc2")
            self.b_fc2 = bias_variable([2], "f_b_fc2")

            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, 1)

            self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

            self.softmax = tf.nn.softmax(self.y_conv)

            saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith("f_")])
            self.sess.run(tf.global_variables_initializer())
            saver.restore(self.sess, "/home/deepsee/facemodel_checkpoint/faceRecog.ckpt")


    def detect_face(self, imgArray):
        processArray = imgArray.flatten().astype(numpy.float32)
        processArray = numpy.multiply(processArray, 1.0 / 255.0)[numpy.newaxis]
        # mean = numpy.mean(processArray)
        # processArray -= mean
        return self.softmax.eval(feed_dict={self.x:processArray}, session=self.sess)