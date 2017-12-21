from tensorflow.examples.tutorials.mnist import input_data
import file_to_numpy
import numpy

data = file_to_numpy.imageFormatter()
data.load_images(10000,1000)

import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 64*64*3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

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

W_conv1 = weight_variable([5, 5, 3, 8], "f_W_conv1")
b_conv1 = bias_variable([8], "f_b_conv1")

x_image = tf.reshape(x, [-1,64,64,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 8, 16], "f_W_conv2")
b_conv2 = bias_variable([16], "f_b_conv2")

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 16, 16], "f_W_conv3")
b_conv3 = bias_variable([16], "f_b_conv3")

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([8 * 8* 16, 256], "f_W_fc1")
b_fc1 = bias_variable([256], "f_b_fc1")

h_pool3_flat = tf.reshape(h_pool3, [-1, 8*8*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([256, 2], "f_W_fc2")
b_fc2 = bias_variable([2], "f_b_fc2")

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

saver = tf.train.Saver([v for v in tf.global_variables() if v.name.startswith("f_")])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
saver.restore(sess, "/home/deepsee/facemodel_checkpoint/faceRecog.ckpt")

for i in range(10000):
  batch = data.next_train_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

test_batch = data.batch_all_test()
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

save_path = saver.save(sess, "/home/deepsee/facemodel_checkpoint/faceRecog.ckpt")


