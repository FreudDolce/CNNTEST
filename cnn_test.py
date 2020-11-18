import tensorflow as tf
import numpy as np
import tensorboard as tb
from tensorflow.examples.tutorials.mnist import input_data

MNIST_FOLDER_PATH = r'/home/ji/Documents/mnist/'
mnist = input_data.read_data_sets(MNIST_FOLDER_PATH, one_hot=True)

LEARNING_RATE = 0.1
EPOCHS = 10
BATCH_SIZE = 50

def var_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.reduce_mean('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('W1'):
    W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.3))

b1 = tf.Variable(tf.random_normal([300]), name='b1')

W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.3), name='W2')
#var_summaries(W2)
b2 = tf.Variable(tf.random_normal([10]),name='b2')

hiden_out = tf.add(tf.matmul(x, W1), b1)
hiden_out = tf.nn.relu(hiden_out)

y_ = tf.nn.softmax(tf.add(tf.matmul(hiden_out, W2), b2))
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * 
                                              tf.log(y_clipped) + (1 - y) * 
                                              tf.log(1 - y_clipped), axis=1))
var_summaries(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / BATCH_SIZE)
    writer = tf.summary.FileWriter(r'/home/ji/Documents/mnist/log/', sess.graph)
    for epoch in range(EPOCHS):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size = BATCH_SIZE)
            a, b, c = sess.run([merged, optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        writer.add_summary(a, epoch)   
        print(avg_cost)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    writer.close()
