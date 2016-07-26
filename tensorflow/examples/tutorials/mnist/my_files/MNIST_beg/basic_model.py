import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

## Read data and store in mnist class in train,
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

######## Implement simple soft-max regression###################

## Define model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

## Define loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


## Define training
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


## Initialize variables before training
init = tf.initialize_all_variables()

## Launch model in a session
sess = tf.Session()
sess.run(init)


## Train model 1000 times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

## Evaluate
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
