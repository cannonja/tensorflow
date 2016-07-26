import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


## Read and unpack images
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

## Launch interactive session
## Interavtive session allows us to interleave operations that build graph
## with those that run/execute graph
## Regular session makes you build first, then execute
sess = tf.InteractiveSession()

########## Build softmax regression single layer ####################

## Start building computation graph by defining placeholders (inputs and target labels)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

## Then define variables, generally model parameters (weights, biases)
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

## Initialize variables so they can be used within the session
sess.run(tf.initialize_all_variables())

## Add activation node
y = tf.nn.softmax(tf.matmul(x,W) + b)

## Define Loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


########### Train the model ###########################################

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for i in range(1000):
      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})


########### Evaluate ##################################################
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
