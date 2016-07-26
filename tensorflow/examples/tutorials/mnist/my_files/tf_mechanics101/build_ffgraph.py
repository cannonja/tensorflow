import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

## Flag crap from Google - I have no idea what is does
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                             'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                             'for unit testing.')

## Download data and unpack
## data_sets is a custom DataSet data type
data_sets = input_data.read_data_sets(FLAGS.train_dir, FLAGS.fake_data)

## Initialize graph and start drawing on it
with tf.Graph().as_default():
    ## Prepare inputs and placeholders
    images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size,
                                                            mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size))

    ## mnist.inference() builds feed-forward portion of graph
    ## It takes the images placeholder and two integers, each representing the
    ## number of neurons for the respective hidden layers and returns logits
    logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
    loss = mnist.loss(logits, labels_placeholder)
    train_op = mnist.training(loss, FLAGS.learning_rate)
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    ## Initialize variables, run session, and write summary writer data
    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    sess.run(init)
