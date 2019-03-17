import tensorflow as tf
import numpy as np

# functions
softmax = tf.nn.softmax
relu = tf.nn.relu

# layers
flatten = tf.layers.flatten
batch_norm = tf.contrib.layers.batch_norm

# blocks
def conv_block(x, w, b, bn_scope='conv_bn'):
  x = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME') + b
  # The same batch_norm usage as in [https://github.com/cbfinn/maml]
  # --> 1. batch_norm statistics (moving mean and variance) are neither updated nor used.
  #        So no need to explicitly update ops in tf.GraphKeys.UPDATE_OPS.
  # --> 2. Beta is used but gamma is ignored (scaling is turned off).
  # --> 3. beta is not updated over the inner-gradient loops. (only meta-training does)
  x = batch_norm(x, activation_fn=relu, scope=bn_scope, reuse=tf.AUTO_REUSE)
  return tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')

# training modules
def cross_entropy(logits, labels):
  return tf.losses.softmax_cross_entropy(logits=logits,
      onehot_labels=labels)

def accuracy(logits, labels):
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))
