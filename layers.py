import tensorflow as tf
import numpy as np

# functions
softmax = tf.nn.softmax
relu = tf.nn.relu

# layers
flatten = tf.layers.flatten
batch_norm = tf.contrib.layers.batch_norm

def pool(x, **kwargs):
  return tf.contrib.layers.max_pool2d(x, 2)

# blocks
def conv_block(x, (w, b), activation=None, scope='conv_block',
    reuse=None):
  x = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME') + b
  x = batch_norm(x, reuse=reuse, scope=scope)
  x = x if activation is None else activation(x)
  return pool(x)

def dense_block(x, (w, b), activation=None, scope='dense_block',
    reuse=None):
  x = tf.matmul(x,w) + b
  x = batch_norm(x, reuse=reuse, scope=scope)
  return x if activation is None else activation(x)

# training modules
def cross_entropy(logits, labels):
  return tf.losses.softmax_cross_entropy(logits=logits,
      onehot_labels=labels)

def weight_decay(decay, var_list=None):
  var_list = tf.trainable_variables() if var_list is None else var_list
  return decay*tf.add_n([tf.nn.l2_loss(var) for var in var_list])

def accuracy(logits, labels):
  correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
  return tf.reduce_mean(tf.cast(correct, tf.float32))
