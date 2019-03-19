from layers import *

class MAML:
  def __init__(self, args):
    self.dataset = args.dataset
    if self.dataset in ['mnist', 'omniglot']:
      self.xdim, self.input_channel = 28, 1
      self.n_channel = 64 # channel dim of conv layers
    elif self.dataset in ['mimgnet']:
      self.xdim, self.input_channel = 84, 3
      self.n_channel = 32

    self.numclass = args.way # num of classes per each episode
    self.alpha = args.alpha # inner gradient stepsize
    self.n_steps = args.n_steps # num of inner gradient steps
    self.metabatch = args.metabatch # metabatch size

    xshape = [self.metabatch, None, self.xdim*self.xdim*self.input_channel]
    yshape = [self.metabatch, None, self.numclass]
    # 's': support, 'q': query
    self.episodes = {
        'xs': tf.placeholder(tf.float32, xshape, name='xs'),
        'ys': tf.placeholder(tf.float32, yshape, name='ys'),
        'xq': tf.placeholder(tf.float32, xshape, name='xq'),
        'yq': tf.placeholder(tf.float32, yshape, name='yq')}

  def get_weights(self, reuse=None):
    conv_init = tf.truncated_normal_initializer(stddev=0.02)
    fc_init = tf.random_normal_initializer(stddev=0.02)
    bias_init = tf.zeros_initializer()
    # In the original repo, following initializers are used:
    # conv_init = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    # fc_init = tf.contrib.layers.xavier_initializer(dtype=tf.float32)
    # bias_init = tf.zeros_initializer()

    with tf.variable_scope('theta', reuse=reuse):
      weights = {}
      for l in [1,2,3,4]:
        indim = self.input_channel if l == 1 else self.n_channel
        weights['conv%d_w'%l] = tf.get_variable('conv%d_w'%l,
            [3, 3, indim, self.n_channel], initializer=conv_init)
        weights['conv%d_b'%l] = tf.get_variable('conb%d_b'%l,
            [self.n_channel], initializer=bias_init)
      factor = 5*5 if self.dataset == 'mimgnet' else 1
      weights['dense_w'] = tf.get_variable('dense_w',
          [factor*self.n_channel, self.numclass], initializer=fc_init)
      weights['dense_b'] = tf.get_variable('dense_b',
          [self.numclass], initializer=bias_init)
      return weights

  def forward(self, x, weights):
    # For mini-imagenet, this cnn is exactly the same as in the original repo.
    # For omniglot, this cnn will be a little bit bigger.
    # I choose this because the implementation is much simpler.
    # See [https://github.com/cbfinn/maml] for the difference.
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.input_channel])
    for l in [1,2,3,4]:
      w, b = weights['conv%d_w'%l], weights['conv%d_b'%l]
      x = conv_block(x, w, b, bn_scope='conv%d_bn'%l)
    return tf.matmul(flatten(x), weights['dense_w']) + weights['dense_b']

  def get_loss_single(self, inputs, weights):
    xs, ys, xq, yq = inputs

    for i in range(self.n_steps):
      inner_logits = self.forward(xs, weights)
      inner_loss = cross_entropy(inner_logits, ys)

      grads = tf.gradients(inner_loss, weights.values()) # compute gradients
      grads = [tf.stop_gradient(grad) for grad in grads] # no hessian by default
      gradients = dict(zip(weights.keys(), grads))

      weights = dict(zip(weights.keys(),
        [weights[key] - self.alpha * gradients[key] for key in weights.keys()]))

    outer_logits = self.forward(xq, weights)
    cent = cross_entropy(outer_logits, yq)
    acc = accuracy(outer_logits, yq)
    return cent, acc

  def get_loss_multiple(self, reuse=None):
    xs, ys = self.episodes['xs'], self.episodes['ys']
    xq, yq = self.episodes['xq'], self.episodes['yq']
    weights = self.get_weights(reuse=reuse)

    # map_fn: enables parallization
    cent, acc = tf.map_fn(
        lambda inputs: self.get_loss_single(inputs, weights),
        elems=(xs, ys, xq, yq),
        dtype=(tf.float32, tf.float32),
        parallel_iterations=self.metabatch)

    net = {}
    net['cent'] = tf.reduce_mean(cent)
    net['acc'] = tf.reduce_mean(acc)
    net['weights'] = tf.trainable_variables()
    return net
