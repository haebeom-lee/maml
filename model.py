from layers import *

class MAML:
  def __init__(self, args):
    if args.dataset in ['mnist', 'omniglot']:
      self.xdim, self.channel = 28, 1
    elif args.dataset in ['miniimagenet']:
      self.xdim, self.channel = 32, 3

    self.hdim = 64 # num of hidden channels
    self.numclass = args.way # num of classes per each episode
    self.alpha = args.alpha # inner gradient stepsize
    self.n_steps = args.n_steps # num of inner gradient steps
    self.hessian = args.hessian # if False, then ignore hessian
    self.metabatch = args.metabatch # metabatch size

    xshape = [self.metabatch, None, self.xdim*self.xdim*self.channel]
    yshape = [self.metabatch, None, self.numclass]
    self.episodes = {
        'xs': tf.placeholder(tf.float32, xshape, name='xs'),
        'ys': tf.placeholder(tf.float32, yshape, name='ys'),
        'xq': tf.placeholder(tf.float32, xshape, name='xq'),
        'yq': tf.placeholder(tf.float32, yshape, name='yq')}

  def network(self, x, weights, name='network', reuse=None):
    x = tf.reshape(x, [-1, self.xdim, self.xdim, self.channel])
    for l in [1,2,3,4]:
      x = conv_block(x, (weights['conv%d'%l], weights['b%d'%l]),
          activation=relu, scope=name+'/conv_block%d'%l, reuse=reuse)
    x = flatten(x)
    x = dense_block(x, (weights['dense'], weights['b']),
        scope=name+'/dense_block', reuse=reuse)
    return x

  def get_weights(self, reuse=None):
    with tf.variable_scope('conv_weights', reuse=reuse):
      # from conv1 to conv4 layer
      weights = {}
      for l in [1,2,3,4]:
        indim = self.channel if l == 1 else self.hdim
        weights['conv%d'%l] = tf.get_variable('conv%d'%l,
            [3,3,indim,self.hdim])
        weights['b%d'%l] = tf.get_variable('b%d'%l, [self.hdim])

    # dense weights = zeros
    weights['dense'] = tf.zeros([self.hdim, self.numclass])
    weights['b'] = tf.zeros([self.numclass])
    return weights

  # loss for a single episode
  def get_loss_single(self, inputs, reuse=None, name='single'):
    xtr, ytr, xte, yte = inputs

    # initial weights
    weights = self.get_weights(reuse=reuse)

    for i in range(self.n_steps):
      # evaluate inner gradient step
      inner_output = self.network(xtr, weights, name=name+'/network',
          reuse=reuse if i == 0 else True)

      inner_loss = cross_entropy(inner_output, ytr)
      # compute gradients
      grads = tf.gradients(inner_loss, list(weights.values()))
      # ignore hessian by stopping gradients
      if not self.hessian:
        grads = [tf.stop_gradient(grad) for grad in grads]
      gradients = dict(zip(weights.keys(), grads))

      # get task-specific weights
      weights = dict(zip(weights.keys(),
        [weights[key] - self.alpha * gradients[key] \
            for key in weights.keys()]))

    # evaluate outer loss
    outer_output = self.network(xte, weights, name=name+'/network',
        reuse=True)

    cent = cross_entropy(outer_output, yte)
    acc = accuracy(outer_output, yte)
    return cent, acc

  # loss for multiple episodes (metabatch)
  def get_loss_multiple(self, name='multiple', reuse=None):
    xs, ys = self.episodes['xs'], self.episodes['ys']
    xq, yq = self.episodes['xq'], self.episodes['yq']

    # map_fn: enables parallelization
    cent, acc = tf.map_fn(
        self.get_loss_single,
        elems=(xs, ys, xq, yq),
        dtype=(tf.float32, tf.float32),
        parallel_iterations=self.metabatch)

    net = {}
    net['cent'] = tf.reduce_mean(cent)
    net['acc'] = tf.reduce_mean(acc)
    conv_weights = [w for w in tf.trainable_variables() \
        if 'conv' in w.name]
    net['wd'] = weight_decay(1e-2, conv_weights)
    return net
