import numpy as np

class Data:
  def __init__(self, args):
    if args.dataset == 'mnist':
      from tensorflow.examples.tutorials.mnist import input_data
      mnist = input_data.read_data_sets('./data/mnist', one_hot=True,
          validation_size=0)
      x, y = mnist.train.images, mnist.train.labels
      y_ = np.argmax(y, axis=1)

      self.N = 1000 # total num instances per class
      self.Ktr = 5 # total num train classes
      self.Kte = 5 # total num test classes

      # meta-train data
      self.xtr = [x[y_==k][:self.N,:] for k in range(self.Ktr)]
      # meta-test data
      self.xte = [x[y_==k][:self.N,:] for k in \
          range(self.Ktr, self.Ktr + self.Kte)]

    elif args.dataset == 'omniglot':
      # TODO
      pass

    elif args.dataset == 'miniimagenet':
      # TODO
      pass

    elif args.dataset == 'sinusoidal':
      # TODO
      pass

    else:
      raise ValueError('No such dataset %s' % args.dataset)

  def generate_episode(self, args, training, n_episodes=1):
    K = self.Ktr if training else self.Kte
    x = self.xtr if training else self.xte

    xs, ys, xq, yq = [], [], [], []
    for t in range(n_episodes):
      # sample WAY classes
      classes = np.random.choice(range(K), size=args.way, replace=False)

      support = []
      query = []
      for k in list(classes):
        # sample SHOT and QUERY instances
        idx = np.random.choice(range(self.N), size=self.N, replace=False)
        x_k = x[k][idx]
        support.append(x_k[:args.shot])
        query.append(x_k[args.shot:args.shot+args.query])

      xs.append(np.concatenate(support, 0))
      xq.append(np.concatenate(query, 0))

      def generate_label(way, numsamp):
        return np.repeat(np.eye(way), numsamp, axis=0)

      ys.append(generate_label(args.way, args.shot))
      yq.append(generate_label(args.way, args.query))

    xs, ys = np.stack(xs, 0), np.stack(ys, 0)
    xq, yq = np.stack(xq, 0), np.stack(yq, 0)
    return [xs, ys, xq, yq]
