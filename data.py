import numpy as np

class Data:
  def __init__(self, args):
    if args.dataset == 'mnist':
      from tensorflow.examples.tutorials.mnist import input_data
      mnist = input_data.read_data_sets('./data/mnist',
          one_hot=True, validation_size=0)
      x, y = mnist.train.images, mnist.train.labels
      y_ = np.argmax(y, axis=1)

      self.N = 1000 # total num instances per class
      self.Ktr = 5 # total num train classes
      self.Kte = 5 # total num test classes

      self.xtr = [x[y_==k][:self.N,:] for k in range(self.Ktr)]
      self.xte = [x[y_==k][:self.N,:] for k in range(self.Ktr, self.Ktr + self.Kte)]

    elif args.dataset == 'omniglot':
      self.N = 20 # total num instances per class
      self.Ktr = 4800 # total num train classes
      self.Kte = 1692 # total num test classes

      # TODO download the dataset and preprocess.
      # 4800, 1692 : total number of train / test classes
      # 20 : total num of instances per class
      # 28*28*1 : Height * Width * Channel

      #xtr = np.load('./data/omniglot/omni_train_rot.npy')
      #xte = np.load('./data/omniglot/omni_test_rot.npy')
      #self.xtr = np.reshape(xtr, [4800,20,28*28*1])
      #self.xte = np.reshape(xte, [1692,20,28*28*1])
      pass

    elif args.dataset == 'mimgnet':
      self.N = 600 # total num instances per class
      self.Ktr = 64 # total num train classes
      self.Kte = 20 # total num test classes

      # TODO download the dataset and preprocess.
      # 64, 20 : total number of train / test classes
      # 600 : total num of instances per class
      # 84*84*3 : Height * Width * Channel

      #xtr = np.load('./data/mimgnet/train.npy')
      #xte = np.load('./data/mimgnet/test.npy')
      #self.xtr = np.reshape(xtr, [64,600,84*84*3])
      #self.xte = np.reshape(xte, [20,600,84*84*3])
      pass

    else:
      raise ValueError('No such dataset %s' % args.dataset)

  def generate_episode(self, args, training=True, n_episodes=1):
    generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
    n_way, n_shot, n_query = args.way, args.shot, args.query
    K = self.Ktr if training else self.Kte
    x = self.xtr if training else self.xte

    xs, ys, xq, yq = [], [], [], []
    for t in range(n_episodes):
      # sample WAY classes
      classes = np.random.choice(range(K), size=n_way, replace=False)

      support_set = []
      query_set = []
      for k in list(classes):
        # sample SHOT and QUERY instances
        idx = np.random.choice(range(self.N), size=n_shot+n_query, replace=False)
        x_k = x[k][idx]
        support_set.append(x_k[:n_shot])
        query_set.append(x_k[n_shot:])

      xs_k = np.concatenate(support_set, 0)
      xq_k = np.concatenate(query_set, 0)
      ys_k = generate_label(n_way, n_shot)
      yq_k = generate_label(n_way, n_query)

      xs.append(xs_k)
      xq.append(xq_k)
      ys.append(ys_k)
      yq.append(yq_k)

    xs, ys = np.stack(xs, 0), np.stack(ys, 0)
    xq, yq = np.stack(xq, 0), np.stack(yq, 0)
    return [xs, ys, xq, yq]
