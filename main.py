from __future__ import print_function
import tensorflow as tf
import argparse
import numpy as np
import time
import os

from model import MAML
from data import Data
from accumulator import Accumulator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--save_freq', type=int, default=1000)

parser.add_argument('--metabatch', type=int, default=10)
parser.add_argument('--n_train_iters', type=int, default=300)
parser.add_argument('--n_test_iters', type=int, default=10)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--way', type=int, default=5)
parser.add_argument('--shot', type=int, default=1)
parser.add_argument('--query', type=int, default=20)

parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--n_steps', type=int, default=1)
parser.add_argument('--hessian', action='store_true', default=False)

args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)

savedir = './results/run' \
    if args.savedir is None else args.savedir
if not os.path.isdir(savedir):
  os.makedirs(savedir)

# data loader
data = Data(args)

n_train_batches=1

def train():
  model = MAML(args)
  net = model.get_loss_multiple()
  loss = net['cent'] + net['wd']

  global_step = tf.train.get_or_create_global_step()
  lr_step = int(n_train_batches*args.n_train_iters/2)
  lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
          [lr_step], [1e-2, 1e-3])
  train_op = tf.train.AdamOptimizer(lr).minimize(loss,
          global_step=global_step)

  saver = tf.train.Saver(tf.trainable_variables())
  logfile = open(os.path.join(savedir, 'train.log'), 'w', 0)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  # train
  train_logger = Accumulator('cent', 'wd', 'acc')
  train_to_run = [train_op, net['cent'], net['wd'], net['acc']]

  for i in range(args.n_train_iters+1):
    # feed_dict
    epi = model.episodes
    placeholders = [epi['xs'], epi['ys'], epi['xq'], epi['yq']]
    episode = data.generate_episode(args, True, n_episodes=args.metabatch)
    fdtr = dict(zip(placeholders, episode))

    train_logger.accum(sess.run(train_to_run, feed_dict=fdtr))

    if i % 10 == 0:
      line = 'Iter %d start, learning rate %f' % (i, sess.run(lr))
      print('\n' + line)
      logfile.write('\n' + line + '\n')
      train_logger.print_(header='train', episode=i*args.metabatch,
          logfile=logfile)
      train_logger.clear()

      # validation (with test classes... be cautious!)
      test_logger = Accumulator('cent', 'wd', 'acc')
      test_to_run = [net['cent'], net['wd'], net['acc']]
      for j in range(args.n_test_iters):
        # feed_dict
        epi = model.episodes
        placeholders = [epi['xs'], epi['ys'], epi['xq'], epi['yq']]
        episode = data.generate_episode(args, False,
            n_episodes=args.metabatch)
        fdte= dict(zip(placeholders, episode))
        test_logger.accum(sess.run(test_to_run, feed_dict=fdte))

      test_logger.print_(header='test ', episode=i*args.metabatch,
          logfile=logfile)
      test_logger.clear()

  logfile.close()
  saver.save(sess, os.path.join(savedir, 'model'))

if __name__=='__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode %s' % args.mode)
