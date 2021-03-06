# MAML (tensorflow)
This repository is (possibly the simplest) tensorflow implementation of Model Agnostic Meta Learning (MAML) by Finn et al. (https://arxiv.org/abs/1703.03400)

See https://github.com/cbfinn/maml for the original repo.

The following datasets are considered.
1. __MNIST__: 5 / 5 classes are used for training / testing, respectively. This dataset will be automatically downloaded to your directory, so you can easily verify that the code is running (without downloading from somewhere else).
2. __Omniglot__, __Mini-imagenet__: You need to download and preprocess the dataset from other sources. See ```data.py``` for some information.

After datasets are ready, just run one of the bash script files (e.g. ```bash run_mimgnet_5way1shot.sh```).

## Results
__Mini-ImageNet__: I report those accuracies after running total 60000 iterations (no early-stopping with validation set). But loss does not converge further after 30000 iterations, so you can stop training around then.

__Omniglot__: Total 40000 iterations, and no early stopping with validation set. For simpler implementation, I used slightly bigger cnn architecture than the one used in the original repo, where they set stride to 2 to reduce the height and width of the feature maps, instead of max pooling. 

|       | mimgnet-5way1shot| mimgnet-5way5shot | omniglot-20way1shot| omniglot-5way1shot |
| ------| ---------------- | ----------------- | ------------------ | ------------------- |
| Paper (first order approx.) | 48.07          | 63.15             | 95.8               | 98.7                |
| Ours (first order approx.)  | __48.67__      | __64.71__         | __94.5__           | __98.0__            |

## Caution
1. Different initializers for the weights are used (see ```model.py```). I couldn't successfully reproduce the results with the one used in the original repo.
2. Meta- learning rate is set to ```1e-4```. I found that ```1e-3``` is too large, especially for Mini-Imagenet dataset. Multiple users have reported the difficulty of training MAML, so I believe that the correct learning rate should be lower than that.
3. According to the original repo, batch normalization statistics (e.g. moving mean and variance) are neither keeped nor used. So the statistics of current batch will be used for both training and testing. Also, centering (beta) variable is learned, but scaling (gamma) variable is ignored, as it will be done at the upper layer. This is the default setting of tf.contrib.layers.batch_norm. This is actually one of the reasons MAML is hard to train, and see MAML++ (https://openreview.net/forum?id=HJGven05Y7) for fixing this problem.
