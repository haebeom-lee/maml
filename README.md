# MAML (tensorflow)
This repository is (possibly the simplest) tensorflow implementation of Model Agnostic Meta Learning (MAML) by Finn et al.

See https://github.com/cbfinn/maml for the original repo.

The following datasets are considered.
1. MNIST: 5 / 5 classes are used for training / testing, respectively. This dataset will be automatically downloaded to your directory, so you can easily verify that the code is running (without download from somewhere else).
2. Omniglot, Mini-imagenet: You need to download and preprocess the dataset from other sources. See ```data.py``` for some information.

After datasets are ready, just run one of the bash script files (e.g. ```bash run_mimgnet_5way1shot.sh```).
