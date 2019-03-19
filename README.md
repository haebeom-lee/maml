# MAML (tensorflow)
This repository is (possibly the simplest) tensorflow implementation of Model Agnostic Meta Learning (MAML) by Finn et al.

See https://github.com/cbfinn/maml for the original repo.

The following datasets are considered.
1. __MNIST__: 5 / 5 classes are used for training / testing, respectively. This dataset will be automatically downloaded to your directory, so you can easily verify that the code is running (without downloading from somewhere else).
2. __Omniglot__, __Mini-imagenet__: You need to download and preprocess the dataset from other sources. See ```data.py``` for some information.

After datasets are ready, just run one of the bash script files (e.g. ```bash run_mimgnet_5way1shot.sh```).

## Results

|       | mimgnet-5way1shot| mimgnet-5way5shot | omniglot-20way1shot| omniglot-20way5shot |
| ------| ---------------- | ----------------- | ------------------ | ------------------- |
| Paper | 48.07            | 63.15             | 95.8               | 98.9                |
| Ours  | 48.67            | xx.xx             | xx.x               | xx.x                |
