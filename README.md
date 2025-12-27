# Active Learning with Adaptive Mixture of Strategies

先不要整这么复杂，先面向过程！

实验也是从最简单开始，不要全部都对比了

先跑一个baselines不要着急

TODOs:
1. dataset preparation ✅
   1. mnist, fashionmnist, svhn, cifar10, iris...
   2. statistics (train, val, test; number, balance)
2. model adaptation ✅
   1. vgg, resnet (from scratch)
3. optim adaptation ✅
   1. sgd, adam
4. tracker
   1. configurations ✅
   2. checkpoints for each round ✅
   3. train, val, test / loss, acc for each round in and out ✅
   4. strategies
      1. items selected (index, number, class) ✅
      2. items features (train and val; index) ✅
      3. score distribution (index + score)
      4. changed weights
5. experiment setup
   1. seed list (5)
   2. logging
   3. configurations (optim, models; set with baselines)
   4. early-stopping?
   5. scripts
6. experiment schedule
   1. baseline (full sets) 找到合适的epoch和init
   2. ...
7. results aggregation and visualization
   1. scripts
   2. mean, std
   3. tsne on features

一共有什么策略：
cluster就有很多（附录）

## file structure


## report

写的时候规定符号（符号表）


## reference

https://github.com/ej0cl6/deep-active-learning