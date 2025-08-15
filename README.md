# dreamerv3-torch
Pytorch implementation of [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104v1). DreamerV3 is a scalable algorithm that outperforms previous approaches across various domains with fixed hyperparameters.

## Instructions

Roxxi: Get dependencies though environment.yml
Get dependencies with python 3.11:
```
conda env create -f environment.yml

```
Run training on DMC Vision:
```
python3 dreamer.py --configs dmc_vision --task dmc_walker_walk --logdir ./logdir/dmc_walker_walk
```
Monitor results:
```
tensorboard --logdir ./logdir
```
Run training on Ai2thor:
```
python3 dreamer.py --configs ai2thor --task ai2thor_nav --logdir ./logdir/dense_test_1234 --habitat_id 1234
```
