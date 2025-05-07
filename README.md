# 🔍 Research Series: Reassessing Classic GNNs

This work is part of our research series that re-evaluates the potential of classic GNNs:
  
| Paper | Venue | Repository |
| - | - | - | 
| [Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification](https://openreview.net/forum?id=xkljKdGe4E) | NeurIPS 2024 | [![GitHub](https://img.shields.io/badge/GitHub-Code-success?logo=github)](https://github.com/LUOyk1999/tunedGNN) |
| [Can Classic GNNs Be Strong Baselines for Graph-Level Tasks?](https://arxiv.org/abs/2502.09263) | ICML 2025    | [![GitHub](https://img.shields.io/badge/GitHub-Code-success?logo=github)](https://github.com/LUOyk1999/GNNPlus) |
| [When Dropout Meets Graph Convolutional Networks](https://openreview.net/forum?id=PwxYoMvmvy) | ICLR 2025 | [![GitHub](https://img.shields.io/badge/GitHub-Code-success?logo=github)](https://github.com/LUOyk1999/dropout-theory) |
| **[Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning](https://openreview.net/forum?id=t9lS1lX9FQ)** | ICLR 2025 | [![GitHub](https://img.shields.io/badge/GitHub-Code-success?logo=github)](https://github.com/LUOyk1999/NodeID) |

# Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning (ICLR 2025)

[![OpenReview](https://img.shields.io/badge/OpenReview-t9lS1lX9FQ-b31b1b.svg)](https://openreview.net/forum?id=t9lS1lX9FQ) [![arXiv](https://img.shields.io/badge/arXiv-2405.16435-b31b1b.svg)](https://arxiv.org/abs/2405.16435)

## Python environment setup with Conda

Tested with Python 3.7, PyTorch 1.12.1, and PyTorch Geometric 2.3.1, dgl 1.0.2.
```bash
pip install pandas
pip install scikit_learn
pip install numpy
pip install scipy
pip install einops
pip install ogb
pip install pyyaml
pip install googledrivedownloader
pip install networkx
pip install vqtorch
pip install gdown
pip install tensorboardX
pip install matplotlib
pip install seaborn
pip install rdkit
pip install tensorboard
```

## Overview

* `./SL` Experiment code of supervised Node ID.

* `./SSL` Experiment code of self-supervised Node ID.

## Reference

If you find our codes useful, please consider citing our work

```
@inproceedings{
luo2025node,
title={Node Identifiers: Compact, Discrete Representations for Efficient Graph Learning},
author={Yuankai Luo and Hongkang Li and Qijiong Liu and Lei Shi and Xiao-Ming Wu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=t9lS1lX9FQ}
}
```

