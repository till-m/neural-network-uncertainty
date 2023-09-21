# neural-network-uncertainty

Contains methods mentioned in my internal SDSC talk about uncertainty quantification for neural network regression.

Methods:
- Direct Estimation of Aleatoric Variance (`GaussNLLNet`) [[1]](#1)
- Simultaneous Quantile Regression + Orthogonal Certificates (`SQRNet`, `OCNet`) [[2]](#2)
- MCDropout (`MCDropoutNet`) [[3]](#3)
- Deep Ensembles (`Ensemble`) [[4]](#4)
- Spectral-Normalized Gaussian Process (`SNGP`) [[5]](#5), [[6]](#6)
- Stochastic Weight Averaged Gaussian (`SWAG`) [[7]](#7), [[8]](#8)

## References
<a id="1">[1]</a> Nix, D.A., and A.S. Weigend. “Estimating the Mean and Variance of the Target Probability Distribution.” In Proceedings of 1994 IEEE International Conference on Neural Networks (ICNN’94), 1:55–60 vol.1, 1994. https://doi.org/10.1109/ICNN.1994.374138.

<a id="2">[2]</a> Tagasovska, Natasa, and David Lopez-Paz. “Single-Model Uncertainties for Deep Learning.” Advances in Neural Information Processing Systems 32 (2019).

<a id="3">[3]</a> Gal, Yarin, and Zoubin Ghahramani. “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.” arXiv, October 4, 2016. https://doi.org/10.48550/arXiv.1506.02142.

<a id="4">[4]</a> Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. “Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles.” arXiv, November 3, 2017. https://doi.org/10.48550/arXiv.1612.01474.

<a id="5">[5]</a> Liu, Jeremiah Zhe, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax-Weiss, and Balaji Lakshminarayanan. “Simple and Principled Uncertainty Estimation with Deterministic Deep Learning via Distance Awareness.” arXiv, October 25, 2020. https://doi.org/10.48550/arXiv.2006.10108.

<a id="6">[6]</a> Adapted from https://github.com/kimjeyoung/SNGP-BERT-Pytorch

<a id="7">[7]</a> Maddox, Wesley, Timur Garipov, Pavel Izmailov, Dmitry Vetrov, and Andrew Gordon Wilson. “A Simple Baseline for Bayesian Uncertainty in Deep Learning.” arXiv, December 31, 2019. http://arxiv.org/abs/1902.02476.

<a id="8">[8]</a> Adapted from https://github.com/wjmaddox/swa_gaussian
