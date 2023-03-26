# Repo Content:

This repository is a porting of Ecosin Caffe Branch to Pytorch.\
If intrested there is also a Tensorflow Porting of the same code: TODO

# Ecosin Framework:

Ecosin in a Caffe branch with SparseConvolution capability.
It aim at accelerating on GPU convoluntion in context where kernels has hight sparsity factor.

It Working principle are based on CSR kernel compression.

The paper that describe the SparseConvolution implementation contained in Ecosin is available at: https://arxiv.org/pdf/1802.10280.pdf\
The original C++ coda is available in the following repository:\
https://github.com/chenxuhao/caffe-escoin\
More specifically in this file: https://github.com/chenxuhao/caffe-escoin/blob/master/src/caffe/util/math_functions.cu => function caffe_gpu_sconv(...)

# How our Pytorch version work:
TODO