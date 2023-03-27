# Repo Content: (Work in progress)

This repository is a porting of Ecosin Caffe Branch to Pytorch.\
If intrested there is also a Tensorflow Porting of the same code: TODO

# Ecosin Framework:

Ecosin in a Caffe branch with SparseConvolution capability.
It aim at accelerating on GPU convoluntion in context where kernels has hight sparsity factor.

It Working principle are based on CSR kernel compression.

The paper that describe the SparseConvolution implementation contained in Ecosin is available at: https://arxiv.org/pdf/1802.10280.pdf

The original C++ coda is available in the following repository:\
https://github.com/chenxuhao/caffe-escoin\

More specifically in this file: https://github.com/chenxuhao/caffe-escoin/blob/master/src/caffe/util/math_functions.cu => function caffe_gpu_sconv(...)

# How To use:
TODO => make better instructions\
To use our custom pytorch layer simply compile it with the Makefile then:\
```
import sparse_conv as sp


self.conv1 = sp.SparseConv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1)
       
```


# How Run the example:
TODO better

- Compile the CUDA library by executing the Makefile => (simply do make on terminal)
```
make all
```
- Execute the example script:
```
python test_behaviour.py
```
- If you see as output the following all works fine.
```
Vanilla vs SparseConv:
SUCCESS => Same Outputs
IN -shape: torch.Size([1, 1, 32, 32])
OUT-shape: torch.Size([1, 6, 28, 28])
```
# How It works:

We have simply written a CUDA => python wrapper using the ctype package of python.\
Then we relized a custom pytorch module (**SparseConv2D**) tthat use Ecosin CUDA kernels to compute sparse conv.