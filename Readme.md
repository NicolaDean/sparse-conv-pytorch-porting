# Repo Content: (Work in progress)

This repository is a porting of Ecosin Caffe Branch to Pytorch.\
If intrested there is also a Tensorflow Porting of the same code: TODO

# Ecosin Framework:
(TODO write better)

Ecosin in a Caffe branch with **SparseConvolution** features.\
It aim GPU accelerating convoluntion in context where kernels has hight sparsity factor.

It Working principle are based on CSR kernel compression.

The paper that describe the SparseConvolution implementation contained in Ecosin is available at: https://arxiv.org/pdf/1802.10280.pdf

The original C++/CUDA code is available at:\
https://github.com/chenxuhao/caffe-escoin

More specifically, search for function **caffe_gpu_sconv(...)** inside this file:\
https://github.com/chenxuhao/caffe-escoin/blob/master/src/caffe/util/math_functions.cu

# How To use:
TODO => make better instructions\
To use our custom pytorch layer simply compile it with the Makefile then:\
```
import sparse_conv as sp


self.conv1 = sp.SparseConv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1)
       
```
You can also use our sp.SparseModel helper class that facilitate the creation and usage of the SparseConv2D layer by automatizing the initialization of CSR and stuff.

**(Check [vgg16_benchmark.py](vgg16_benchmark.py) or [resnet_benchmark.py](resnet_benchmark.py) to have better understanding of usage)**


```
class VGG16(sp.SparseModel):
    """
    A standard VGG16 model
    """

    def __init__(self, n_classes,sparse_conv_flag=True):
        self._sparse_conv_flag=sparse_conv_flag
        super(VGG16, self).__init__(sparse_conv_flag)

        #self.conv will be a 
        #-SparseConv2D if sparse_conv_flag is True 
        #-Conv2d if sparse_conv_flag is False

        self.layer1 = nn.Sequential(
            self.conv(1, 64, kernel_size=3, stride=1, padding=1),
            ...
        )

        [...Model Definition...]

```

# How Run the example:
TODO better

### Basic Single Layer Behavioural test:
- Compile the CUDA library by executing the Makefile => (simply do make on terminal)
```
make all
```
- Execute the example script:
```
python test_behaviour.py
```
- If you see as output the following all works fine.
- (This function is to fix a little since sometimes also a single digit of difference in output will make this function trigger exception)
```
Vanilla vs SparseConv:
SUCCESS => Same Outputs
IN -shape: torch.Size([1, 1, 32, 32])
OUT-shape: torch.Size([1, 6, 28, 28])
```

### Basic Full Network Behavioural test
Same working principle of the previous example but with a full net (Vgg16)
```
python test_behaviour_full_net.py
```

### VGG16 Benchmark script
Check [vgg16_benchmark.py](vgg16_benchmark.py)

```
python vgg16_benchmark.py
```
### Resnet Benchmark script
Check [resnet_benchmark.py](resnet_benchmark.py)

```
python resnet_benchmark.py
```
# How It works:

We have simply written a CUDA => python wrapper using the ctype package of python.\
Then we relized a custom pytorch module (**SparseConv2D**) tthat use Ecosin CUDA kernels to compute sparse conv.