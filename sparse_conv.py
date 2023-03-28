import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch import Tensor


import math
import copy
import numpy as np
import matplotlib.pyplot as plt

from sparse_conv_wrapper import *

from enum import Enum

# class syntax
class Sparse_modes(Enum):
        Training                = 1
        Inference_Vanilla       = 2
        Inference_Sparse        = 3
        Test                    = 4

#--------------------------------------------------------
#-----------Sparse Conv Custom Layer---------------------
#--------------------------------------------------------

#THIS LAYER MAKE SENSE ONLY ON CUDA, WE HAVE NOT DONE THE PORTING OF THE C++ VERSION, IF NO CUDA AVAILABLE IT WILL USE THE CLASSIC nn.conv2d
class SparseConv2D(torch.nn.Conv2d):
        
        def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t  = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = None):
                self.use_sparse = False
                self.sparse_kernel = None
                self.mode = Sparse_modes.Training
                self.padded_input = None
                self.padded_batch_size = 0

                groups = 1
                super(SparseConv2D, self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,None)#Conv2D init function

                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.bias = bias

        def set_mode(self,mode=Sparse_modes.Training):
                '''
                This method change the configuration of inference of this layer
                1.Training                => Like nn.Conv2d
                2.Inference_Vanilla       => Like nn.Conv2d
                3.Inference_Sparse        => Sparse Convolution
                '''

                self.mode =  mode
                '''
                if mode == Sparse_modes.Training:
                        self.training   = True
                        self.use_sparse = False
                elif mode == Sparse_modes.Inference_Vanilla:
                        self.training   = False
                        self.use_sparse = False
                elif mode == Sparse_modes.Inference_Sparse:
                        self.training   = False
                        self.use_sparse = True
                '''
        def make_kernel_sparse(self,in_height,in_width):
                '''
                This method convert the kernel from (N,C,H,W) => (N,C,H*W)
                Then Convert the kernel into CSR format with ctypes compatible with the CUDA implementation
                '''
                #Copy kernel
                k = copy.deepcopy(self.weight.detach())
                #print(k)
                print(f"Kernel Shape:{k.shape}")

                #Reshape kernel from (CH , W , H) => (CH , W * H)
                k_size = k.shape[1] * k.shape[2] * k.shape[3]
                x = torch.reshape(k, (-1,k_size))
                print(f"New Kernel Shape:{x.shape}")

                #Convert to CSR
                self.sparse_kernel = x.to_sparse_csr()
                
                #Define the CSR format in tenrors to CUDA
                self.rowptr = self.sparse_kernel.crow_indices().to(torch.int).cuda()
                self.colidx = self.sparse_kernel.col_indices().to(torch.int).cuda()
                self.values = self.sparse_kernel.values().cuda()
                
                #Stretch the Kernel to input size: (SEE PAPER OF ESCOTING)
                kernel_h = self.weight.shape[2]
                kernel_w = self.weight.shape[3]
                gpu_kernel_stretch(self.rowptr,self.colidx,self.out_channels,in_height,in_width,self.padding,self.padding,kernel_h,kernel_w)
                
                #print(f"rowptr: {self.rowptr} => {self.rowptr.type()}")
                #print(f"colidx: {self.colidx} => {self.colidx.type()}")
                #print(f"values: {self.values} => {self.values.type()}")

                return
                #End Deprecated => Sequential Code
                for out_channel in range(self.out_channels):
                        print(f"ROW [{out_channel}]")
                        for j in range(self.rowptr[out_channel] , self.rowptr[out_channel+1]):
                                col = copy.deepcopy(self.colidx[j])
                                kernel_col = col%kernel_w
                                kernel_row = math.floor((col/kernel_w))%kernel_h
                                in_channel = math.floor(col/(kernel_w*kernel_h))
                                self.colidx[j] = math.floor((in_channel*(in_height + self.padding) + kernel_row)*(in_width + self.padding) + kernel_col)
                                print(f"Changing colidx[{j}] from {col} => {self.colidx[j]}")
                
        def test_behaviour(self, input:Tensor,print_flag=True) ->Tensor:
                '''
                Check if the Sparse Layer and Conv2D layer has same behaviour
                '''
                #Set mode to sparse
                self.set_mode(Sparse_modes.Inference_Sparse)
                #Compute nn.Conv2D output
                vanilla_out = super().forward(input)
                #Compute the SparseConv2D output
                sparse_out  = self.forward(input)

                #Comparing the Output
                print("Vanilla vs SparseConv:")
                comparison = sparse_out.eq(vanilla_out)
                comparison = comparison.to("cpu")
                if torch.all(comparison):
                        self.set_mode(Sparse_modes.Test)
                        if print_flag:
                                print("\033[92mSUCCESS => Same Outputs\033[0m")
                        #print(comparison)
                        return sparse_out
                else:
                        if print_flag:
                                print("\033[91mFAIL => Divergent Outputs\033[0m")
                        print(f"Vanilla:{vanilla_out}")
                        print(f"Sparse:{sparse_out}")
                        #plt.imshow(comparison.numpy())
                        #plt.colorbar()
                        #plt.show()
                        raise Exception("\033[91mFAILED TEST SPARSE BEHAVIOUR => Divergent Outputs\033[0m") 

                        return False

        def forward(self, input: Tensor) -> Tensor:  # input: HWCN
                #TODO CHECK if CUDA is available and in case not use nn.conv2D forward

                #TODO CHECK SPARSITY

                #TODO ADD "Group > 1" compatibility

                #Training mode

                if self.mode == Sparse_modes.Training or self.mode == Sparse_modes.Inference_Vanilla:
                        return super().forward(input) #IF WE ARE IN TRAINING USE THE CLASSIC CONV2D forward to build the weights
                elif self.mode == Sparse_modes.Test:
                        return self.test_behaviour(input,print_flag=True)

                #No training with sparse conv

                #USE CUDA SPARSE CONV
                #Input shape
                in_height = input.shape[2]
                in_width  = input.shape[3]

                #Kernel Shape
                kernel_h = self.weight.shape[2]
                kernel_w = self.weight.shape[3]

                #Output Shape (TODO convert 1D padding and stride to 2D)
                output_h = math.floor((in_height + 2 * self.padding - (self.dilation * (kernel_h - 1) + 1)) / self.stride + 1)
                output_w = math.floor((in_width  + 2 * self.padding - (self.dilation * (kernel_w - 1) + 1)) / self.stride + 1)
                
                #Convert kernel to CSR
                batch_size = input.shape[0] #TODO extract from input

                #Convert kernel to CSR format if not already done
                if self.sparse_kernel == None:
                        self.make_kernel_sparse(in_height,in_width)
                
                ifmap_size =  self.in_channels * (in_height + self.padding) * (in_width + self.padding)

                #Compute the padded input
                if (self.padded_input == None and self.padding != 0) or batch_size != self.padded_batch_size:
                        self.padded_batch_size = batch_size
                        padded_input_size = batch_size * (self.in_channels * (in_height + self.padding) * (in_width + self.padding) + self.padding * (in_width + 2 * self.padding))
                        self.padded_input = torch.zeros(padded_input_size).cuda()

                #Align the input to the padded version of the input (Add 0 to the borders)
                if self.padding != 0:
                        padding_input_alignment(self.padded_input,input,self.in_channels,in_height,in_width,self.padding,self.padding,batch_size)
                        input = self.padded_input

                #Allocate outputs       
                output  = torch.zeros(batch_size, self.out_channels,output_h, output_w).cuda()
                input   = input.cuda()
                #Calculate sparse conv
                sparse_conv(input,self.in_channels,ifmap_size,in_height,in_width,self.padding,self.padding,self.stride,self.stride,self.dilation,self.dilation,self.rowptr,self.colidx,self.values,kernel_h,kernel_w,self.bias,output,self.out_channels,self.groups,batch_size)

                #Return output
                return output



#-------------------------------------------------------------------------------------------------------
#-----------Helper Model Module with some custom method to initialize sparseConv layers-----------------
#-------------------------------------------------------------------------------------------------------

class SparseModel(nn.Module):
        '''
        This custom model simply help with the usage of Models that integrate our custom SparseConv module
        Contain helper method to initialize the sparseConv layers and change inference mode from vanilla to sparse (and opposite)
        '''
        def __init__(self, sparse_conv_flag=True):
                self._sparse_conv_flag=sparse_conv_flag
                super(SparseModel, self).__init__()
                if sparse_conv_flag:
                        self.conv = SparseConv2D
                else:
                        self.conv = nn.Conv2d

        def _set_sparse_layers_mode(self,mode=Sparse_modes.Training):
                '''
                Set the mode of all SparseConv2D layer in the net to "mode"
                '''
                for name, m in self.named_modules():
                        if isinstance(m, SparseConv2D):
                                m.set_mode(mode)

        def _initialize_sparse_layers(self,input_shape):
                #Change the Network mode to sparse
                self._set_sparse_layers_mode(mode = Sparse_modes.Inference_Sparse)
                #Generate a dummy input
                dummy_input = torch.randn(1, 1,input_shape[2],input_shape[3], dtype=torch.float).cuda()
                dummy_input = dummy_input.cuda()
                #Do a forword so that all sparseConv layer initialize the CSR kernel and stuff
                self.forward(dummy_input)

        def forward(self,input: Tensor) -> Tensor:
                #IMPLEMENT IT AS YOU LIKE
                return

               


                                



