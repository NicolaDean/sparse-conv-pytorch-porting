import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch import Tensor


import math
import copy


from sparse_conv_wrapper import *

#--------------------------------------------------------
#-----------TEST THE FUNCTIONS---------------------------
#--------------------------------------------------------



class SparseConv2D(torch.nn.Conv2d):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t  = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = None):
                self.use_sparse = False
                self.sparse_kernel = None
                groups = 1
                super(SparseConv2D, self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,None)#Conv2D init function

                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.stride = stride
                self.padding = padding
                self.dilation = dilation
                self.bias = bias


        def make_kernel_sparse(self,in_height,in_width):
                #Copy kernel
                k = copy.deepcopy(self.weight.detach())
                print(k)
                print(f"Kernel Shape:{k.shape}")

                #Reshape kernel from (CH , W , H) => (CH , W * H)
                k_size = k.shape[2] * k.shape[3]
                x = torch.reshape(k, (-1,k_size))
                print(f"New Kernel Shape:{x.shape}")

                #Convert to CSR
                self.sparse_kernel = x.to_sparse_csr()
                
                #Define the CSR format in tenrors to CUDA
                self.rowptr = self.sparse_kernel.crow_indices().cuda()
                self.colidx = self.sparse_kernel.col_indices().cuda()
                self.values = self.sparse_kernel.values().cuda()
                
                print(f"colidx: {self.colidx}")
                #Stretch the Kernel to input size: (SEE PAPER OF ESCOTING)
                kernel_h = self.weight.shape[2]
                kernel_w = self.weight.shape[3]
                gpu_kernel_stretch(self.rowptr,self.colidx,self.out_channels,in_height,in_width,self.padding,self.padding,kernel_h,kernel_w)

                print(f"rowptr: {self.rowptr}")
                print(f"colidx: {self.colidx}")
                print(f"values: {self.values}")

                return
                for out_channel in range(self.out_channels):
                        print(f"ROW [{out_channel}]")
                        for j in range(self.rowptr[out_channel] , self.rowptr[out_channel+1]):
                                col = copy.deepcopy(self.colidx[j])
                                kernel_col = col%kernel_w
                                kernel_row = math.floor((col/kernel_w))%kernel_h
                                in_channel = math.floor(col/(kernel_w*kernel_h))
                                self.colidx[j] = math.floor((in_channel*(in_height + self.padding) + kernel_row)*(in_width + self.padding) + kernel_col)
                                print(f"Changing colidx[{j}] from {col} => {self.colidx[j]}")
                
                #End
                

        def forward(self, input: Tensor) -> Tensor:  # input: HWCN

                #Training mode
                if self.training:
                        return super().forward(input) #IF WE ARE IN TRAINING USE THE CLASSIC CONV2D forward to build the weights
                #No training but no sparse conv
                if not self.use_sparse:
                        return super().forward(input)

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
                batch_size = 1 #TODO extract from input

                #Convert kernel to CSR format if not already done
                if self.sparse_kernel == None:
                        self.make_kernel_sparse(in_height,in_width)
                
                #Allocate outputs
                output = torch.zeros(batch_size, self.out_channels,output_h, output_w).cuda()

                #Calculate sparse conv
                sparse_conv(input,self.in_channels,1,in_height,in_width,self.padding,self.padding,self.stride,self.stride,self.dilation,self.dilation,self.rowptr,self.colidx,self.values,kernel_h,kernel_w,self.bias,output,self.out_channels,self.groups)

                #Return output
                return output


