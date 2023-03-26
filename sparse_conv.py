import torch
import torch.nn as nn
from torch import Tensor
import math
import os
import ctypes
from tqdm import tqdm
import copy
LIB_PATH = "./lib"
LIB_NAME = "sparse_conv.so"

path = os.path.join(LIB_PATH,LIB_NAME)

#--------------------------------------------------------
#-----------LOAD THE SHARED LIBRARY----------------------
#--------------------------------------------------------

#Step1: Load the actual binary
_lib = ctypes.cdll.LoadLibrary(path)
#Step2: Set up the wrapper parameters (define the function parameters types)
#DECLARE EXTERN C and then
_lib.test_wrapper.restype = None
_lib.test_wrapper.argtypes = []

_lib.gpu_sparse_conv.restype = None

_lib.gpu_sparse_conv.argtypes = [
                                ctypes.c_bool,#use_relu
                                ctypes.c_int,#num
                                ctypes.c_void_p,#input
                                ctypes.c_int,#if_map
                                ctypes.c_void_p,#rowprt
                                ctypes.c_void_p,#colidx
                                ctypes.c_void_p,#values
                                ctypes.c_void_p,#bias
                                ctypes.c_int,#height
                                ctypes.c_int,#width
                                ctypes.c_int,#pad_h
                                ctypes.c_int,#pad_w
                                ctypes.c_int,#stride_h
                                ctypes.c_int,#stride_w
                                ctypes.c_int,#dilation_h
                                ctypes.c_int,#dilation_w
                                ctypes.c_int,#kernel_h
                                ctypes.c_int,#kernel_w
                                ctypes.c_void_p,#output
                                ctypes.c_int,#out_channels
                                ctypes.c_int#numgrpups
                                ]

def sparse_conv(input,in_channels,ifmap_size,height,width,pad_h,pad_w,stride_h,stride_w,dilatation_h,dilatation_w,rowptr,colidx,values,kernel_h,kernel_w,bias,output,output_channels,num_groups):
        
        ifmap_size =  in_channels * (height + pad_h) * (width + pad_w)

        _lib.gpu_sparse_conv(
                             ctypes.c_bool(False),#use_relu
                             ctypes.c_int(1),#num
                             ctypes.c_void_p(input.data_ptr()),
                             ctypes.c_int(ifmap_size),#ifmap
                             ctypes.c_void_p(rowptr.data_ptr()),#rowptr
                             ctypes.c_void_p(colidx.data_ptr()),#colidx
                             ctypes.c_void_p(values.data_ptr()),#values
                             ctypes.c_void_p(None),#bias
                             ctypes.c_int(height),
                             ctypes.c_int(width),
                             ctypes.c_int(pad_h),
                             ctypes.c_int(pad_w),
                             ctypes.c_int(stride_h),
                             ctypes.c_int(stride_w),
                             ctypes.c_int(dilatation_h),
                             ctypes.c_int(dilatation_w),
                             ctypes.c_int(kernel_h),
                             ctypes.c_int(kernel_w),
                             ctypes.c_void_p(output.data_ptr()),#output
                             ctypes.c_int(output_channels),
                             ctypes.c_int(num_groups),
                             )
        return output

#--------------------------------------------------------
#-----------PREPARE THE DATA FOR TESTING-----------------
#--------------------------------------------------------
_lib.test_wrapper()
#--------------------------------------------------------
#-----------TEST THE FUNCTIONS---------------------------
#--------------------------------------------------------

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

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


        def make_kernel_sparse(self):
                k = copy.deepcopy(self.weight.detach())
                print(k)
                print(f"Kernel Shape:{k.shape}")
                k_size = k.shape[2] * k.shape[3]
                x = torch.reshape(k, (-1,k_size))
                print(f"New Kernel Shape:{x.shape}")
                self.sparse_kernel = x.to_sparse_csr()
                #print(self.sparse_kernel)
        def forward(self, input: Tensor) -> Tensor:  # input: HWCN
                if self.training:
                        return super().forward(input) #IF WE ARE IN TRAINING USE THE CLASSIC CONV2D forward to build the weights

                if not self.use_sparse:
                        return super().forward(input)

                if self.sparse_kernel == None:
                        self.make_kernel_sparse()

                rowptr = self.sparse_kernel.crow_indices().cuda()
                colidx = self.sparse_kernel.col_indices().cuda()
                values = self.sparse_kernel.values().cuda()
                
                print(f"rowptr: {rowptr}")
                print(f"colidx: {colidx}")
                print(f"values: {values}")

                #USE CUDA SPARSE CONV
                #Input shape
                print(f"IN SHAPE: ({input.shape})")
                in_height = input.shape[2]
                in_width  = input.shape[3]

                kernel_h = self.weight.shape[2]
                kernel_w = self.weight.shape[3]

                #Convert kernel to CSR
                batch_size = 1 #TODO extract from input
                print(self.dilation)
                print(self.padding)
                print(self.stride)

               
                #output_h = (in_height + 2 * self.padding[0] - (self.dilation[0] * (kernel_h - 1) + 1)) / self.stride[0] + 1
                #output_w = (in_width  + 2 * self.padding[1] - (self.dilation[1] * (kernel_w - 1) + 1)) / self.stride[1] + 1
                
                output_h = math.floor((in_height + 2 * self.padding - (self.dilation * (kernel_h - 1) + 1)) / self.stride + 1)
                output_w = math.floor((in_width  + 2 * self.padding - (self.dilation * (kernel_w - 1) + 1)) / self.stride + 1)
                
                #Reshape input todo
                output = torch.zeros(batch_size, self.out_channels,output_h, output_w).cuda()
                input = input.cuda()
                sparse_conv(input,self.in_channels,1,in_height,in_width,self.padding,self.padding,self.stride,self.stride,self.dilation,self.dilation,rowptr,colidx,values,kernel_h,kernel_w,self.bias,output,self.out_channels,self.groups)
                return output


