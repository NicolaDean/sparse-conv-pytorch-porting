from torch import Tensor
import math
import os
import ctypes
from tqdm import tqdm

#DECLARE PATH TO THE .so shared library (Compile the CUDA code with make command)
LIB_PATH = "./lib"
LIB_NAME = "sparse_conv.so"

path = os.path.join(LIB_PATH,LIB_NAME)

#--------------------------------------------------------
#-----------LOAD THE SHARED LIBRARY----------------------
#--------------------------------------------------------

#Step1: Load the actual binary
_SP_lib = ctypes.cdll.LoadLibrary(path)
#Step2: Set up the wrapper parameters (define the function parameters types)
#DECLARE EXTERN C and then
_SP_lib.test_wrapper.restype = None
_SP_lib.test_wrapper.argtypes = []

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#-------------------SPARSE CONV---------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

_SP_lib.gpu_sparse_conv.restype = None
_SP_lib.gpu_sparse_conv.argtypes = [
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
        '''
        Compute the sparse convolution (Porting of Escotin Caffe framework)
        '''
        ifmap_size =  in_channels * (height + pad_h) * (width + pad_w)

        _SP_lib.gpu_sparse_conv(
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

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#-------------------STRETCH KERNEL---------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

_SP_lib.gpu_kernel_stretch.restype = None
_SP_lib.gpu_kernel_stretch.argtypes = [
                                ctypes.c_void_p,#rowptr
                                ctypes.c_void_p,#colidx
                                ctypes.c_int,#M
                                ctypes.c_int,#height
                                ctypes.c_int,#width
                                ctypes.c_int,#pad_h
                                ctypes.c_int,#pad_w
                                ctypes.c_int,#kernel_h
                                ctypes.c_int,#kernel_w
                                ]

def gpu_kernel_stretch(rowptr,colidx,out_channels,height,width,pad_h,pad_w,kernel_h,kernel_w):
    '''
    Stretch the kernel to input shape by modifing the colidx (No extra memory required to store the padding data)
    '''
    _SP_lib.gpu_kernel_stretch(ctypes.c_void_p(rowptr.data_ptr()),
                               ctypes.c_void_p(colidx.data_ptr()),
                               ctypes.c_int(out_channels),
                               ctypes.c_int(height),
                               ctypes.c_int(width),
                               ctypes.c_int(pad_h),
                               ctypes.c_int(pad_w),
                               ctypes.c_int(kernel_h),
                               ctypes.c_int(kernel_w),)
    
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#-------------------PADDING ALIGNMENT KERNEL--------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

_SP_lib.padding_input_alignment.restype = None
_SP_lib.padding_input_alignment.argtypes = [
                                ctypes.c_void_p,#dst
                                ctypes.c_void_p,#src
                                ctypes.c_int,#num_channels
                                ctypes.c_int,#height
                                ctypes.c_int,#width
                                ctypes.c_int,#pad_h
                                ctypes.c_int,#pad_w
                                ]

def padding_input_alignment(dst,src,num_channels,height,width,pad_h,pad_w):
      _SP_lib.padding_input_alignment(  
                                        ctypes.c_void_p(dst.data_ptr()),
                                        ctypes.c_void_p(src.data_ptr()),
                                        ctypes.c_int(num_channels),
                                        ctypes.c_int(height),
                                        ctypes.c_int(width),
                                        ctypes.c_int(pad_h),
                                        ctypes.c_int(pad_w)
                                     )