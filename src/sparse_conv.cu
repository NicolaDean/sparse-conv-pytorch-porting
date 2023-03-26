#include <cuda_runtime.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#include <cmath>


extern "C" void test_wrapper(); 

extern "C" void gpu_sparse_conv(bool FUSE_RELU, int num, const void *input, const int ifmap_size, const void *rowptr, 
	const void *colidx, const void *values, const void *bias, int height, int width, int pad_h, int pad_w, 
	int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, void *output, int num_oc, int num_groups);

extern "C" void gpu_kernel_stretch(const void *rowptr, void *colidx, int M, 
		int height, int width, int pad_h, int pad_w, int kernel_h, int kernel_w);

static unsigned CudaTest(const char *msg) {
	cudaError_t e;
	cudaDeviceSynchronize();
	if (cudaSuccess != (e = cudaGetLastError())) {
		fprintf(stderr, "%s: %d\n", msg, e); 
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		exit(-1);
		//return 1;
	}
	return 0;
}

void gpu_sparse_conv2(bool FUSE_RELU, int num, const void *input, const int ifmap_size, const void *rowptr, 
	const void *colidx, const void *values, const void *bias, int height, int width, int pad_h, int pad_w, 
	int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, void *output, int num_oc, int num_groups){
		printf("Num of inputs: %d\n",num);
		printf("Num of input channels: %d\n",num);
		printf("Num of output channels: %d\n",num_oc);

		printf("Kernel Shape (%d,%d)\n",kernel_h,kernel_w);
		printf("Padding Shape (%d,%d)\n",pad_h,pad_w);
		printf("Stride Shape (%d,%d)\n",stride_h,stride_w);
		printf("Dilation Shape (%d,%d)\n",dilation_h,dilation_w);

		float * v = (float*)values;

		for(int i=0;i<5;i++){
			printf("Val[%d]:%f\n",i,v[i]);
		} 

	}


void test_wrapper(){
	printf("Banana\n");
	
}

__global__ void stretch_kernel(const int *rowptr, int *colidx, int M,
		int height, int width, int pad_h, int pad_w, int kernel_h, int kernel_w) {
	int out_channel = blockIdx.x * blockDim.x + threadIdx.x;
	if(out_channel < M) {
		for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
			int col = colidx[j];
			int kernel_col = col % kernel_w;
			int kernel_row = (col / kernel_w) % kernel_h;
			int in_channel = col / (kernel_w * kernel_h);
			//assert(in_channel < conv_in_channels_);
			colidx[j] = (in_channel * (height + pad_h) + kernel_row) * (width + pad_w) + kernel_col;
		}
	}
}

void caffe_gpu_stretch(const int *rowptr, int *colidx, int M, 
		int height, int width, int pad_h, int pad_w, int kernel_h, int kernel_w) {
	int nthreads = 512;
	int nblocks = (M - 1) / nthreads + 1;
	stretch_kernel<<<nblocks, nthreads>>>(rowptr, colidx, M, 
			height, width, pad_h, pad_w, kernel_h, kernel_w);
	CudaTest("Kernel Stretch Error");
}

void gpu_kernel_stretch(const void *rowptr, void *colidx, int M, 
		int height, int width, int pad_h, int pad_w, int kernel_h, int kernel_w){
		
	printf("Stretch kernel\n");
	caffe_gpu_stretch((int*)rowptr,(int*)colidx,M,height,width,pad_h,pad_w,kernel_h,kernel_w);		
}


template <typename Dtype>
__global__ void sconv_dilation(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input, int height, int width, int pad_h, int pad_w, int stride_h, int stride_w, 
		int dilation_h, int dilation_w, int kernel_h, int kernel_w, const Dtype *bias,
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	int output_row = blockIdx.x * blockDim.x + threadIdx.x;
	int output_col = blockIdx.y * blockDim.y + threadIdx.y;
	int out_channel = blockIdx.z * blockDim.z + threadIdx.z;
	if (output_row < output_h) {
		if (output_col < output_w) {
			if(out_channel < num_oc) {
				Dtype sum = 0;
				for (int j = rowptr[out_channel]; j < rowptr[out_channel + 1]; ++j) {
					int col = colidx[j];
					int kernel_col = col%(width + pad_w);
					int kernel_row = (col/(width + pad_w))%(height + pad_h);
					int in_channel = col/((width + pad_w)*(height + pad_h));
					int input_row = kernel_row * dilation_h + output_row * stride_h;
					int input_col = kernel_col * dilation_w + output_col * stride_w;
					sum += values[j] * input[(in_channel * (height + pad_h) + input_row) * (width + pad_w) + input_col];
				}
				output[(out_channel*output_h + output_row)*output_w + output_col] = sum;
			}
		}
	}
}

template <typename Dtype>
__global__ void sconv_base(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int kernel_h, int kernel_w, const Dtype *bias,
		Dtype *output, int num_oc, const int output_h, const int output_w) {

	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc = blockIdx.z * blockDim.z + threadIdx.z;

	if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				//Project output grid on input grid
				const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				Dtype sum = 0;
				for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
					sum += values[j] * in_ptr[colidx[j]];
				}
				output[(oc * output_h + output_row) * output_w + output_col] = sum;
			}
		}
	}
}

template <typename Dtype>
__global__ void sconv_relu_base(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input, int height, int width, int pad_h, int pad_w,
		int stride_h, int stride_w, int kernel_h, int kernel_w, const Dtype *bias,
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc = blockIdx.z * blockDim.z + threadIdx.z;
	if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				Dtype sum = bias[oc];
				for (int j = rowptr[oc]; j < rowptr[oc + 1]; ++j) {
					sum += values[j] * in_ptr[colidx[j]];
				}
				output[(oc * output_h + output_row) * output_w + output_col] = max(sum, (Dtype)0);
			}
		}
	}
}

template <typename Dtype>
__global__ void sconv_batch_base(const int *rowptr, const int *colidx, const Dtype *values,
		const Dtype *input, int ifmap_size, int height, int width, int pad_h, int pad_w, 
		int stride_h, int stride_w, int kernel_h, int kernel_w, const Dtype *bias, 
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	const int row = blockIdx.y * blockDim.y + threadIdx.y; // the row id of output channel
	const int col = blockIdx.x * blockDim.x + threadIdx.x; // the column id of output channel
	const int zid = blockIdx.z * blockDim.z + threadIdx.z;
	const int oc_id = zid % num_oc; // the output channel id
	const int fmap_id = zid / num_oc; // the feature map id
	const int ofmap_size = output_h * output_w * num_oc;
	//if (zid < batch_size * num_oc) {
	if (row < output_h) {
		if (col < output_w) {
			const Dtype *ifmap = input + fmap_id * ifmap_size;
			const Dtype *in_ptr = ifmap + row * stride_h * (width + pad_w) + col * stride_w;
			Dtype sum = 0;
			for (int j = rowptr[oc_id]; j < rowptr[oc_id + 1]; ++j) {
				sum += values[j] * in_ptr[colidx[j]];
			}
			output[fmap_id * ofmap_size + (oc_id * output_h + row) * output_w + col] = sum;
		}
	}
}


#define BLOCK_SIZE 256 // 4*4*32
#define WARP_SIZE 32
#define VLEN 32
#define OC_BLOCK 1
#define DIVIDE_INTO(x,y) ((x + y - 1)/y)
#define MIN(x,y) ((x < y)? x : y)
#define SHMEM_SIZE 1024
#define ITER (SHMEM_SIZE/BLOCK_SIZE)
#define REG_BLOCK_SIZE 32 // number of registers per warp
#define REG_BLOCK_H 4
#define REG_BLOCK_W 1
#define COARSEN 4

template <typename Dtype, int TILE_W, int TILE_H>
__global__ void sconv_shm(const int * rowptr, const int * colidx, const Dtype * values, 
		const Dtype * __restrict__ input, const int height, const int width, const int pad_h, const int pad_w, 
		const int stride_h, const int stride_w, const int kernel_h, const int kernel_w, const Dtype *bias,
		Dtype *output, const int num_oc, const int output_h, const int output_w) {

	
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc = blockIdx.z * blockDim.z + threadIdx.z;

	

	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;

	//for(int oc = oc_id; oc < num_oc; oc += gridDim.z) {
	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	const int length = row_end - row_start;
	const int BLK_SIZE = TILE_H * TILE_W;

	//printf("Thread: (%d,%d,%d) => [%d,%d]\n",output_row,output_col,oc,row_start,row_end);
	Dtype sum = 0;
	//Dtype sum = bias[oc];
	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}

		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				int end = MIN(SHMEM_SIZE, length - i);
				for (int off = 0; off < end; ++off) {
					Dtype weight = values_s[off];
					int pos = colidx_s[off];
					sum += weight * __ldg(in_ptr+pos);
				}
			}
		}
		__syncthreads();
	}

	//if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				output[(oc * output_h + output_row) * output_w + output_col] = sum;
			}
		}
	//}
}

template <typename Dtype, int TILE_H, int TILE_W, int WIDTH, int K, int PAD = (K - 1) / 2>
__global__ void sconv_coarsened(const int * rowptr, const int * colidx, const Dtype * values, 
		const Dtype * __restrict__ input, const int height, const int width, const int pad_h, const int pad_w, 
		const int stride_h, const int stride_w, const int kernel_h, const int kernel_w, const Dtype *bias,
		Dtype *output, const int num_oc, const int output_h, const int output_w) {
	//assert(PAD <= (K - 1) / 2);
	//const int WOUT = WIDTH + 2 * PAD - K + 1;
	//const int ALIGNED_W = (WOUT + 16 - 1) / 16 * 16;
	//const int REG_BLOCK_W = (WOUT + VLEN - 1) / VLEN;
	//assert(REG_BLOCK_W <= REG_BLOCK_SIZE);
	//const int REG_BLOCK_H = 4;//WOUT < REG_BLOCK_SIZE/REG_BLOCK_W ? WOUT : REG_BLOCK_SIZE/REG_BLOCK_W;
	// WIDTH = 13 (AlexNet conv3-5), AVX2 : REG_BLOCK_W = 2, REG_BLOCK_H = 7, ALIGNED_W = 16
	// WIDTH = 56 (GoogLeNet), AVX2 : REG_BLOCK_W = 7, REG_BLOCK_H = 2, ALIGNED_W = 64

	const int xid = blockIdx.x * blockDim.x + threadIdx.x;
	const int yid = blockIdx.y * blockDim.y + threadIdx.y;
	const int zid = blockIdx.z * blockDim.z + threadIdx.z;
	const int output_row = yid  * COARSEN;
	const int output_col = xid;
	const int oc = zid;
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;
	const int BLK_SIZE = TILE_H * TILE_W;

	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	//int row_start = __ldg(rowptr+oc);
	//int row_end = __ldg(rowptr+oc+1);
	const int length = row_end - row_start;

	Dtype sum[REG_BLOCK_H][REG_BLOCK_W];
	for (int h = 0; h < REG_BLOCK_H; ++h) {
		for (int w = 0; w < REG_BLOCK_W; ++w) {
			//sum[h][w] = bias[oc];
			sum[h][w] = 0;
		}
	}

	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}

		const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
		int end = MIN(SHMEM_SIZE, length - i);
		for (int off = 0; off < end; ++off) {
			Dtype weight = values_s[off];
			int pos = colidx_s[off];
			for (int h = 0; h < REG_BLOCK_H; ++h) {
				for (int w = 0; w < REG_BLOCK_W; ++w) {
					if (output_row + h < output_h) {
						if (output_col + w < output_w) {
							sum[h][w] += weight * __ldg(in_ptr + pos + h * stride_h * (width + pad_w) + w * stride_w);
						}
					}
				}
			}
		}
		__syncthreads();
	}

	for (int h = 0; h < REG_BLOCK_H; ++h) {
		for (int w = 0; w < REG_BLOCK_W; ++w) {
			if (output_row + h < output_h) {
				if (output_col + w < output_w) {
					output[(oc * output_h + (output_row + h)) * output_w + output_col + w] = sum[h][w];
				}
			}
		}
	}
}

template <typename Dtype, int TILE_H, int TILE_W>
__global__ void sconv_relu_tiled(const int * rowptr, const int * colidx, 
		const Dtype * values, const Dtype * __restrict__ input, 
		int height, int width, int pad_h, int pad_w, int stride_h, 
		int stride_w, int kernel_h, int kernel_w, const Dtype *bias,
		Dtype *output, int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int oc = blockIdx.z * blockDim.z + threadIdx.z;
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;
	const int BLK_SIZE = TILE_H * TILE_W;

	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	const int length = row_end - row_start;
	Dtype sum = bias[oc];
	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}

		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *in_ptr = input + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				int end = MIN(SHMEM_SIZE, length - i);
				for (int off = 0; off < end; ++ off) {
					Dtype weight = values_s[off];
					int pos = colidx_s[off];
					sum += weight * __ldg(in_ptr+pos);
				}
			}
		}
		__syncthreads();
	}
	if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
				output[(oc * output_h + output_row) * output_w + output_col] = max(sum, (Dtype)0);
			}
		}
	}
}

template <typename Dtype, int FMAP_BLOCK, int TILE_H, int TILE_W, int WIDTH, int K, int PAD = (K - 1) / 2>
__global__ void sconv_batch_tiled1(const int * rowptr, const int * colidx, const Dtype * values, 
		const Dtype * __restrict__ input, const int ifmap_size, const int height, const int width, 
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int kernel_h, const int kernel_w,
		const Dtype *bias, Dtype *output, const int num_oc, const int output_h, const int output_w) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int zid = blockIdx.z * blockDim.z + threadIdx.z;
	const int oc = zid % num_oc; // the output channel id
	const int fmap_id = zid / num_oc; // the feature map id
	const int ofmap_size = output_h * output_w * num_oc;
	
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;
	const int BLK_SIZE = TILE_H * TILE_W;

	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	const int length = row_end - row_start;
	Dtype sum[FMAP_BLOCK];
	for (int i=0; i < FMAP_BLOCK; i++)
		//sum[i] = bias[oc];
		sum[i] = 0;
	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}
		if (output_row < output_h) {
			if (output_col < output_w) {
				const Dtype *ifmap = input + (fmap_id * FMAP_BLOCK) * ifmap_size;
				const Dtype *in_ptr = ifmap + output_row * stride_h * (width + pad_w) + output_col * stride_w;
				int end = MIN(SHMEM_SIZE, length - i);
				for (int offset = 0; offset < end; ++ offset) { 
					Dtype weight = values_s[offset];
					int pos = colidx_s[offset];
					for(int k = 0; k < FMAP_BLOCK; k ++) {
						sum[k] += weight * __ldg(in_ptr + pos + k * ifmap_size);
					}
				}
			}
		}
		__syncthreads();
	}
	for(int k = 0; k < FMAP_BLOCK; k ++) {
		if (oc < num_oc) {
			if (output_row < output_h) {
				if (output_col < output_w) {
						output[(fmap_id * FMAP_BLOCK + k) * ofmap_size + (oc * output_h + output_row) * output_w + output_col] = sum[k];
				}
			}
		}
	}
}

template <typename Dtype, int FMAP_BLOCK, int TILE_H, int TILE_W, int WIDTH, int K, int PAD = (K - 1) / 2>
__global__ void sconv_batch_tiled(const int * rowptr, const int * colidx, const Dtype * values, 
		const Dtype * __restrict__ input, const int ifmap_size, const int height, const int width, 
		const int pad_h, const int pad_w, const int stride_h, const int stride_w, const int kernel_h, const int kernel_w,
		const Dtype *bias, Dtype *output, const int num_oc, const int output_h, const int output_w, const int num_groups) {
	const int output_row = blockIdx.y * blockDim.y + threadIdx.y;
	const int output_col = blockIdx.x * blockDim.x + threadIdx.x;
	const int zid = blockIdx.z * blockDim.z + threadIdx.z;
	const int oc = zid % num_oc; // the output channel id
	const int fmap_id = zid / num_oc; // the feature map id
	const int ofmap_size = output_h * output_w * num_oc * num_groups;
	
	__shared__ Dtype values_s[SHMEM_SIZE];
	__shared__ int colidx_s[SHMEM_SIZE];
	const int tid = threadIdx.y * TILE_W + threadIdx.x;
	const int BLK_SIZE = TILE_H * TILE_W;

	const int row_start = rowptr[oc];
	const int row_end = rowptr[oc+1];
	const int length = row_end - row_start;
	
	Dtype sum[FMAP_BLOCK];
	for (int i=0; i < FMAP_BLOCK; i++)
		//sum[i] = bias[oc];
		sum[i] = 0;
	for(int i = 0; i < length; i += SHMEM_SIZE) {
		int base_addr = row_start + i;
		for (int j = 0; j < SHMEM_SIZE; j += BLK_SIZE) {
			int index_s = tid + j;
			int index = base_addr + index_s;
			if(index >= row_end) {
				colidx_s[index_s] = 0;
				values_s[index_s] = 0;
			} else {
				colidx_s[index_s] = colidx[index];
				values_s[index_s] = values[index];
			}
			__syncthreads();
		}
		for(int k = 0; k < FMAP_BLOCK; k ++) {
			if (output_row < output_h) {
				if (output_col < output_w) {
					const Dtype *ifmap = input + (fmap_id * FMAP_BLOCK) * ifmap_size * num_groups;
					const Dtype *in_ptr = ifmap + output_row * stride_h * (width + pad_w) + output_col * stride_w;
					int end = MIN(SHMEM_SIZE, length - i);
					for (int offset = 0; offset < end; ++ offset) {
						Dtype weight = values_s[offset];
						int pos = colidx_s[offset];
						sum[k] += weight * __ldg(in_ptr + pos + k * ifmap_size * num_groups);
					}
				}
			}
		}
		__syncthreads();
	}
	for(int k = 0; k < FMAP_BLOCK; k ++) {
		//if (oc < num_oc) {
		if (output_row < output_h) {
			if (output_col < output_w) {
					output[(fmap_id * FMAP_BLOCK + k) * ofmap_size + (oc * output_h + output_row) * output_w + output_col] = sum[k];
			}
		}
	}
}

#define TILED_KERNEL
template <typename Dtype>
void caffe_gpu_sconv(bool FUSE_RELU, int num, const Dtype *input, const int ifmap_size, const int *rowptr, 
	const int *colidx, const Dtype *values, const Dtype *bias, int height, int width, int pad_h, int pad_w, 
	int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, Dtype *output, int num_oc, int num_groups)
{
	printf("Num of inputs: %d\n",num);
	printf("Num of input channels: %d\n",num);
	printf("Num of output channels: %d\n",num_oc);
	printf("Num of groups: %d\n",num_groups);
	

	printf("Kernel Shape (%d,%d)\n",kernel_h,kernel_w);
	printf("Padding Shape (%d,%d)\n",pad_h,pad_w);
	printf("Stride Shape (%d,%d)\n",stride_h,stride_w);
	printf("Dilation Shape (%d,%d)\n",dilation_h,dilation_w);
	/*
	printf("Rowptr: [");
	for(int i=0;i<6;i++){
			printf("%f,",rowptr[i]);
	}*/
	printf("]\n");
	//print_device_info(0);
	//Compute the output shape based on the 
	const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int output_w = (width  + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

	printf("OUTPUT SHAPE: [%d,%d]\n",output_h,output_w);
	//printf("We have a sparse conv with:\n-INPUT:(%d,%d,%d)\n-Kernel:(%d,%d,%d)-Output:(%d,%d,%d)\n",);
	int TILE_H = 16;
	int TILE_W = 16;
	int ntiles_h = (output_h - 1) / TILE_H + 1;
	int ntiles_w = (output_w - 1) / TILE_W + 1;
	int nblocks = (num_oc - 1) / OC_BLOCK + 1;
	//printf("num=%d, nblocks=%d, num_oc=%d\n", num, nblocks, num_oc);
	//printf("height=%d, width=%d, output_h=%d, output_w=%d\n", height, width, output_h, output_w);
	//printf("stride_h=%d, stride_w=%d, pad_h=%d, pad_width=%d\n", stride_h, stride_w, pad_h, pad_w);

	//Dilatation is a kernel with some empty columns and rows
	if (dilation_h != 1 || dilation_w != 1) {
		dim3 threads(TILE_W, TILE_H, OC_BLOCK);
		dim3 grid(ntiles_w, ntiles_h, nblocks);
		sconv_dilation<Dtype><<<grid, threads>>>(rowptr, colidx, values, input, 
			height, width, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, 
			kernel_h, kernel_w, bias, output, num_oc, output_h, output_w);
	} else if (stride_h == 1 && stride_w == 1 && height == width && kernel_h == kernel_w && pad_h == pad_w) {
		if(FUSE_RELU) {
			dim3 threads(16, 16, OC_BLOCK);
			dim3 grid(ntiles_w, ntiles_h, nblocks);
			sconv_relu_tiled<Dtype,16,16><<<grid, threads>>>(rowptr, colidx, values, input, 
				height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
				bias, output, num_oc, output_h, output_w);
		} else {
			if(num == 1) {
				if(height == 27) {
				//if(0) {
					ntiles_w = DIVIDE_INTO(output_w, 32);
					ntiles_h = DIVIDE_INTO(output_h, 8);
					dim3 grid(ntiles_w, ntiles_h, nblocks);
					//dim3 threads(32, 8, 1);
					//sconv_coarsened<Dtype,8,32,27,1><<<grid, threads>>>(rowptr, colidx, values, input, 
					//	height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
					//	bias, output, num_oc, output_h, output_w);
					dim3 threads(32, 8, 1);
					sconv_shm<Dtype,32,8><<<grid, threads>>>(rowptr, colidx, values, input, 
						height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
						bias, output, num_oc, output_h, output_w);
				//} else(height == 13) {
				} else {
/*
					const int nthreads = 256;
					cudaDeviceProp deviceProp;
					//CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp, 0));
					const int nSM = 28;//deviceProp.multiProcessorCount;
					//const int max_blocks_per_SM = maximum_residency(sconv_shm<Dtype,16,16>, nthreads, 0);
					const int max_blocks_per_SM = 8;
					const int max_blocks = max_blocks_per_SM * nSM;
					nblocks = std::min(max_blocks, nblocks);
					//printf("Launching CUDA solver: %d CTAs (max %d/SM), %d threads/CTA ...\n", nblocks, max_blocks_per_SM, nthreads);
//*/	
					dim3 threads(TILE_W, TILE_H, 1);
					dim3 grid(ntiles_w, ntiles_h, nblocks);
					sconv_shm<Dtype,16,16><<<grid, threads>>>(rowptr, colidx, values, input, 
						height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
						bias, output, num_oc, output_h, output_w);
				}
			} else {
				dim3 threads(16, 16, 1);
				//if(nblocks >= 128 && nblocks < 224) {
				if(0) {
					nblocks = (num/1) * ((num_oc - 1) / OC_BLOCK + 1);
					dim3 grid(ntiles_w, ntiles_h, nblocks);
					sconv_batch_tiled<Dtype,1,16,16,56,1><<<grid, threads>>>(rowptr, colidx, values, 
						input, ifmap_size, height, width, pad_h, pad_w, stride_h, stride_w, 
						kernel_h, kernel_w, bias, output, num_oc, output_h, output_w, num_groups);
				} else {	
					nblocks = (num/2) * ((num_oc - 1) / OC_BLOCK + 1);
					dim3 grid(ntiles_w, ntiles_h, nblocks);
					sconv_batch_tiled<Dtype,2,16,16,56,1><<<grid, threads>>>(rowptr, colidx, values, 
						input, ifmap_size, height, width, pad_h, pad_w, stride_h, stride_w, 
						kernel_h, kernel_w, bias, output, num_oc, output_h, output_w, num_groups);
				}
			}
		}
	} else {
		// fall through to the default path
		dim3 threads(TILE_W, TILE_H, OC_BLOCK);
		dim3 grid(ntiles_w, ntiles_h, nblocks);
		if(FUSE_RELU) {
			sconv_relu_base<Dtype><<<grid, threads>>>(rowptr, colidx, values, input, 
				height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
				bias, output, num_oc, output_h, output_w);
		} else {
			if(num == 1){
				sconv_base<Dtype><<<grid, threads>>>(rowptr, colidx, values, input, 
					height, width, pad_h, pad_w, stride_h, stride_w, kernel_h, kernel_w, 
					bias, output, num_oc, output_h, output_w);
			}
			else {
				nblocks = num * ((num_oc - 1) / OC_BLOCK + 1);
				sconv_batch_base<Dtype><<<grid, threads>>>(rowptr, colidx, values, input, 
					ifmap_size, height, width, pad_h, pad_w, stride_h, stride_w, 
					kernel_h, kernel_w, bias, output, num_oc, output_h, output_w);
			}
		}
	}


	printf("End of computation\n");
	CudaTest("sconv_kernel solving failed");
}

template void caffe_gpu_sconv<int>(bool FUSE_RELU, int num, const int *input, const int ifmap_size, const int *rowptr, 
		const int *colidx, const int *values, const int *bias, int height, int width, int pad_h, int pad_w, 
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, int *output, int num_oc, int num_groups);
template void caffe_gpu_sconv<float>(bool FUSE_RELU, int num, const float *input, const int ifmap_size, const int *rowptr, 
		const int *colidx, const float *values, const float *bias, int height, int width, int pad_h, int pad_w, 
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, float *output, int num_oc, int num_groups);
template void caffe_gpu_sconv<double>(bool FUSE_RELU, int num, const double *input, const int ifmap_size, const int *rowptr, 
		const int *colidx, const double *values, const double *bias, int height, int width, int pad_h, int pad_w, 
		int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, double *output, int num_oc, int num_groups);


void gpu_sparse_conv(bool FUSE_RELU, int num, const void *input, const int ifmap_size, const void *rowptr, 
	const void *colidx, const void *values, const void *bias, int height, int width, int pad_h, int pad_w, 
	int stride_h, int stride_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, void *output, int num_oc, int num_groups)
	{
		//TODO Apply casting of variables
		//SIMPLY CALL THE FUNCTION
		caffe_gpu_sconv<float>(FUSE_RELU,num,(float*)input,ifmap_size,(int*)rowptr,(int*)colidx,(float*)values,(float*)bias,height,width,pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,kernel_h,kernel_w,(float*)output,num_oc,num_groups);
	}