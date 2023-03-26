
#DEPENDENCY

#1. glog/logging.h => sudo apt install libgoogle-glog-dev => https://codeyarns.com/tech/2017-10-26-how-to-install-and-use-glog.html#gsc.tab=0
#2. gflags  https://github.com/gflags/gflags/blob/master/INSTALL.md
#3.
SRC = ./src/sparse_conv.cu
COMPILED_LIB = ./lib/sparse_conv.so

all:
	nvcc -arch=sm_50  -gencode=arch=compute_50,code=sm_50 -arch=sm_52  -gencode=arch=compute_52,code=sm_52  -gencode=arch=compute_60,code=sm_60  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70  -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 -Xptxas "-v -dlcm=ca" -shared -Xcompiler=\"-fPIC\" -o ${COMPILED_LIB} ${SRC}
	
	
