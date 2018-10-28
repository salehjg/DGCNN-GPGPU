#include <stdio.h>
#include <cuda_runtime_api.h>
#include <npp.h>

template <typename T>
__global__ void kernel_split_3d_overdim2(
        const T* __restrict__  g_i,
        T* __restrict__  g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int new_dim2){
    // threads are mapped into output tensor
    unsigned long _len_output = dim0*dim1*new_dim2;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < _len_output) {
        unsigned int d0 = idx / (dim1 * new_dim2);
        unsigned int d1 = (idx % (dim1 * new_dim2)) / (new_dim2);
        unsigned int d2 = (idx % (new_dim2)) / 1;

        g_o[idx] = g_i[d0 * dim1 * dim2 + d1 * dim2 + d2];
    }
}

void split_3d_overdim2_float(
        const float* __restrict__  g_i,
        float* __restrict__  g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int new_dim2){
    unsigned int blockSize, gridSize;
    blockSize = 256;
    gridSize = ((dim0*dim1*new_dim2)+blockSize-1)/blockSize;
    //cudaDeviceSynchronize();
    kernel_split_3d_overdim2<float><<<gridSize,blockSize>>>(g_i,g_o,dim0,dim1,dim2,new_dim2);
    //cudaDeviceSynchronize();
}

void split_3d_overdim2_integer(
        const int* __restrict__  g_i,
        int* __restrict__  g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int new_dim2){
    unsigned int blockSize, gridSize;
    blockSize = 256;
    gridSize = ((dim0*dim1*new_dim2)+blockSize-1)/blockSize;
    //cudaDeviceSynchronize();
    kernel_split_3d_overdim2<int><<<gridSize,blockSize>>>(g_i,g_o,dim0,dim1,dim2,new_dim2);
    //cudaDeviceSynchronize();
}
