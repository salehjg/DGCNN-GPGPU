#include <stdio.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include "common.h"


// COPYRIGHT TO CHARLESQ34 @ GitHub : PointNet++
// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void kernel_group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}

void gather(
        const float* points,   // (b,n,c)
        const int *indices,    // (b,m,nsample)
        float* output,  // (b,m,nsample,c)
        int b,
        int n,
        int c,
        int m,
        int nsample){
    unsigned int blockSize = 256;

    cudaDeviceSynchronize();
    kernel_group_point_gpu<<<b,blockSize>>>(b,n,c,m,nsample,points,indices,output);
    cudaDeviceSynchronize();
}