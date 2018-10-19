#include <stdio.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include "common.h"

extern
void split_3d_overdim2_float(
        const float*  __restrict__ g_i,
        float*  __restrict__ g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int new_dim2);
extern
void split_3d_overdim2_integer(
        const int*  __restrict__ g_i,
        int*  __restrict__ g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int new_dim2);


// COPYRIGHT TO CHARLESQ34 @ GitHub : PointNet++
// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void kernel_selection_sort_gpu(int b, int n, int m, int k, const float * __restrict__ dist, int * __restrict__ outi, float * __restrict__ out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s;
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

void top_k(
        const float* __restrict__  distance_matrix,   // (b,m,n)
        int * __restrict__ output_indices,            // (b,m,k)
        float* output_values,           // (b,m,k)
        int b,
        int n,
        int m,
        int k){
    unsigned int blockSize = 256;

    float* tmpVal;
    int* tmpIndices;
    CHECK(cudaMalloc((float**)&tmpVal  , (b*m*n)*sizeof(float)));
    CHECK(cudaMalloc((int**)&tmpIndices, (b*m*n)*sizeof(int)));



    //cudaDeviceSynchronize();

    kernel_selection_sort_gpu<<<b,blockSize>>>(b,n,m,k,distance_matrix,tmpIndices,tmpVal);
    split_3d_overdim2_float(tmpVal, output_values,b,m,n,k);     //split BxMxN into BxMxK (float)
    split_3d_overdim2_integer(tmpIndices, output_indices,b,m,n,k);  //split BxMxN into BxMxK (integer)

    //cudaDeviceSynchronize();



    CHECK(cudaFree(tmpVal));
    CHECK(cudaFree(tmpIndices));
}