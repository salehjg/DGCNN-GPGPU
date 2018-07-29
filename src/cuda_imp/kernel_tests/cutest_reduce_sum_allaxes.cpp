//
// Created by saleh on 7/18/18.
//
#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>

/*
 * An example of using CUDA shuffle instructions to optimize performance of a
 * parallel reduction.
 */

#define DIM     128
#define SMEMDIM 4     // 128/32 = 8

extern void reduce_sum_all_axes(dim3 , dim3 , float *, float *, unsigned int);


// Recursive Implementation of Interleaved Pair Approach
float recursiveReduce(float *data, int const size)
{
    if (size == 1) return data[0];

    int const stride = size / 2;

    for (int i = 0; i < stride; i++)
        data[i] += data[i + stride];

    return recursiveReduce(data, stride);
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\n%s starting reduction at ", argv[0]);
    printf("\ndevice %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // initialization
    int size = (1 << 24) * 1; // total number of elements to reduce
    printf("\nwith array size %d  ", size);

    // execution configuration
    int blocksize = 128;   // initial block size (MUST BE SAME AS 'DIM' DEFINED IN KERNEL FILE (*.CU)

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t fbytes = size * sizeof(float);
    float *h_f_idata = (float *) malloc(fbytes);
    float *h_f_odata = (float *) malloc(grid.x * sizeof(float));

    size_t ibytes = size * sizeof(int);
    int *h_i_idata = (int *) malloc(ibytes);
    int *h_i_odata = (int *) malloc(grid.x * sizeof(int));

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_f_idata[i] = (float)(rand() & 0xFF);
        h_i_idata[i] = (int)(rand() & 0xFF);
    }

    double iStart, iElaps;
    float f_gpu_sum = 0.0f;
    int i_gpu_sum = 0;

    // allocate device memory
    float *d_f_idata = NULL;
    float *d_f_odata = NULL;
    CHECK(cudaMalloc((void **) &d_f_idata, fbytes));
    CHECK(cudaMalloc((void **) &d_f_odata, grid.x * sizeof(float)));

    int *d_i_idata = NULL;
    int *d_i_odata = NULL;
    CHECK(cudaMalloc((void **) &d_i_idata, ibytes));
    CHECK(cudaMalloc((void **) &d_i_odata, grid.x * sizeof(int)));

    //
    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    reduce_sum_all_axes(grid.x/4,block,d_f_idata,d_f_odata,size);

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_f_odata, d_f_odata, grid.x / 4 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    f_gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) f_gpu_sum += h_f_odata[i];

    printf("gpu Cmptnroll8Float  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, f_gpu_sum, grid.x / 4, block.x);


    // free host memory
    free(h_i_idata);
    free(h_i_odata);
    free(h_f_idata);
    free(h_f_odata);

    // free device memory
    CHECK(cudaFree(d_i_idata));
    CHECK(cudaFree(d_i_odata));
    CHECK(cudaFree(d_f_idata));
    CHECK(cudaFree(d_f_odata));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


