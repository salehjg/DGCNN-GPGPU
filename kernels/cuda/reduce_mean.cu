#include <stdio.h>
#include <cuda_runtime_api.h>
#include "common.h"


#define BLOCK_SIZE 1024
/*
**LA_Mean: Rank: 4  dims: 5,1024,20,64   overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,64   overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,128  overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,128  overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,1,1024  overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,1,1024  overaxes: 1,1,1,0;


**LA_Mean: Rank: 2  dims: 5,512,0,0      overaxes: 1,0,0,0;
**LA_Mean: Rank: 2  dims: 5,512,0,0      overaxes: 1,0,0,0;
**LA_Mean: Rank: 2  dims: 5,256,0,0      overaxes: 1,0,0,0;
**LA_Mean: Rank: 2  dims: 5,256,0,0      overaxes: 1,0,0,0;


**LA_Mean: Rank: 4  dims: 5,1024,20,64   overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,64   overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,64   overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,64   overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,64   overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,64   overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,128  overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,20,128  overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,1,1024  overaxes: 1,1,1,0;
**LA_Mean: Rank: 4  dims: 5,1024,1,1024  overaxes: 1,1,1,0;


**LA_Mean: Rank: 2  dims: 5,512,0,0      overaxes: 1,0,0,0;
**LA_Mean: Rank: 2  dims: 5,512,0,0      overaxes: 1,0,0,0;
**LA_Mean: Rank: 2  dims: 5,256,0,0      overaxes: 1,0,0,0;
**LA_Mean: Rank: 2  dims: 5,256,0,0      overaxes: 1,0,0,0;


Rank and Axes :
    -Rank=4 ==> TTTF
    -Rank=2 ==> TF
*/

extern __global__ void kernel_reduce_sum_4d_try04(
        const float * __restrict__  g_idata,
        float * __restrict__  g_buff,
        float * __restrict__  g_odata,
        const int pow_y,
        const unsigned long dim0,
        const unsigned long dim1,
        const unsigned long dim2,
        const unsigned long dim3,
        const bool overaxis0,
        const bool overaxis1,
        const bool overaxis2,
        const bool overaxis3,

        const unsigned long TGC,
        const unsigned long TGPB,
        const unsigned long SPT,
        const unsigned long TGO);


// Multiplies constant coef. into whole array element wise.
__global__ void kernel_divide_by_const_try01(
        const float * __restrict__  g_idata,
        float * __restrict__  g_odata,
        const unsigned long dim,
        const float coef) {


    unsigned long tidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(tidx<dim){
        //printf("*** tidx: %ld, coef: %f \t\t g_i: %f\n",tidx,coef, g_idata[tidx]);
        g_odata[tidx] = g_idata[tidx] / coef;
        //printf("*** tidx: %ld, coef: %f \t\t g_o: %f\n",tidx,coef,  g_odata[tidx]);
    }


}



void reduce_mean_4d_try02(
        float* g_idata,
        float* g_odata,
        unsigned long dim0,
        unsigned long dim1,
        unsigned long dim2,
        unsigned long dim3,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3){

    //cudaStream_t local_stream;
    //cudaStreamCreate(&local_stream);


    float* g_tempbuff;

    if( !(overaxis0 && overaxis1 && overaxis2 && !overaxis3) ) {
        printf("ERROR @reduce_sum_4d_try01 --NOT IMPLEMENTED\n"); return;
    }

    // 1. reduce_sum
    {
        unsigned long block = BLOCK_SIZE;
        unsigned long SPT, TGC, TGO, TGPB, grid, TPG;

        //Dim3 slice per thread
        SPT = 512; //cte

        //thread group offset
        TGO = dim3 * SPT;

        //thread group count
        TGC = (unsigned long) ((dim0 * dim1 * dim2 + (SPT - 1)) / SPT);

        //thread group per block
        TGPB = (unsigned long) ((BLOCK_SIZE) / dim3);
        if (TGPB % 2 && TGPB > 1) TGPB--;

        //grid size
        grid = (TGC + (TGPB - 1)) / TGPB;

        TPG = (unsigned long) dim3; //threads per group

        printf("-------------------------------------------------------\n");
        printf("KERNEL_SHAPE  : %ldx%ldx%ldx%ld\n", dim0,dim1,dim2,dim3);
        printf("KERNEL_GRID  : %ld\n", grid);
        printf("KERNEL_BLOCK : %ld\n", block);
        printf("KERNEL_SPT :   %ld\n", SPT);
        printf("KERNEL_TGO :   %ld\n", TGO);
        printf("KERNEL_TGC :   %ld\n", TGC);
        printf("KERNEL_TGPB :  %ld\n", TGPB);

        float *g_buffer;
        CHECK(cudaMalloc((float **) &g_tempbuff, (dim3) * sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
        //CHECK(cudaMalloc((float **) &g_buffer, (TGC * TPG) * sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
        CHECK(cudaMemset(g_tempbuff, 0, (dim3) * sizeof(float) ));
        kernel_reduce_sum_4d_try04 << < grid, block, TGPB * TPG * sizeof(float)/*, local_stream*/ >> > (
            g_idata, /*g_buffer*/ nullptr, g_tempbuff,1,
            dim0, dim1, dim2, dim3,
            overaxis0, overaxis1, overaxis2, overaxis3,
            TGC,
            TGPB,
            SPT,
            TGO
        );
        //CHECK(cudaFree(g_buffer));
    }

    CHECK(cudaDeviceSynchronize());

    // 2. Multiplying (1/n) to each element of resulted tensor from step 1.
    {
        unsigned long len = dim3; //Axes combination is TTTF
        unsigned long block,grid;

        float coef = (dim0*dim1*dim2);
        printf("WRAPPER: COEF: %f\n",coef);


        block = BLOCK_SIZE;
        grid = (len + block -1 )/(block);
        kernel_divide_by_const_try01 << < grid, block, 0/*, local_stream */>> > (
                g_tempbuff, g_odata, len, coef
                );
        CHECK(cudaFree(g_tempbuff));
    }

}