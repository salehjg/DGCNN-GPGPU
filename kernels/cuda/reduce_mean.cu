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


// Inherited from 'kernel_reduce_sum_4d_try04' @ reduce_sum_4d.cu
__global__ void kernel_reduce_mean_4d_try01(
        const float * __restrict__  g_idata,
        float * __restrict__  g_buff,
        float * __restrict__  g_odata,
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
        const unsigned long TGO) {
    extern __shared__ float smem_buff[];
    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
        unsigned long TPG = dim3; //threads per group
        unsigned long GroupIndex = blockIdx.x * TGPB + threadIdx.x / TPG;
        unsigned long _limit = dim0 * dim1 * dim2 * dim3;
        unsigned long GroupIndexWithinBlock = (GroupIndex - blockIdx.x * TGPB);

        // Ignore incomplete groups at the end of each thread block.
        if (GroupIndexWithinBlock < TGPB && GroupIndex < TGC) {
            //Thread Index In Thread Group
            unsigned long TIITG = (threadIdx.x - (GroupIndexWithinBlock * TPG));
            float thread_sum = 0.0; // sum for current thread in the thread group

            //------------------------------------------------------------
            for (unsigned long iSPT = 0; iSPT < SPT; iSPT++) {
                unsigned long gidx = TGO * GroupIndex + iSPT * dim3 + TIITG;
                if (gidx < _limit) {

                    //if(blockIdx.x==35)
                    //    printf("bid: %06d, tid: %06d, GroupIndex: %06ld, ThreadIndexInGroup: %06ld, gidx: %06ld\n",
                    //        blockIdx.x, threadIdx.x, GroupIndex, TIITG, gidx);

                    thread_sum += g_idata[gidx];
                }
                //else
                //{
                //    printf("** gidx: %ld, GroupIndex: %ld, sum: %f\n",gidx,GroupIndex,thread_sum);
                //}
            }

            //------------------------------------------------------------
            // |---element0s---|-----element1s----|------....|
            smem_buff[TIITG * TGPB + GroupIndexWithinBlock] = thread_sum;
            //if(thread_sum!=10.0)
            //    printf("bid: %06d, tid: %06d, GroupIndex: %06ld, ThreadIndexInGroup: %06ld, thread_sum: %f, smem_buff[%06ld]: %f \n",
            //           blockIdx.x, threadIdx.x, GroupIndex, TIITG, thread_sum, TIITG*TGPB + GroupIndexWithinBlock , smem_buff[TIITG*TGPB + GroupIndexWithinBlock]);


        }

        __syncthreads();

        // parallel reduction of current block's shared memory buffer
        unsigned int thid = threadIdx.x;
        while (thid < TGPB) { // block stride loop

            for (unsigned long stride = TGPB / 2; stride > 0; stride >>= 1) {
                if (thid < stride) {
                    for (int d3 = 0; d3 < TPG; d3++) {
                        smem_buff[d3 * TGPB + (thid)] += smem_buff[d3 * TGPB + (thid + stride)];
                        //printf("#P.Reduc.# bid: %06d, tid: %06d, stride: %06ld, d3: %06d, indx1: %06ld, indx2: %06ld, val1++: %f,\tval2: %f\n",
                        //        blockIdx.x,threadIdx.x,stride,d3, d3*TGPB + (thid) ,d3*TGPB + (thid+stride), smem_buff[d3*TGPB + (thid)] ,smem_buff[d3*TGPB + (thid+stride)]);
                    }
                }
                __syncthreads();
            }
            // -------------------
            thid += blockDim.x;
        }


        __syncthreads();

        if (threadIdx.x == 0) {
            for (int d3 = 0; d3 < TPG; d3++) {
                atomicAdd(&g_odata[d3], smem_buff[d3 * TGPB] / (float)(dim0*dim1*dim2) );
                //atomicAdd(&g_odata[d3], 1);
                //if(blockIdx.x==35)printf("bid: %06d, thid: %06d, smem_buff[%06ld]: %f, g_odata[%06d]: %f\n",blockIdx.x, threadIdx.x, d3 * TGPB, smem_buff[d3 * TGPB], d3, g_odata[d3]);
            }
        }

    }
}


void reduce_mean_4d_try01(
        float* g_idata,
        float* g_odata,
        unsigned long dim0,
        unsigned long dim1,
        unsigned long dim2,
        unsigned long dim3,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3)
{
    if( !(overaxis0 && overaxis1 && overaxis2 && !overaxis3) ) {
        printf("ERROR @reduce_sum_4d_try01 --NOT IMPLEMENTED\n"); return;
    }

    unsigned long block = BLOCK_SIZE;
    unsigned long SPT,TGC,TGO,TGPB, grid,TPG;

    //Dim3 slice per thread
    SPT = 2048; //cte

    //thread group offset
    TGO = dim3 * SPT;

    //thread group count
    TGC = (unsigned long)((dim0*dim1*dim2+(SPT-1))/SPT);

    //thread group per block
    TGPB = (unsigned long)((BLOCK_SIZE)/ dim3);
    if(TGPB%2 && TGPB > 1) TGPB--;

    //grid size
    grid = ( TGC+(TGPB-1) ) / TGPB;

    TPG = (unsigned long)dim3; //threads per group

    printf("-------------------------------------------------------\n");
    printf("KERNEL_GRID  : %ld\n", grid);
    printf("KERNEL_BLOCK : %ld\n", block);
    printf("KERNEL_SPT :   %ld\n", SPT);
    printf("KERNEL_TGO :   %ld\n", TGO);
    printf("KERNEL_TGC :   %ld\n", TGC);
    printf("KERNEL_TGPB :  %ld\n", TGPB);

    float* g_buffer;
    CHECK(cudaMalloc((float**)&g_buffer, (TGC*TPG)*sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
    CHECK(cudaMemset(g_buffer,0,(TGC*TPG)*sizeof(float)));
    kernel_reduce_mean_4d_try01 <<<grid, block, TGPB*TPG*sizeof(float)>>> (
        g_idata, g_buffer, g_odata,
        dim0, dim1, dim2, dim3,
        overaxis0, overaxis1, overaxis2, overaxis3,
        TGC,
        TGPB,
        SPT,
        TGO
    );
}