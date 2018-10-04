//
// Created by saleh on 9/27/18.
//

#include <assert.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "common.h"

__global__ void kernel_conv2d_mlp_try01(
        const float* gInput_i,
        const float* gWeight_i,
        float* gOutput_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int chOut){
    // Dimension Example:
    //      input shape  = 5 x 1024 x 20 x 6
    //      weight shape = 1 x 1 x 6 x 64
    //      chOut = 64

    unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
    //const unsigned int buff_len = dim3*chOut; //this should ALWAYS be an even number!
    unsigned int d3 = tid % dim3;
    unsigned int d2 = (tid % (dim2*dim3)) / (dim3);
    unsigned int d1 = (tid % (dim1*dim2*dim3)) / (dim2*dim3);
    unsigned int d0 = tid / (dim1*dim2*dim3);
    unsigned long idx1;

    float mulVal;
    for(int iCh = 0; iCh<chOut; iCh++){
        mulVal = gInput_i[tid] * gWeight_i[d3*chOut+iCh];
        idx1 = d0*dim1*dim2*chOut+ d1*dim2*chOut+ d2*chOut+ iCh ;
        atomicAdd(&gOutput_o[idx1], mulVal);
    }
}

void conv2d_mlp_try01(
        float* gInput_i,
        float* gWeight_i,
        float* gOutput_o,
        unsigned int B,
        unsigned int N,
        unsigned int K,
        unsigned int D,
        unsigned int chOut){
    assert(D<=1024); // this kernel cannot accept dim3>1024
    ///TODO: CHECK GRID SIZE DEVICE LIMITATION
    unsigned long blockSize = D;
    unsigned long gridSize = B*N*K;

    printf("BlockSize: \t%lu\n",blockSize);
    printf("GridSize: \t%lu\n",gridSize);
    printf("B: %u\n",B);
    printf("N: %u\n",N);
    printf("K: %u\n",K);
    printf("D: %u\n",D);
    printf("C: %u\n",chOut);

    kernel_conv2d_mlp_try01 <<<gridSize, blockSize>>>(
            gInput_i, gWeight_i, gOutput_o, B, N, K, D, chOut);
}


















/*
__global__ void kernel_conv2d_mlp_try02(
        const float* __restrict__ gInput_i,
        const float* __restrict__ gWeight_i,
        float* __restrict__ gOutput_o,

        const unsigned long dim0,
        const unsigned long dim1,
        const unsigned long dim2,
        const unsigned long dim3,
        const unsigned long chOut,

        const unsigned long TGC,
        const unsigned long TGPB,
        const unsigned long SPT,
        const unsigned long TGO) {

    extern __shared__ float smem_buff[];
    unsigned long TPG = dim3; //threads per group
    unsigned long GroupIndex = blockIdx.x * TGPB + threadIdx.x / TPG;
    unsigned long _limit = dim0*dim1*dim2*dim3;
    unsigned long GroupIndexWithinBlock = (GroupIndex - blockIdx.x *TGPB);

    {
        // Fill unused shared mem with zeros! - Share memory is NOT initialized to zero
        // https://stackoverflow.com/questions/22172881/why-cuda-shared-memory-is-initialized-to-zero?noredirect=1&lq=1
        smem_buff[threadIdx.x] = 0;
    }

    __syncthreads();


    // Ignore incomplete groups at the end of each thread block.
    if( GroupIndexWithinBlock < TGPB && GroupIndex < TGC){
        //Thread Index In Thread Group
        unsigned long TIITG = (threadIdx.x - (GroupIndexWithinBlock*TPG));
        float thread_sum = 0.0; // sum for current thread in the thread group

        //------------------------------------------------------------
        for(unsigned long iSPT=0;iSPT<SPT;iSPT++){
            unsigned long gidx =  TGO*GroupIndex + iSPT*dim3 + TIITG;
            if(gidx < _limit ){

                //if(blockIdx.x==35)
                //    printf("bid: %06d, tid: %06d, GroupIndex: %06ld, ThreadIndexInGroup: %06ld, gidx: %06ld\n",
                //        blockIdx.x, threadIdx.x, GroupIndex, TIITG, gidx);
                for (unsigned int outputElement = 0; outputElement < chOut; outputElement++) {
                    smem_buff[iSPT*dim3 + TIITG] = gInput_i[gidx] * gWeight_i[TIITG * dim3 + outputElement];
                }
            }
            //else
            //{
            //    printf("** gidx: %ld, GroupIndex: %ld, sum: %f\n",gidx,GroupIndex,thread_sum);
            //}
        }

        //------------------------------------------------------------
        // |---element0s---|-----element1s----|------....|
        smem_buff[TIITG*TGPB + GroupIndexWithinBlock] = thread_sum;
        //printf("TIITG: %ld, TGPB: %ld, GroupIndexLocal: %ld, smemIndx: %ld\n",
        //        TIITG,TGPB,GroupIndexWithinBlock,TIITG*TGPB + GroupIndexWithinBlock);
        //if(thread_sum!=10.0)
        //    printf("bid: %06d, tid: %06d, GroupIndex: %06ld, ThreadIndexInGroup: %06ld, thread_sum: %f, smem_buff[%06ld]: %f \n",
        //           blockIdx.x, threadIdx.x, GroupIndex, TIITG, thread_sum, TIITG*TGPB + GroupIndexWithinBlock , smem_buff[TIITG*TGPB + GroupIndexWithinBlock]);


    }


    __syncthreads();

    // parallel reduction of current block's shared memory buffer
    unsigned int thid = threadIdx.x;
    while(thid<TGPB){ // block stride loop

        for(unsigned long stride=TGPB/2; stride>0; stride >>= 1){
            if (thid < stride){
                for(int d3=0;d3<TPG;d3++){
                    smem_buff[d3*TGPB + (thid)] += smem_buff[d3*TGPB + (thid+stride)];
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

    if(threadIdx.x==0) {
        for (int d3 = 0; d3 < TPG; d3++) {
            atomicAdd(&gOutput_o[d3], smem_buff[d3 * TGPB]);
            //atomicAdd(&g_odata[d3], 1);
            //if(blockIdx.x==35)printf("bid: %06d, thid: %06d, smem_buff[%06ld]: %f, g_odata[%06d]: %f\n",blockIdx.x, threadIdx.x, d3 * TGPB, smem_buff[d3 * TGPB], d3, g_odata[d3]);
        }
    }


}


void conv2d_mlp_try02(
        float* gInput_i,
        float* gWeight_i,
        float* gOutput_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int dim3,
        unsigned int chOut)
{
    unsigned long BLOCK_SIZE = 1024;
    unsigned long block = BLOCK_SIZE;
    unsigned long SPT,TGC,TGO,TGPB, grid,TPG;

    //Dim3 slice per thread
    SPT = 1; //cte

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
    printf("KERNEL_SHAPE  : %ux%ux%ux%u\n", dim0,dim1,dim2,dim3);
    printf("KERNEL_GRID  : %ld\n", grid);
    printf("KERNEL_BLOCK : %ld\n", block);
    printf("KERNEL_SPT :   %ld\n", SPT);
    printf("KERNEL_TGO :   %ld\n", TGO);
    printf("KERNEL_TGC :   %ld\n", TGC);
    printf("KERNEL_TGPB :  %ld\n", TGPB);


    //CHECK(cudaMalloc((float**)&g_buffer, (TGC*TPG)*sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
    //CHECK(cudaMemset(g_buffer,0,(TGC*TPG)*sizeof(float)));

    CHECK(cudaMemset(g_odata,0,(dim3)*sizeof(float)));

    kernel_conv2d_mlp_try02 <<<grid, block, BLOCK_SIZE*sizeof(float)>>> (
                                                                                gInput_i, gOutput_o,
            dim0, dim1, dim2, dim3,
            TGC,
            TGPB,
            SPT,
            TGO
    );
}
*/