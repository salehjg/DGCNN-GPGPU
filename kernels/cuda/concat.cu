/*
**LA_Concat2: Rank: 4  concatDim: 3     dimA: 5,1024,20,3
 *                                      dimB: 5,1024,20,3   ;
 *
**LA_Concat2: Rank: 4  concatDim: 3     dimA: 5,1024,20,3
 *                                      dimB: 5,1024,20,3   ;
 *
**LA_Concat2: Rank: 4  concatDim: 3     dimA: 5,1024,20,64
 *                                      dimB: 5,1024,20,64  ;
 *
**LA_Concat2: Rank: 4  concatDim: 3     dimA: 5,1024,20,64
*                                       dimB: 5,1024,20,64  ;
 *
**LA_Concat2: Rank: 4  concatDim: 3     dimA: 5,1024,20,64
 *                                      dimB: 5,1024,20,64  ;
 *
**LA_Concat2: Rank: 4  concatDim: 3     dimA: 5,1024,1,64
 *                                      dimB: 5,1024,1,64   ;
 *
**LA_Concat2: Rank: 4  concatDim: 3     dimA: 5,1024,1,128
 *                                      dimB: 5,1024,1,64   ;
 *
**LA_Concat2: Rank: 4  concatDim: 3     dimA: 5,1024,1,192
 *                                      dimB: 5,1024,1,128  ;
*/

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include "../../../../../../../../../opt/cuda/include/vector_types.h"

#define BLOCK_SIZE 1024

///
/// \param g_idata
/// \param g_odata
/// \param dim0
/// \param dim1
/// \param dim2
/// \param dim3A
/// \param dim3B
/// \param concatAxis
/// \param EPT              Elements Per Thread
__global__ void kernel_concat_try01(
        const float * __restrict__  g_iA,
        const float * __restrict__  g_iB,
        float * __restrict__  g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3A,
        const unsigned int dim3B,
        const int concatAxis,
        const int EPT){

    if(concatAxis==3) {
        unsigned long dim3 = dim3A + dim3B;
        unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long d3 = (tid * EPT % dim3);
        unsigned long idx = tid * EPT;
        unsigned long _limit = dim0*dim1*dim2*dim3;


        for (int i = 0; i < EPT && idx<_limit; i++) {

            if (d3 < dim3A) {
                g_o[idx] = g_iA[tid];

                //if (threadIdx.x < 20) printf("A*bid: %06d, thid: %06d, tid: %06lu, d3: %06lu, idx: %06lu\n", blockIdx.x, threadIdx.x, tid,d3, idx);

            } else {
                g_o[idx] = g_iB[tid];

                //if (threadIdx.x < 20) printf("B*bid: %06d, thid: %06d, tid: %06lu, d3: %06lu, idx: %06lu\n", blockIdx.x, threadIdx.x, tid,d3, idx);
            }

            d3++;
            idx++;
        }
    }
}


void concat_try01(
        float* g_iA,
        float* g_iB,
        float* g_o,
        const unsigned int dim0A,
        const unsigned int dim1A,
        const unsigned int dim2A,
        const unsigned int dim3A,

        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const unsigned int concatAxis)
{
    if (concatAxis==3){
        unsigned long block,grid, EPT , dim3 = dim3A + dim3B;

        EPT = 2; //Elements(of output tensor) Per Thread

        block = BLOCK_SIZE;
        grid = (dim0A*dim1A*dim2A*dim3 + block*EPT -1 )/(block*EPT);

        printf("TensorA: %d,%d,%d,%d   TensorB: %d,%d,%d,%d\n", dim0A,dim1A,dim2A,dim3A,dim0B,dim1B,dim2B,dim3B);
        printf("BLOCKSIZE: %lu, GRID SIZE: %lu, EPT: %lu\n",block,grid,EPT);

        kernel_concat_try01 <<< grid, block >>> (
              g_iA,g_iB, g_o,dim0A,dim1A,dim2A,dim3A,dim3B,3,EPT);

    }
    else{
        printf("concat_try01: ERROR-NOTIMPLEMENTED\n");
    }
}
