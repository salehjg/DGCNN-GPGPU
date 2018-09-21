// Mat Add
// Mat Add Tiled
// Mat Add Tiled Scalar

// Mat Sub
// Mat Sub Tiled
// Mat Sub Tiled Scalar

// Mat ElementMul
// Mat ElementMul Tiled
// Mat ElementMul Tiled Scalar


#include <stdio.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include "../../../../../../../../../opt/cuda/include/vector_types.h"

#define BLOCK_SIZE 1024


__global__ void kernel_mat_ops_try01(
        const float * __restrict__  g_iA,
        const float * __restrict__  g_iB,
        float * __restrict__  g_o,
        const unsigned int dimA0,
        const unsigned int dimA1,
        const unsigned int dimA2,
        const unsigned int dimA3,
        const unsigned int dimB0,
        const unsigned int dimB1,
        const unsigned int dimB2,
        const unsigned int dimB3,
        const unsigned int dimB0_IsNotZero, // 0 OR 1 ONLY
        const unsigned int dimB1_IsNotZero, // 0 OR 1 ONLY
        const unsigned int dimB2_IsNotZero, // 0 OR 1 ONLY
        const unsigned int dimB3_IsNotZero, // 0 OR 1 ONLY
        const int mode,
        const int EPT,
        const unsigned long limit){

    if(mode==0) {       //Add
        unsigned long indxS1, indxS2;
        unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long idx;
        for(unsigned long i=0;i<EPT;i++){
            idx = tid * EPT + i;
            unsigned long d0 = idx / (dimA1 * dimA2 * dimA3);
            unsigned long d1 = (idx % (dimA1 * dimA2 * dimA3)) / (dimA2 * dimA3);
            unsigned long d2 = (idx % (dimA2 * dimA3)) / dimA3;
            unsigned long d3 = (idx % (dimA3)) / 1;

            indxS1 = d0 * dimA1 * dimA2 * dimA3 +
                     d1 * dimA2 * dimA3 +
                     d2 * dimA3 +
                     d3;
            indxS2 = d0 * dimB1 * dimB2 * dimB3 * dimB0_IsNotZero +
                     d1 * dimB2 * dimB3 * dimB1_IsNotZero +
                     d2 * dimB3 * dimB2_IsNotZero +
                     d3 * dimB3_IsNotZero;
            if(indxS1<limit) {

                /*
                printf("grid: %ld, tid: %ld,"
                       " d0: %ld, d1: %ld, d2: %ld, d3: %ld, iEPT: %ld, indxS1: %ld, indxS2: %ld\n",
                       blockIdx.x, tid,
                       d0, d1, d2, d3, i, indxS1, indxS2);
                */


                g_o[indxS1] = g_iA[indxS1] + g_iB[indxS2];
            }
        }
    } else if(mode==1){ //Sub
        unsigned long indxS1, indxS2;
        unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long idx;
        for(unsigned long i=0;i<EPT;i++){
            idx = tid * EPT + i;
            unsigned long d0 = idx / (dimA1 * dimA2 * dimA3);
            unsigned long d1 = (idx % (dimA1 * dimA2 * dimA3)) / (dimA2 * dimA3);
            unsigned long d2 = (idx % (dimA2 * dimA3)) / dimA3;
            unsigned long d3 = (idx % (dimA3)) / 1;

            indxS1 = d0 * dimA1 * dimA2 * dimA3 +
                     d1 * dimA2 * dimA3 +
                     d2 * dimA3 +
                     d3;
            indxS2 = d0 * dimB1 * dimB2 * dimB3 * dimB0_IsNotZero +
                     d1 * dimB2 * dimB3 * dimB1_IsNotZero +
                     d2 * dimB3 * dimB2_IsNotZero +
                     d3 * dimB3_IsNotZero;
            if(indxS1<limit) {
                /*
                printf("grid: %ld, tid: %ld,"
                       " d0: %ld, d1: %ld, d2: %ld, d3: %ld, iEPT: %ld, indxS1: %ld, indxS2: %ld\n",
                       blockIdx.x, tid,
                       d0, d1, d2, d3, i, indxS1, indxS2);
                */

                g_o[indxS1] = g_iA[indxS1] - g_iB[indxS2];
            }
        }
    } else if(mode==2){ //Element Wise Multiplication
        unsigned long indxS1, indxS2;
        unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long idx;
        for(unsigned long i=0;i<EPT;i++){
            idx = tid * EPT + i;
            unsigned long d0 = idx / (dimA1 * dimA2 * dimA3);
            unsigned long d1 = (idx % (dimA1 * dimA2 * dimA3)) / (dimA2 * dimA3);
            unsigned long d2 = (idx % (dimA2 * dimA3)) / dimA3;
            unsigned long d3 = (idx % (dimA3)) / 1;

            indxS1 = d0 * dimA1 * dimA2 * dimA3 +
                     d1 * dimA2 * dimA3 +
                     d2 * dimA3 +
                     d3;
            indxS2 = d0 * dimB1 * dimB2 * dimB3 * dimB0_IsNotZero +
                     d1 * dimB2 * dimB3 * dimB1_IsNotZero +
                     d2 * dimB3 * dimB2_IsNotZero +
                     d3 * dimB3_IsNotZero;
            if(indxS1<limit) {
                /*
                printf("grid: %ld, tid: %ld,"
                       " d0: %ld, d1: %ld, d2: %ld, d3: %ld, iEPT: %ld, indxS1: %ld, indxS2: %ld\n",
                       blockIdx.x, tid,
                       d0, d1, d2, d3, i, indxS1, indxS2);
                */

                g_o[indxS1] = g_iA[indxS1] * g_iB[indxS2];
            }
        }
    } else if(mode==3){ //Element Wise Division
        unsigned long indxS1, indxS2;
        unsigned long tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned long idx;
        for(unsigned long i=0;i<EPT;i++){
            idx = tid * EPT + i;
            unsigned long d0 = idx / (dimA1 * dimA2 * dimA3);
            unsigned long d1 = (idx % (dimA1 * dimA2 * dimA3)) / (dimA2 * dimA3);
            unsigned long d2 = (idx % (dimA2 * dimA3)) / dimA3;
            unsigned long d3 = (idx % (dimA3)) / 1;

            indxS1 = d0 * dimA1 * dimA2 * dimA3 +
                     d1 * dimA2 * dimA3 +
                     d2 * dimA3 +
                     d3;
            indxS2 = d0 * dimB1 * dimB2 * dimB3 * dimB0_IsNotZero +
                     d1 * dimB2 * dimB3 * dimB1_IsNotZero +
                     d2 * dimB3 * dimB2_IsNotZero +
                     d3 * dimB3_IsNotZero;
            if(indxS1<limit) {
                /*
                printf("grid: %ld, tid: %ld,"
                       " d0: %ld, d1: %ld, d2: %ld, d3: %ld, iEPT: %ld, indxS1: %ld, indxS2: %ld\n",
                       blockIdx.x, tid,
                       d0, d1, d2, d3, i, indxS1, indxS2);
                */

                g_o[indxS1] = g_iA[indxS1] / g_iB[indxS2];
            }
        }
    }
}


void mat_ops_try01(
        float* g_iA,
        float* g_iB,
        float* g_o,
        const unsigned int rankA,
        const unsigned int dim0A,
        const unsigned int dim1A,
        const unsigned int dim2A,
        const unsigned int dim3A,

        const unsigned int rankB,
        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,

        const int operationMode){

    unsigned long block,grid;
    unsigned int dimB0_IsNotZero=1, dimB1_IsNotZero=1, dimB2_IsNotZero=1, dimB3_IsNotZero=1;
    if(rankA>4 || rankB>4){printf("Error@mat_ops_try01: BAD_RANK");return;}

    block = BLOCK_SIZE;
    grid = (dim0A*dim1A*dim2A*dim3A + block -1 )/(block);

    int tmp =15>>(4-rankB);
    dimB0_IsNotZero = (tmp >> 3) & 1;
    dimB1_IsNotZero = (tmp >> 2) & 1;
    dimB2_IsNotZero = (tmp >> 1) & 1;
    dimB3_IsNotZero = (tmp >> 0) & 1;

    if(rankB==1 && dim0B==0&&dim1B==0&&dim2B==0&&dim3B==1){//scalar value
        dimB3_IsNotZero=0; //force it to be zero, so in the kernel, indxS2 would be zero;
    }
    printf("TnA: dim0: %u, dim1: %u, dim2: %u, dim3: %u\n",dim0A,dim1A,dim2A,dim3A);
    printf("TnB: dim0: %u, dim1: %u, dim2: %u, dim3: %u\n",dim0B,dim1B,dim2B,dim3B);

    printf("block: %ld, grid: %ld\n",block,grid);
    printf("mode: %d, isNotZero0: %d, isNotZero1: %d, isNotZero2: %d, isNotZero3: %d\n",
           operationMode,dimB0_IsNotZero,dimB1_IsNotZero,dimB2_IsNotZero,dimB3_IsNotZero);

    kernel_mat_ops_try01<<<grid,block>>>(
            g_iA,
            g_iB,
            g_o,
            dim0A, dim1A, dim2A, dim3A,
            dim0B, dim1B, dim2B, dim3B,
            dimB0_IsNotZero,
            dimB1_IsNotZero,
            dimB2_IsNotZero,
            dimB3_IsNotZero,
            operationMode,
            1, // <----- EPT
            dim0A*dim1A*dim2A*dim3A);
}
