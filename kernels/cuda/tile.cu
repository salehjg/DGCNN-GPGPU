
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include "../../../../../../../../../opt/cuda/include/vector_types.h"

#define BLOCK_SIZE 1024

// ONLY to tile BxNxD to BxNxcountxD (tileAxis=2)
// Input:  dim0, dim1, 1    , dim2
// Output: dim0, dim1, count, dim2
__global__ void kernel_tile_try01(
        const float * __restrict__  g_i,
        float * __restrict__  g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const int tileAxis,
        const int count,
        const int EPT){


    unsigned long tid = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned long idx = tid * EPT;
    unsigned long d0_o;
    unsigned long d1_o;
    unsigned long d2_o;
    unsigned long d3_o;
    unsigned long src_idx;

    unsigned long _limit = dim0*dim1*count*dim2;


    for (int i = 0; i < EPT && idx<_limit; i++) {
        d0_o =  idx / (dim1 * count * dim2);
        d1_o = (idx % (dim1 * count * dim2)) / (count * dim2);
        d2_o = (idx % (count * dim2) ) / (dim2);
        d3_o =  idx % (dim2);

        src_idx = d0_o * dim1*dim2 +
                  d1_o * dim2+
                  d3_o;

        g_o[idx] = g_i[src_idx];

        /*
        if (threadIdx.x < 20) printf("A*bid: %06d, thid: %06d, tid: %06lu, idx: %06lu, "
                                     "d0_o: %06lu, d1_o: %06lu, d2_o: %06lu, d3_o: %06lu, src_idx: %06lu\n",
                                     blockIdx.x, threadIdx.x, tid, idx,
                                     d0_o, d1_o, d2_o, d3_o, src_idx);
        */
        idx++;
    }

}


void tile_try01(
        float* g_i,
        float* g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int tileAxis,
        const unsigned int count
        )
{
    if (tileAxis==2){
        unsigned long block,grid, EPT;

        EPT = 8; //Elements(of output tensor) Per Thread

        block = BLOCK_SIZE;
        grid = (dim0*dim1*count*dim2 + block*EPT -1 )/(block*EPT);

        printf("Tensor: %d,%d,%d\n", dim0,dim1,dim2);
        printf("BLOCKSIZE: %lu, GRID SIZE: %lu, EPT: %lu\n",block,grid,EPT);

        kernel_tile_try01 <<< grid, block >>> (
                g_i, g_o,dim0,dim1,dim2,tileAxis,count,EPT);
    }
    else{
        printf("tile_try01: ERROR-NOTIMPLEMENTED\n");
    }
}



// ONLY to tile BxNxD to BxNxcountxD (tileAxis=2)
// Input:  dim0, dim1, 1    , dim2
// Output: dim0, dim1, count, dim2
__global__ void kernel_tile_try02(
        const float * __restrict__  g_i,
        float * __restrict__  g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const int tileAxis,
        const int count,
        const int EPT){


    unsigned long tid = (blockIdx.x * blockDim.x + threadIdx.x);
    unsigned long idx = tid * EPT;

    unsigned long d0_o;
    unsigned long d1_o;
    unsigned long d3_o;

    unsigned long dst_idx;
    float currentVal;

    unsigned long _limit = dim0*dim1*dim2;


    for (int i = 0; i < EPT && idx<_limit; i++) {

        currentVal = g_i[idx];

        for(unsigned long d2_o=0; d2_o < count; d2_o++){ //output: dim0,dim1,count,dim2

            d0_o =  idx / (dim1 * dim2);
            d1_o = ( idx % (dim1 * dim2) ) / (dim2);
            //d2_o = ( idx % (count * dim2) ) / (dim2);
            d3_o =  idx % (dim2);

            dst_idx = d0_o * dim1*count*dim2 +
                      d1_o * count*dim2 +
                      d2_o * dim2 +
                      d3_o ;

            g_o[dst_idx] = currentVal;
            /*
            if (threadIdx.x < 200) printf("A*bid: %06d, thid: %06d, tid: %06lu, idx: %06lu, "
                                         "d0_o: %06lu, d1_o: %06lu, d2_o: %06lu, d3_o: %06lu, dst_idx: %06lu\n",
                                         blockIdx.x, threadIdx.x, tid, idx,
                                         d0_o, d1_o, d2_o, d3_o, dst_idx);
                                         */
        }





        idx++;
    }

}


void tile_try02(
        float* g_i,
        float* g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int tileAxis,
        const unsigned int count
        )
{
    if (tileAxis==2){
        unsigned long block,grid, EPT;

        EPT = 1; //Elements(of output tensor) Per Thread

        block = BLOCK_SIZE;
        grid = (dim0*dim1*dim2 + block*EPT -1 )/(block*EPT);

        printf("Tensor: %d,%d,%d\n", dim0,dim1,dim2);
        printf("BLOCKSIZE: %lu, GRID SIZE: %lu, EPT: %lu\n",block,grid,EPT);

        kernel_tile_try02 <<< grid, block >>> (
                g_i, g_o,dim0,dim1,dim2,tileAxis,count,EPT);
    }
    else{
        printf("tile_try01: ERROR-NOTIMPLEMENTED\n");
    }
}
