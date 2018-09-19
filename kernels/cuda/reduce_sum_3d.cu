//
// Created by saleh on 7/23/18.
//
#include <stdio.h>
#include <cuda_runtime_api.h>
#include "../../../../../../../../../opt/cuda/include/vector_types.h"

#define BLOCK_SIZE 256

// reduce sum over specific axis
__global__ void kernel_reduce_sum_3d_try01(float *g_idata,
                                           float *g_odata,
                                           int dim0, int dim1, int dim2,
                                           int overaxis0, int overaxis1,int overaxis2) {
    // dim0, dim1, dim2 : TTT, TFF, FTF, FFT

    if(overaxis0==1 && overaxis1==0 && overaxis2==0){ // TFF
        /* Each thread handles 4 pairs of elements.
         * BlockDim is 1D
         * BlockDim should be dividable by 4
         *
         * */

        // static shared memory
        __shared__ float smem[BLOCK_SIZE*4];

        // set thread ID
        //unsigned int tid = threadIdx.x;
        unsigned int tid4 = 4 * threadIdx.x;

        // global index
        unsigned int idx4 = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

        // unrolling
        float tmpSum0 = 0;
        float tmpSum1 = 0;
        float tmpSum2 = 0;
        float tmpSum3 = 0;
        int offset = dim1*dim2;
        int i=0;


        for(i=0;i<dim0;i++){
            if(idx4 + 0 <= offset) tmpSum0 += g_idata[idx4 + i*offset + 0];
            if(idx4 + 1 <= offset) tmpSum1 += g_idata[idx4 + i*offset + 1];
            if(idx4 + 2 <= offset) tmpSum2 += g_idata[idx4 + i*offset + 2];
            if(idx4 + 3 <= offset) tmpSum3 += g_idata[idx4 + i*offset + 3];
        }


        if(idx4 + 0 <= offset) smem[tid4 + 0] = tmpSum0;
        if(idx4 + 1 <= offset) smem[tid4 + 1] = tmpSum1;
        if(idx4 + 2 <= offset) smem[tid4 + 2] = tmpSum2;
        if(idx4 + 3 <= offset) smem[tid4 + 3] = tmpSum3;
        __syncthreads();

        unsigned int oindx = idx4;
        if(idx4 + 0 <= offset) g_odata[oindx + 0] = smem[tid4 + 0];
        if(idx4 + 1 <= offset) g_odata[oindx + 1] = smem[tid4 + 1];
        if(idx4 + 2 <= offset) g_odata[oindx + 2] = smem[tid4 + 2];
        if(idx4 + 3 <= offset) g_odata[oindx + 3] = smem[tid4 + 3];

    }

}

void reduce_sum_3d_try01(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        int overaxis0,
        int overaxis1,
        int overaxis2)
{
    dim3 block (BLOCK_SIZE, 1);
    dim3 grid_overdim0  ((dim1*dim2 + block.x - 1) / block.x / 4, 1);
    dim3 grid_overdim1  ((dim0*dim2 + block.x - 1) / block.x / 4, 1);
    dim3 grid_overdim2  ((dim0*dim1 + block.x - 1) / block.x / 4, 1);

    dim3 grid = (overaxis0==1)?(grid_overdim0):(overaxis1==1?grid_overdim1:grid_overdim2);

    kernel_reduce_sum_3d_try01 <<< grid, block >>> (
            g_idata,
            g_odata,
            dim0, dim1, dim2,
            overaxis0, overaxis1,overaxis2);
}




__global__ void kernel_reduce_sum_3d_try02(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        int overaxis0,
        int overaxis1,
        int overaxis2)
{
    if (overaxis0 == 0 && overaxis1 == 1 && overaxis2 == 0) { // FTF
        /* Each thread handles 1 pairs of elements.
         * BlockDim is 1D
         * BlockDim should be dividable by 1
         *
         * */

        // static shared memory
        __shared__ float smem_store[BLOCK_SIZE];
        //__shared__ float smem_load[BLOCK_SIZE];

        // set thread ID
        //unsigned int tid = threadIdx.x;
        unsigned int tid =  threadIdx.x;

        // global index
        unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x);

        // unrolling
        float tmpSum0 = 0;
        unsigned int i = 0;
        unsigned int src_index ;
        unsigned int _limit = (unsigned int)(dim0 * dim1 * dim2);

        //Indices over output's matrix (NOT OVER INPUT) (OUTPUT'S CONSIDERED AS A ROW-MAJOR MATRIX)
        int thrd_d0 = (idx) / (1*dim2);
        int thrd_d2 = (idx - thrd_d0*dim2);

        //Only for debugging kernel
        //printf("tid: %03d \tidx: %03d\td0: %02d\td2: %02d\n",tid,idx,thrd_d0,thrd_d2);



        //Merging the thread's DIM1 element from all DIM1's elements of current DIM0.
        for (i = 0; i < dim1; i++) {
            src_index = thrd_d0*dim1*dim2 + i * dim2 + thrd_d2;
            printf("idx: %d : src_index: %d\n",idx,src_index);
            if(src_index < _limit)
                tmpSum0 += g_idata[src_index];
        }


        if (src_index + 0 < _limit) smem_store[tid + 0] = tmpSum0;
        __syncthreads();

        unsigned int oindx = (unsigned int)( thrd_d0*dim2 + thrd_d2 );
        if (src_index + 0 <= _limit) g_odata[oindx + 0] = smem_store[tid + 0];

    }
    else
    {
        if (overaxis0 == 0 && overaxis1 == 0 && overaxis2 == 1) { // FFT
            /* Each thread handles 1 pairs of elements.
             * BlockDim is 1D
             * BlockDim should be dividable by 1
             *
             * */

            // static shared memory
            __shared__ float smem_store[BLOCK_SIZE];

            // set thread ID
            //unsigned int tid = threadIdx.x;
            unsigned int tid =  threadIdx.x;

            // global index
            unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x);

            // unrolling
            float tmpSum0 = 0;
            unsigned int i = 0;
            unsigned int src_index ;
            unsigned int _limit = (unsigned int)(dim0 * dim1 * dim2);

            //Indices over output's matrix (NOT OVER INPUT) (OUTPUT'S CONSIDERED AS A ROW-MAJOR MATRIX)
            int thrd_d0 = (idx) / (dim1*1);
            int thrd_d1 = (idx - thrd_d0*dim1);

            //Only for debugging kernel
            printf("tid: %03d \tidx: %03d\td0: %02d\td1: %02d\n",tid,idx,thrd_d0,thrd_d1);


            //Merging the thread's DIM1 element from all DIM1's elements of current DIM0.
            for (i = 0; i < dim2; i++) {
                src_index = thrd_d0*dim1*dim2 + thrd_d1 * dim2 + i;
                if(idx<15) printf("idx: %d : src_index: %d\n",idx,src_index);
                if(src_index < _limit)
                    tmpSum0 += g_idata[src_index];
            }


            if (src_index + 0 < _limit) smem_store[tid + 0] = tmpSum0;
            __syncthreads();

            unsigned int oindx = (unsigned int)( thrd_d0*dim1 + thrd_d1 );
            if (src_index + 0 <= _limit) g_odata[oindx + 0] = smem_store[tid + 0];

        }
    }

}

void reduce_sum_3d_try02(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        int overaxis0,
        int overaxis1,
        int overaxis2)
{
    dim3 block (BLOCK_SIZE, 1);
    dim3 grid_overdim0  ((dim1*dim2 + block.x - 1) / block.x, 1);
    dim3 grid_overdim1  ((dim0*dim2 + block.x - 1) / block.x, 1);
    dim3 grid_overdim2  ((dim0*dim1 + block.x - 1) / block.x, 1);

    dim3 grid = (overaxis0==1)?(grid_overdim0):(overaxis1==1?grid_overdim1:grid_overdim2);

    kernel_reduce_sum_3d_try02 <<< grid, block >>> (
            g_idata,
            g_odata,
            dim0, dim1, dim2,
            overaxis0, overaxis1,overaxis2);
}



__global__ void kernel_reduce_sum_3d_try03(
        const float * __restrict__  g_idata,
        float * __restrict__  g_odata,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const bool overaxis0,
        const bool overaxis1,
        const bool overaxis2)
{
    // WANING : dim0 means dim2 and dim2 means dim0
    __shared__ float sm[BLOCK_SIZE];
    if (overaxis2 && !overaxis1 && !overaxis0)
    {
        // Case 1 - sums in X-direction
        // each threadblock is responsible for a separate row sum
        unsigned int bidx = blockIdx.x;
        unsigned int tidx = threadIdx.x;
        sm[threadIdx.x] = 0;
        while (tidx < dim0)
        {
            sm[threadIdx.x] += g_idata[bidx*dim0+tidx];
            /*if(bidx==21){
                //dbg
                printf("thid: %04d\tg_index_to_read:%d\n",threadIdx.x,bidx*dim0+tidx);
            }*/
            tidx += blockDim.x;
        } // block-stride loop

        __syncthreads();

        // parallel reduction
        for (int i = blockDim.x>>1; i > 0; i>>=1)
        {
            if (threadIdx.x < i) sm[threadIdx.x] += sm[threadIdx.x + i];
            __syncthreads();
        }

        if (!threadIdx.x) g_odata[bidx] = sm[0];
    }
    else if (!overaxis2 && overaxis1 && !overaxis0)
    {
        // Case 2 - sums in Y-direction
        // each thread is responsible for a separate Y-column sum
        unsigned int idx = threadIdx.x+blockDim.x*blockIdx.x;
        if (idx < (dim0*dim2))
        {
            unsigned int tidx = idx%dim0 + (idx/dim0)*(dim0*dim1); //indices over input tensor (begining of axis1 slices)

            float tsum = 0;

            for (unsigned int i = 0; i < dim1; i++)
            {
                printf("idx: %03d \t\t tidx: %03d\n",idx,tidx);
                tsum += g_idata[tidx];
                tidx += dim0;
            }

            g_odata[idx] = tsum;
        }
    }
    else if (!overaxis2 && !overaxis1 && overaxis0)
    {
        // Case 3 - sums in Z-direction
        // each thread is responsible for a separate Z-column sum

        unsigned int idx = threadIdx.x + blockDim.x*blockIdx.x;

        //printf("%d,%d,%d\n",dbg_blockid,dbg_thid,idx);


        if (idx < (dim0*dim1))
        {
            unsigned int tidx = idx;
            float tsum = 0;

            for (int i = 0; i < dim2; i++)
            {
                //printf("%d,%d,%d,%d,%d\n",dbg_blockid,dbg_thid,idx,tidx,i);
                //printf("idx:%02d, tidx:%02d, i=%02d\n",idx,tidx,i);
                tsum += g_idata[tidx];
                tidx += dim0*dim1;
            }

            g_odata[idx] = tsum;
        }
    }
    else {
        printf("reduce_sum: ERROR-NOTIMPLEMENTED\n");
    }




}



void reduce_sum_3d_try03(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2)
{
    dim3 block (BLOCK_SIZE, 1);
    dim3 grid_overdim0  ((dim1*dim2 + block.x - 1) / block.x, 1);
    dim3 grid_overdim1  ((dim0*dim2 + block.x - 1) / block.x, 1);
    dim3 grid_overdim2  (dim0*dim1 , 1);

    dim3 grid = overaxis0 ? (grid_overdim0) : (overaxis1 ? grid_overdim1 : grid_overdim2);

    //printf("-------------------------------------------------------\n");
    //printf("KERNEL_GRID  : %d\n", grid.x);
    //printf("KERNEL_BLOCK : %d\n", block.x);

    kernel_reduce_sum_3d_try03 <<<grid, block>>> (
            g_idata, g_odata,
            dim2, dim1, dim0,
            overaxis0, overaxis1,overaxis2);


}