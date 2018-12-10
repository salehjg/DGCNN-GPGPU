//
// Created by saleh on 7/30/18.
//

#include <stdio.h>
#include <cuda_runtime_api.h>
#include "common.h"

#define BLOCK_SIZE 1024


///
/// \param g_idata
/// \param g_buff           Temporary global memory buffer
/// \param g_odata
/// \param dim0
/// \param dim1
/// \param dim2
/// \param dim3
/// \param overaxis0
/// \param overaxis1
/// \param overaxis2
/// \param overaxis3
/// \param TGC              Thread group count
/// \param TGPB             Thread group per block
/// \param SPT              Dim3 slices per thread
/// \param TGO              Thread group offset
__global__ void kernel_reduce_sum_4d_try03(
        const float * __restrict__  g_idata,
        float * __restrict__  g_buff,
        float * __restrict__  g_odata,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const bool overaxis0,
        const bool overaxis1,
        const bool overaxis2,
        const bool overaxis3,

        const unsigned int TGC,
        const unsigned int TGPB,
        const unsigned int SPT,
        const unsigned int TGO)
{
    extern __shared__ float smem_buff[];
    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
        unsigned int TPG = dim3; //threads per group
        unsigned int GroupIndex = blockIdx.x * TGPB + threadIdx.x / TPG;
        unsigned int _limit = dim0*dim1*dim2*dim3;
        unsigned int GroupIndexWithinBlock = (GroupIndex - blockIdx.x *TGPB);
        // Ignore incomplete groups at the end of each thread block.
        if( GroupIndexWithinBlock < TGPB && GroupIndex < TGC){
            //Thread Index In Thread Group
            unsigned int TIITG = (threadIdx.x - (GroupIndexWithinBlock*TPG));
            float thread_sum = 0.0; // sum for current thread in the thread group

            //------------------------------------------------------------
            for(unsigned int iSPT=0;iSPT<SPT;iSPT++){
                unsigned int gidx =  TGO*GroupIndex + iSPT*dim3 + TIITG;
                if(gidx < _limit){
                    /*
                    if(GroupIndex>06)
                        printf("bid: %06d, tid: %06d, GroupIndex: %06d, ThreadIndexInGroup: %06d, gidx: %06d\n",
                            blockIdx.x, threadIdx.x, GroupIndex, TIITG, gidx);
                    */
                    thread_sum += g_idata[gidx];
                }
            }

            //------------------------------------------------------------
            //g_buff[GroupIndex*TPG + TIITG] = thread_sum;
            smem_buff[GroupIndexWithinBlock*TPG + TIITG] = thread_sum;

        }

        __syncthreads();

        if(threadIdx.x < dim3){
            g_odata[threadIdx.x] = 0;
        }

        __syncthreads();


        if(threadIdx.x == 0){

            for(unsigned int d3=0;d3<dim3;d3++) {
                float sum = 0;
                for (unsigned int local_grp = 0; local_grp < TGPB; local_grp++) {
                    sum += smem_buff[local_grp*TPG + d3];
                }
                //g_odata[d3] += sum;
                atomicAdd(&g_odata[d3], sum);
            }
        }

        /*
        if(threadIdx.x ==0 && blockIdx.x==0){
            for(int d3=0;d3<dim3;d3++){
                float sum=0;
                for(int grp=0;grp<TGC;grp++){
                    sum += g_buff[grp*TPG + d3];
                    printf("d3: %06d, grp: %06d, g_buff: %f, sum: %f\n",d3,grp,g_buff[grp*TPG + d3],sum);
                }
                g_odata[d3] = sum;
            }
        }
        */
    }
    //-------------------------------------------------------------------------------------------------------------
}

void reduce_sum_4d_try03(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        int dim3,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3)
{
    if( !(overaxis0 && overaxis1 && overaxis2 && !overaxis3) ) {
        printf("ERROR @reduce_sum_4d_try01 --NOT IMPLEMENTED\n"); return;
    }

    unsigned int block = BLOCK_SIZE;


    unsigned int SPT,TGC,TGO,TGPB, grid,TPG;

    //Dim3 slice per thread
    SPT = 512; //cte

    //thread group offset
    TGO = dim3 * SPT;

    //thread group count
    TGC = (unsigned int)((float)(dim0*dim1*dim2)/(float)SPT) + 1;

    //thread group per block
    TGPB = (unsigned int)(BLOCK_SIZE / dim3);

    //grid size
    grid = ( TGC+(TGPB-1) ) / TGPB;

    TPG = (unsigned int)dim3; //threads per group

    //printf("-------------------------------------------------------\n");
    //printf("KERNEL_GRID  : %d\n", grid);
    //printf("KERNEL_BLOCK : %d\n", block);
    //printf("KERNEL_SPT : %d\n", SPT);
    //printf("KERNEL_TGO : %d\n", TGO);
    //printf("KERNEL_TGC : %d\n", TGC);
    //printf("KERNEL_TGPB : %d\n", TGPB);

    float* g_buffer;
    CHECK(cudaMalloc((float**)&g_buffer, (TGC*TPG)*sizeof(float))); // ThreadGroupCount * ThreadsPerGroup

    kernel_reduce_sum_4d_try03 <<<grid, block, TGPB*TPG*sizeof(float)>>> (
            g_idata, g_buffer, g_odata,
            dim0, dim1, dim2, dim3,
            overaxis0, overaxis1, overaxis2, overaxis3,
            TGC,
            TGPB,
            SPT,
            TGO
            );
}


///
/// \param g_idata
/// \param g_buff           Temporary global memory buffer
/// \param g_odata
/// \param pow_y            The value that each element of input tensor will be powered to
/// \param dim0
/// \param dim1
/// \param dim2
/// \param dim3
/// \param overaxis0
/// \param overaxis1
/// \param overaxis2
/// \param overaxis3
/// \param TGC              Thread group count
/// \param TGPB             Thread group per block
/// \param SPT              Dim3 slices per thread
/// \param TGO              Thread group offset
__global__ void kernel_reduce_sum_4d_try04(
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
        const unsigned long TGO)
{
    extern __shared__ float smem_buff[];
    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
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
                    float pow_rslt=g_idata[gidx];
                    for(int ipwr=0;ipwr<pow_y-1;ipwr++){
                        pow_rslt = pow_rslt * pow_rslt;
                    }
                    thread_sum += pow_rslt;
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
                atomicAdd(&g_odata[d3], smem_buff[d3 * TGPB]);
                //atomicAdd(&g_odata[d3], 1);
                //if(blockIdx.x==35)printf("bid: %06d, thid: %06d, smem_buff[%06ld]: %f, g_odata[%06d]: %f\n",blockIdx.x, threadIdx.x, d3 * TGPB, smem_buff[d3 * TGPB], d3, g_odata[d3]);
            }
        }

    }

}

/* BUGS:
 *  --------------------------------------------------------------------------------------------------------------------
 *  Input: 16x1024x1024x16
 *  SPT=15
 *  BLOCKSIZE=1024
 *  --------------------------------------------------------------------------------------------------------------------
 *  Mismatch when TGPB isn't a pure multiply of 2.(being odd is a subset of this bug)
 *  --------------------------------------------------------------------------------------------------------------------
 *  Fixed       Mismatch when TGPB gets equal to zero(when dim3 > BLOCKSIZE)
 *  --------------------------------------------------------------------------------------------------------------------
 *  Fixed       Mismatch when input shape is greater than 8x8x8x8 in any dimension!
 *              BLOCKSIZE=1024
 *              SPT=512
 *              INPUT TENSOR VALUES : 0.0001f  ---------> but it works with sth like 1.0f
 *              Answer: https://developer.nvidia.com/sites/default/files/akamai/cuda/files/NVIDIA-CUDA-Floating-Point.pdf
 *  --------------------------------------------------------------------------------------------------------------------
 *  --------------------------------------------------------------------------------------------------------------------
 *  --------------------------------------------------------------------------------------------------------------------
 *  --------------------------------------------------------------------------------------------------------------------
 *  --------------------------------------------------------------------------------------------------------------------
 *  --------------------------------------------------------------------------------------------------------------------
 *  --------------------------------------------------------------------------------------------------------------------
 *
 *
 */

void reduce_sum_4d_try04(
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

    //printf("-------------------------------------------------------\n");
    //printf("KERNEL_SHAPE  : %ldx%ldx%ldx%ld\n", dim0,dim1,dim2,dim3);
    //printf("KERNEL_GRID  : %ld\n", grid);
    //printf("KERNEL_BLOCK : %ld\n", block);
    //printf("KERNEL_SPT :   %ld\n", SPT);
    //printf("KERNEL_TGO :   %ld\n", TGO);
    //printf("KERNEL_TGC :   %ld\n", TGC);
    //printf("KERNEL_TGPB :  %ld\n", TGPB);

    float* g_buffer;
    CHECK(cudaMalloc((float**)&g_buffer, (TGC*TPG)*sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
    CHECK(cudaMemset(g_buffer,0,(TGC*TPG)*sizeof(float)));
    CHECK(cudaMemset(g_odata,0,(dim3)*sizeof(float)));
    kernel_reduce_sum_4d_try04 <<<grid, block, TGPB*TPG*sizeof(float)>>> (
            g_idata, g_buffer, g_odata, 1,
                    dim0, dim1, dim2, dim3,
                    overaxis0, overaxis1, overaxis2, overaxis3,
                    TGC,
                    TGPB,
                    SPT,
                    TGO
    );
}


__global__ void kernel_reduce_sum_4d_try05(
        const float * __restrict__  g_idata,
        float * __restrict__  g_odata,
        const int pow_y,
        const unsigned long slice_count,
        const unsigned long dim3,
        const bool overaxis0,
        const bool overaxis1,
        const bool overaxis2,
        const bool overaxis3,

        const unsigned long TGC,
        const unsigned long TGPB,
        const unsigned long SPT,
        const unsigned long TGO)
{
    extern __shared__ float smem_buff[];
    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
        unsigned long TPG = dim3; //threads per group
        unsigned long GroupIndex = blockIdx.x * TGPB + threadIdx.x / TPG;
        unsigned long _limit = slice_count*dim3;
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
                    //    printf("bid: %06d, tid: %06d, GroupIndex: %06ld, ThreadIndexInGroup: %06ld, _limit:%06ld, gidx: %06ld\n",
                    //        blockIdx.x, threadIdx.x, GroupIndex, TIITG,_limit, gidx);
                    float pow_rslt=g_idata[gidx];
                    for(int ipwr=0;ipwr<pow_y-1;ipwr++){
                        pow_rslt = pow_rslt * pow_rslt;
                    }
                    thread_sum += pow_rslt;
                }
            }

            //------------------------------------------------------------
            // |---element0s---|-----element1s----|------....|
            smem_buff[TIITG*TGPB + GroupIndexWithinBlock] = thread_sum;
        }


        __syncthreads();

        // parallel reduction of current block's shared memory buffer
        unsigned int thid = threadIdx.x;
        while(thid<TGPB){ // block stride loop

            for(unsigned long stride=TGPB/2; stride>0; stride >>= 1){
                if (thid < stride){
                    for(int d3=0;d3<TPG;d3++){
                        smem_buff[d3*TGPB + (thid)] += smem_buff[d3*TGPB + (thid+stride)];
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
                g_odata[blockIdx.x * dim3 + d3] = smem_buff[d3 * TGPB];
                //printf("** bid: %06d, tid: %06d, GroupIndex: %06ld, g_out_idx: %06ld, val: %f\n",
                //       blockIdx.x, threadIdx.x, GroupIndex, blockIdx.x * dim3 + d3,smem_buff[d3 * TGPB]);
            }
        }

    }

}

int __Find_Kernel_Launches_Needed(int sliceCount, int SPT, int TGPB){
    int i=0, sliceLeft=sliceCount,p=sliceCount,q=SPT*TGPB;
    int LIMIT=50;
    for(i=0;i<LIMIT;i++){
        if(i==0){
            sliceLeft = ( p + (q-1) ) / q;
        }else{
            sliceLeft = ( sliceLeft + (q-1) ) / q;
        }
        if(sliceLeft==1){
            return i;
        }
    }
    return -1;
}

void reduce_sum_4d_try05(
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

    //printf("-------------------------------------------------------\n");
    //printf("KERNEL_SHAPE  : %ldx%ldx%ldx%ld\n", dim0,dim1,dim2,dim3);
    //printf("KERNEL_GRID  : %ld\n", grid);
    //printf("KERNEL_BLOCK : %ld\n", block);
    //printf("KERNEL_SPT :   %ld\n", SPT);
    //printf("KERNEL_TGO :   %ld\n", TGO);
    //printf("KERNEL_TGC :   %ld\n", TGC);
    //printf("KERNEL_TGPB :  %ld\n", TGPB);

    CHECK(cudaMemset(g_odata,0,(dim3)*sizeof(float)));


    float* g_buffer1,*g_buffer2;
    CHECK(cudaMalloc((float**)&g_buffer1, (grid*dim3)*sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
    CHECK(cudaMemset(g_buffer1,0,(grid*dim3)*sizeof(float)));
    CHECK(cudaMalloc((float**)&g_buffer2, (grid*dim3)*sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
    CHECK(cudaMemset(g_buffer2,0,(grid*dim3)*sizeof(float)));

    long iLast = __Find_Kernel_Launches_Needed(dim0*dim1*dim2,SPT,TGPB) ;
    int grid_old=0;
    for(long i=0;i<=(iLast);i++){
        //printf("i=%d of %d\n",i,iLast);
        //printf("launching kernel_reduce_sum_4d_try05...\n");
        kernel_reduce_sum_4d_try05 <<<grid, block, TGPB*TPG*sizeof(float)>>> (
                (i==0)? g_idata : (i%2)?g_buffer1:g_buffer2,
                //(i==0 && iLast!=0)? g_buffer1 : (i==iLast)? g_odata : g_buffer2,
                (i==iLast)?g_odata: (i%2)?g_buffer2:g_buffer1,
                1,
                (i==0) ? dim0*dim1*dim2 :grid_old*TGPB,
                dim3,
                overaxis0, overaxis1, overaxis2, overaxis3,
                TGC,
                TGPB,
                SPT,
                TGO
        );
        cudaDeviceSynchronize();
        TGC = (unsigned long)((TGC+(SPT-1))/SPT);
        grid_old = grid;
        grid = ( TGC+(TGPB-1) ) / TGPB;
        //printf("========================\n");
        //printf("KERNEL_TGC_NEXT   :   %ld\n", TGC);
        //printf("KERNEL_GRID_NEXT  :   %ld\n", grid);
    }

    CHECK(cudaFree(g_buffer1));
    CHECK(cudaFree(g_buffer2));
}