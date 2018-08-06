//
// Created by saleh on 7/30/18.
//

#include <stdio.h>
#include <cuda_runtime_api.h>
#include "common.h"

#define BLOCK_SIZE 256


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

    printf("-------------------------------------------------------\n");
    printf("KERNEL_GRID  : %d\n", grid);
    printf("KERNEL_BLOCK : %d\n", block);
    printf("KERNEL_SPT : %d\n", SPT);
    printf("KERNEL_TGO : %d\n", TGO);
    printf("KERNEL_TGC : %d\n", TGC);
    printf("KERNEL_TGPB : %d\n", TGPB);

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