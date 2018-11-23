
inline void atomicAdd_g_f(volatile __global float *addr, float val)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;
    current.f32    = *addr;
    do {
        expected.f32 = current.f32;
        next.f32     = expected.f32 + val;
        current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr,
                                       expected.u32, next.u32);
    } while( current.u32 != expected.u32 );
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
kernel void kernel_reduce_sum_4d_try04(
        global const float * __restrict__  g_idata, 
        global float * __restrict__  g_odata,
        local float *smem_buff,
        const int pow_y,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2,
        const int overaxis3,

        const unsigned long TGC,
        const unsigned long TGPB,
        const unsigned long SPT,
        const unsigned long TGO,
        const unsigned long worksize_x)
{ 
    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
        unsigned long TPG = dim3; //threads per group
        unsigned long GroupIndex = get_group_id(0) * TGPB + get_local_id(0) / TPG;
        unsigned long _limit = dim0*dim1*dim2*dim3;
        unsigned long GroupIndexWithinBlock = (GroupIndex - get_group_id(0) *TGPB);

        {
            // Fill unused shared mem with zeros! - Share memory is NOT initialized to zero
            // https://stackoverflow.com/questions/22172881/why-cuda-shared-memory-is-initialized-to-zero?noredirect=1&lq=1
            smem_buff[get_local_id(0)] = 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);


        // Ignore incomplete groups at the end of each thread block.
        if( GroupIndexWithinBlock < TGPB && GroupIndex < TGC){
            //Thread Index In Thread Group
            unsigned long TIITG = (get_local_id(0) - (GroupIndexWithinBlock*TPG));
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


        barrier(CLK_LOCAL_MEM_FENCE);

        // parallel reduction of current block's shared memory buffer
        unsigned int thid = get_local_id(0);
        while(thid<TGPB){ // block stride loop

            for(unsigned long stride=TGPB/2; stride>0; stride >>= 1){
                if (thid < stride){
                    for(int d3=0;d3<TPG;d3++){
                        smem_buff[d3*TGPB + (thid)] += smem_buff[d3*TGPB + (thid+stride)];
                        //printf("#P.Reduc.# bid: %06d, tid: %06d, stride: %06ld, d3: %06d, indx1: %06ld, indx2: %06ld, val1++: %f,\tval2: %f\n",
                        //        blockIdx.x,threadIdx.x,stride,d3, d3*TGPB + (thid) ,d3*TGPB + (thid+stride), smem_buff[d3*TGPB + (thid)] ,smem_buff[d3*TGPB + (thid+stride)]);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            // -------------------
            thid += get_local_size(0);
        }


        barrier(CLK_LOCAL_MEM_FENCE);

        if(get_local_id(0)==0) {
            for (int d3 = 0; d3 < TPG; d3++) {
                atomicAdd_g_f(&g_odata[d3], smem_buff[d3 * TGPB]);
                //atomicAdd(&g_odata[d3], 1);
                //if(blockIdx.x==35)printf("bid: %06d, thid: %06d, smem_buff[%06ld]: %f, g_odata[%06d]: %f\n",blockIdx.x, threadIdx.x, d3 * TGPB, smem_buff[d3 * TGPB], d3, g_odata[d3]);
            }
        }

    }

}

/*
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
 */