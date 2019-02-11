


kernel void kernel_reduce_sum_4d_try05(
        global const float * __restrict__  g_idata,
        global float * __restrict__  g_odata,
        local float *smem_buff,
        const int pow_y,
        const unsigned long slice_count,
        const unsigned long dim3,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2,
        const int overaxis3,

        const unsigned long TGC,
        const unsigned long TGPB,
        const unsigned long SPT,
        const unsigned long TGO)
{
    if (overaxis0 && overaxis1 && overaxis2 && !overaxis3) {
        unsigned long TPG = dim3; //threads per group
        unsigned long GroupIndex = get_group_id(0) * TGPB + get_local_id(0) / TPG;
        unsigned long _limit = slice_count*dim3;
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


        barrier(CLK_LOCAL_MEM_FENCE);

        // parallel reduction of current block's shared memory buffer
        unsigned int thid = get_local_id(0);
        while(thid<TGPB){ // block stride loop

            for(unsigned long stride=TGPB/2; stride>0; stride >>= 1){
                if (thid < stride){
                    for(int d3=0;d3<TPG;d3++){
                        smem_buff[d3*TGPB + (thid)] += smem_buff[d3*TGPB + (thid+stride)];
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
                g_odata[get_group_id(0) * dim3 + d3] = smem_buff[d3 * TGPB];
                //printf("** bid: %06d, tid: %06d, GroupIndex: %06ld, g_out_idx: %06ld, val: %f\n",
                //       blockIdx.x, threadIdx.x, GroupIndex, blockIdx.x * dim3 + d3,smem_buff[d3 * TGPB]);
            }
        }

    }

}

/*
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
        bool overaxis3,
        int pow_y)
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
                        (i==0)?pow_y:1,
                        (i==0) ? dim0*dim1*dim2 :grid_old,
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
*/