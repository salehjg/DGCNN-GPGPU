
// Input:  dim0, dim1, dim2, dim3 (expanded dim)
// Input:  3D and 4D
// Output:
kernel void kernel_tile_try03(
        global const float * __restrict__  g_i,
        global float * __restrict__  g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const int rank,
        const int tileAxis,
        const int tileCount,
        const int EPT){
    if(rank==3){
        //==================================================================================================
        unsigned long tid = (get_group_id(0) * get_local_size(0) + get_local_id(0));
        unsigned long idx = tid * EPT;
        unsigned long d0_o;
        unsigned long dst_idx;
        float currentVal;
        unsigned long _limit = dim0*dim1*dim2;
        //----------------------------------------------------------
        if(tileAxis==1){
            unsigned long d2_o;
            for(unsigned long i = 0; i < EPT && idx<_limit; i++) {
                currentVal = g_i[idx];
                for(unsigned long d1_o=0; d1_o < tileCount; d1_o++){

                    d0_o =  idx / (dim2);
                    //d1_o = ...
                    d2_o =  idx % (dim2);

                    dst_idx = d0_o * tileCount*dim2 +
                              d1_o * dim2 +
                              d2_o ;

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
        } else if(tileAxis==2){
            unsigned long d1_o;
            for(unsigned long i = 0; i < EPT && idx<_limit; i++) {
                currentVal = g_i[idx];
                for(unsigned long d2_o=0; d2_o < tileCount; d2_o++){

                    d0_o =  idx / (dim1);
                    d1_o = (idx % dim1) / (dim2);
                    //d2_o = ...

                    dst_idx = d0_o * dim1*tileCount +
                              d1_o * tileCount +
                              d2_o ;

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
        //==================================================================================================
    }
    else if(rank==4){
        //==================================================================================================
        unsigned long tid = (get_group_id(0) * get_local_size(0) + get_local_id(0));
        unsigned long idx = tid * EPT;
        unsigned long d0_o,d1_o,d3_o;
        unsigned long dst_idx;
        float currentVal;
        unsigned long _limit = dim0*dim1*dim2*dim3;
        //----------------------------------------------------------
        for (unsigned long i = 0; i < EPT && idx<_limit; i++) {
            currentVal = g_i[idx];
            for(unsigned long d2_o=0; d2_o < tileCount; d2_o++){ //output: dim0,dim1,tileCount,dim2

                d0_o =  idx / (dim1 * dim3);
                d1_o = ( idx % (dim1 * dim3) ) / (dim3);
                //d2_o = ( idx % (tileCount * dim2) ) / (dim2);
                d3_o =  idx % (dim3);

                dst_idx = d0_o * dim1*tileCount*dim3 +
                          d1_o * tileCount*dim3 +
                          d2_o * dim3 +
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
        //==================================================================================================
    }

}



/*
// Inputs        TileAxis            Outputs
// d0xd1x1          2               d0xd1xK
// d0x1xd2          1               d0xKxd2
// d0xd1x1xd3       2               d0xd1xKxd3
void tile_try03(
        float* g_i,
        float* g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const int rank,
        const int tileAxis,
        const int tileCount)
{
    if(!((rank==3 && (tileAxis==1 || tileAxis==2)) || (rank==4 && tileAxis==2))){
        printf("tile_try03: ERROR-NOTIMPLEMENTED\n");
    }

    unsigned long block,grid, EPT;

    EPT = 2; //Elements(of output tensor) Per Thread

    block = BLOCK_SIZE;
    if(rank==3)
        grid = (dim0*dim1*dim2 + block*EPT -1 )/(block*EPT);
    else
        grid = (dim0*dim1*dim2*dim3 + block*EPT -1 )/(block*EPT);

    //printf("BLOCKSIZE: %lu, GRID SIZE: %lu, EPT: %lu\n",block,grid,EPT);

    kernel_tile_try03 <<< grid, block >>> (
            g_i, g_o,dim0,dim1,dim2,dim3,rank,tileAxis,tileCount,EPT);
}*/