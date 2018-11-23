
kernel void kernel_reduce_sum_3d_try03(
        global const float * __restrict__  g_idata,
        global float * __restrict__  g_odata,
        local float * sm,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2,
        const unsigned long worksize_x)
{
    int gtid_x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    //local float sm[get_local_size(0)]; //BlockDIM

    //Padded because non-uniform work-group is not available in OCL1.2 (Trying to implement dynamic shared memory)
    if(gtid_x < worksize_x ) { ///TODO: Branch Divergance and big delays!! beacause of sync threads!

        // WANING : dim0 means dim2 and dim2 means dim0

        if (overaxis2 && !overaxis1 && !overaxis0) {
            // Case 1 - sums in X-direction
            // each threadblock is responsible for a separate row sum
            unsigned int bidx = get_group_id(0);
            unsigned int tidx = get_local_id(0);
            sm[get_local_id(0)] = 0;
            while (tidx < dim0) {
                sm[get_local_id(0)] += g_idata[bidx * dim0 + tidx];
                /*if(bidx==21){
                    //dbg
                    printf("thid: %04d\tg_index_to_read:%d\n",get_local_id(0),bidx*dim0+tidx);
                }*/
                tidx += get_local_size(0);
            } // block-stride loop

            barrier(CLK_LOCAL_MEM_FENCE);

            // parallel reduction
            for (int i = get_local_size(0) >> 1; i > 0; i >>= 1) {
                if (get_local_id(0) < i) sm[get_local_id(0)] += sm[get_local_id(0) + i];
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if (!get_local_id(0)) g_odata[bidx] = sm[0];
        } else if (!overaxis2 && overaxis1 && !overaxis0) {
            // Case 2 - sums in Y-direction
            // each thread is responsible for a separate Y-column sum
            unsigned int idx = get_local_id(0) + get_local_size(0) * get_group_id(0);
            if (idx < (dim0 * dim2)) {
                unsigned int tidx = idx % dim0 +
                                    (idx / dim0) * (dim0 * dim1); //indices over input tensor (begining of axis1 slices)

                float tsum = 0;

                for (unsigned int i = 0; i < dim1; i++) {
                    //printf("idx: %03d \t\t tidx: %03d\n", idx, tidx);
                    tsum += g_idata[tidx];
                    tidx += dim0;
                }

                g_odata[idx] = tsum;
            }
        } else if (!overaxis2 && !overaxis1 && overaxis0) {
            // Case 3 - sums in Z-direction
            // each thread is responsible for a separate Z-column sum

            unsigned int idx = get_local_id(0) + get_local_size(0) * get_group_id(0);

            //printf("%d,%d,%d\n",dbg_blockid,dbg_thid,idx);


            if (idx < (dim0 * dim1)) {
                unsigned int tidx = idx;
                float tsum = 0;

                for (int i = 0; i < dim2; i++) {
                    //printf("%d,%d,%d,%d,%d\n",dbg_blockid,dbg_thid,idx,tidx,i);
                    //printf("idx:%02d, tidx:%02d, i=%02d\n",idx,tidx,i);
                    tsum += g_idata[tidx];
                    tidx += dim0 * dim1;
                }

                g_odata[idx] = tsum;
            }
        } else {
            printf("reduce_sum: ERROR-NOTIMPLEMENTED\n");
        }

    }


}


/*
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
 */