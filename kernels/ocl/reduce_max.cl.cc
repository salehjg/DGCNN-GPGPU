
// Inherited from reduce_sum_3d_try03 @ reduce_sum_3d.cu
kernel void kernel_reduce_max_try01(
        global const float * __restrict__  g_idata,
        global float * __restrict__  g_odata,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const int overaxis0,
        const int overaxis1,
        const int overaxis2)
{
    // WANING : dim0 means dim2 and dim2 means dim0
    if (!overaxis2 && overaxis1 && !overaxis0)
    {
        // Case 2 - sums in Y-direction
        // each thread is responsible for a separate Y-column sum
        unsigned int idx = (get_group_id(0) * get_local_size(0) + get_local_id(0));
        if (idx < (dim0*dim2))
        {
            unsigned int tidx = idx%dim0 + (idx/dim0)*(dim0*dim1); //indices over input tensor (begining of axis1 slices)

            float tMax = FLT_MIN;
            float gval ;

            for (unsigned int i = 0; i < dim1; i++)
            {
                //printf("idx: %03d \t\t tidx: %03d\n",idx,tidx);
                gval = g_idata[tidx];
                if(gval > tMax)tMax = gval;
                tidx += dim0;
            }

            g_odata[idx] = tMax;
        }
    }
}

/*
void reduce_max_4d_try01(
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
    if(!(!overaxis0 && !overaxis1 && overaxis2 && !overaxis3)){
        printf("**reduce_max_4d_try01: unimplemented axes combination.\n");
        return;
    }

    int kDim0,kDim1,kDim2, kGrid;
    kDim0 = dim0*dim1;
    kDim1 = dim2;
    kDim2 = dim3;
    kGrid = (kDim0*kDim2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //printf("-------------------------------------------------------\n");
    //printf("KERNEL_GRID  : %d\n", kGrid);
    //printf("KERNEL_BLOCK : %d\n", BLOCK_SIZE);

    kernel_reduce_max_try01 <<<kGrid, BLOCK_SIZE>>> (
            g_idata, g_odata,
            kDim2, kDim1, kDim0,
            overaxis0 && overaxis1, overaxis2,overaxis3);
}

void reduce_max_3d_try01(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2)
{
    if(!(!overaxis0 && overaxis1 && !overaxis2 )){
        printf("**reduce_max_4d_try01: unimplemented axes combination.\n");
        return;
    }

    int kDim0,kDim1,kDim2, kGrid;
    kDim0 = dim0;
    kDim1 = dim1;
    kDim2 = dim2;
    kGrid = (kDim0*kDim2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    //printf("-------------------------------------------------------\n");
    //printf("KERNEL_GRID  : %d\n", kGrid);
    //printf("KERNEL_BLOCK : %d\n", BLOCK_SIZE);

    kernel_reduce_max_try01 <<<kGrid, BLOCK_SIZE>>> (
            g_idata, g_odata,
            kDim2, kDim1, kDim0,
            overaxis0 ,overaxis1, overaxis2);
}
 */