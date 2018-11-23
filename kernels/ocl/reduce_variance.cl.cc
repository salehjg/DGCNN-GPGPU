
// g_odata = g_a[i] / const_k - g_b[i]*g_b[i]
kernel void kernel_multiply_const_sub_try01(
        global const float * __restrict__  g_a,
        global const float * __restrict__  g_b,
        const float const_k,
        global float * __restrict__  g_odata,
        const unsigned long dim) {
    unsigned long tidx = (get_group_id(0) * get_local_size(0) + get_local_id(0));
    if(tidx<dim){
        //printf("*** tidx: %ld, coef: %f \t\t g_a: %f \t\t g_b: %f\n",tidx,const_k, g_a[tidx], g_b[tidx]);
        g_odata[tidx] = g_a[tidx] / const_k - g_b[tidx]*g_b[tidx];
        //printf("*** tidx: %ld, g_o: %f\n",tidx, g_odata[tidx]);
    }
}


/*
void reduce_variance_4d_try01(
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
    cudaStream_t local_stream;
    cudaStreamCreate(&local_stream);
    float* g_tempbuff;
    float* g_variance_xi2;
    float* g_median;

    if( !(overaxis0 && overaxis1 && overaxis2 && !overaxis3) ) {
        printf("ERROR @reduce_sum_4d_try01 --NOT IMPLEMENTED\n"); return;
    }

    // 1. reduce_sum (MEDIAN )
    {
        unsigned long block = BLOCK_SIZE;
        unsigned long SPT, TGC, TGO, TGPB, grid, TPG;

        //Dim3 slice per thread
        SPT = 512; //cte

        //thread group offset
        TGO = dim3 * SPT;

        //thread group count
        TGC = (unsigned long) ((dim0 * dim1 * dim2 + (SPT - 1)) / SPT);

        //thread group per block
        TGPB = (unsigned long) ((BLOCK_SIZE) / dim3);
        if (TGPB % 2 && TGPB > 1) TGPB--;

        //grid size
        grid = (TGC + (TGPB - 1)) / TGPB;

        TPG = (unsigned long) dim3; //threads per group

        //printf("-------------------------------------------------------\n");
        //printf("KERNEL_GRID  : %ld\n", grid);
        //printf("KERNEL_BLOCK : %ld\n", block);
        //printf("KERNEL_SPT :   %ld\n", SPT);
        //printf("KERNEL_TGO :   %ld\n", TGO);
        //printf("KERNEL_TGC :   %ld\n", TGC);
        //printf("KERNEL_TGPB :  %ld\n", TGPB);

        CHECK(cudaMalloc((float **) &g_tempbuff, (dim3) * sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
        CHECK(cudaMalloc((float **) &g_median, (dim3) * sizeof(float))); // ThreadGroupCount * ThreadsPerGroup
        CHECK(cudaMalloc((float **) &g_variance_xi2, (dim3) * sizeof(float))); // ThreadGroupCount * ThreadsPerGroup

        CHECK(cudaMemset(g_tempbuff, 0, (dim3) * sizeof(float)));
        kernel_reduce_sum_4d_try04 << < grid, block, TGPB * TPG * sizeof(float), local_stream >> > (
                g_idata, nullptr, g_tempbuff,1,
                        dim0, dim1, dim2, dim3,
                        overaxis0, overaxis1, overaxis2, overaxis3,
                        TGC,
                        TGPB,
                        SPT,
                        TGO
        );

        CHECK(cudaMemset(g_variance_xi2, 0, (dim3) * sizeof(float)));
        kernel_reduce_sum_4d_try04 << < grid, block, TGPB * TPG * sizeof(float), local_stream >> > (
                g_idata, nullptr, g_variance_xi2,2,
                        dim0, dim1, dim2, dim3,
                        overaxis0, overaxis1, overaxis2, overaxis3,
                        TGC,
                        TGPB,
                        SPT,
                        TGO
        );
    }

    // 2. Multiplying (1/n) to each element of resulted tensor from step 1. (MEDIAN )
    // 3. Compute Variance
    {
        unsigned long len = dim3; //Axes combination is TTTF
        unsigned long block,grid;

        float coef = (dim0*dim1*dim2);
        //printf("WRAPPER: COEF: %f\n",coef);


        block = BLOCK_SIZE;
        grid = (len + block -1 )/(block);
        kernel_divide_by_const_try01 <<< grid, block, 0, local_stream >>> (
                g_tempbuff, g_median, len,coef
        );
        CHECK(cudaFree(g_tempbuff));



        kernel_multiply_const_sub_try01 <<< grid, block, 0, local_stream >>>(
                g_variance_xi2,
                g_median,
                coef,
                g_odata,
                len);
        CHECK(cudaFree(g_variance_xi2));
        CHECK(cudaFree(g_median));
    }



}
*/