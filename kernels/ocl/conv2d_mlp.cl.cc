
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

kernel void kernel_conv2d_mlp_try01(
        global const float*  __restrict__ gInput_i,
        global const float*  __restrict__ gWeight_i,
        global float*  __restrict__ gOutput_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int chOut){
    // Dimension Example:
    //      input shape  = 5 x 1024 x 20 x 6
    //      weight shape = 1 x 1 x 6 x 64
    //      chOut = 64

    unsigned long tid = (get_group_id(0) * get_local_size(0) + get_local_id(0));
    //const unsigned int buff_len = dim3*chOut; //this should ALWAYS be an even number!
    unsigned int d3 = tid % dim3;
    unsigned int d2 = (tid % (dim2*dim3)) / (dim3);
    unsigned int d1 = (tid % (dim1*dim2*dim3)) / (dim2*dim3);
    unsigned int d0 = tid / (dim1*dim2*dim3);
    unsigned long idx1;

    float mulVal;
    for(int iCh = 0; iCh<chOut; iCh++){
        mulVal = gInput_i[tid] * gWeight_i[d3*chOut+iCh];
        idx1 = d0*dim1*dim2*chOut+ d1*dim2*chOut+ d2*chOut+ iCh ;
        atomicAdd_g_f(&gOutput_o[idx1], mulVal);
    }
}

kernel void kernel_conv2d_mlp_try02(
        global const float*  __restrict__ gInput_i,
        global const float*  __restrict__ gWeight_i,
        global float*  __restrict__ gOutput_o,
        local float* smem,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int dim3,
        const unsigned int chOut){
     /*********************************************************
     * 1. get_local_size(0) should be equal to 1024
     * 2. D should be less or equal to 1024.
     * 3. reduction algorithm will always reduce entire smem.
     *    so it should be padded with zeros
     * 4. each thread block grabs a single slice of last dim 'D'
     * 5.
     *********************************************************/

    // Dimension Example:
    //      input shape  = 5 x 1024 x 20 x 6
    //      weight shape = 1 x 1 x 6 x 64
    //      chOut = 64
    
    unsigned long tid = (get_group_id(0) * dim3 + get_local_id(0));

    //----------------------------------------------------------------
    // 1. Zero Padding smem:
    //int sidx=0;
    //while(sidx<1024){
    //    smem[sidx]=0;
    //    sidx+=get_local_size(0);
    //}

    //----------------------------------------------------------------
    // 2.
    unsigned int d3 = tid % dim3;
    unsigned int d2 = (tid % (dim2*dim3)) / (dim3);
    unsigned int d1 = (tid % (dim1*dim2*dim3)) / (dim2*dim3);
    unsigned int d0 = tid / (dim1*dim2*dim3);
    unsigned long idx1;
    float inputVal;

    if(d0 < dim0 && (get_local_id(0)<dim3) ){
        inputVal = gInput_i[tid];
    }else{
        inputVal = 0;
    }

    tid = get_local_id(0);
    for(int iCh = 0; iCh<chOut; iCh++){
        smem[tid] = inputVal * gWeight_i[iCh*dim3+d3] ;
        barrier(CLK_LOCAL_MEM_FENCE);
        //--------------------------------------------------------------
        // 3. Parallel reduction to get 1 element of output tensor
        // in-place reduction in shared memory
        if (get_local_size(0) >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_size(0) >= 512 && tid < 256) smem[tid] += smem[tid + 256];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_size(0) >= 256 && tid < 128) smem[tid] += smem[tid + 128];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_size(0) >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
        barrier(CLK_LOCAL_MEM_FENCE);

        // unrolling warp
        if (tid < 32) {
            local volatile float *vsmem = smem;
            vsmem[tid] += vsmem[tid + 32];
            vsmem[tid] += vsmem[tid + 16];
            vsmem[tid] += vsmem[tid +  8];
            vsmem[tid] += vsmem[tid +  4];
            vsmem[tid] += vsmem[tid +  2];
            vsmem[tid] += vsmem[tid +  1];
        }
        //--------------------------------------------------------------
        // 4. Assigning element to output tensor
        if(d0 < dim0 && (get_local_id(0)<dim3) && get_local_id(0)==0) {
            idx1 = d0*dim1*dim2*chOut+ d1*dim2*chOut+ d2*chOut+ iCh ;
            gOutput_o[idx1] = smem[0];
        }
    }
}

/*
void conv2d_mlp_try01(
        const float* gInput_i,
        const float* gWeight_i,
        float* gOutput_o,
        unsigned int B,
        unsigned int N,
        unsigned int K,
        unsigned int D,
        unsigned int chOut){
    assert(D<=1024); // this kernel cannot accept dim3>1024
    ///TODO: CHECK GRID SIZE DEVICE LIMITATION
    unsigned long blockSize = D;
    unsigned long gridSize = B*N*K;

    //printf("BlockSize: \t%lu\n",blockSize);
    //printf("GridSize: \t%lu\n",gridSize);
    //printf("B: %u\n",B);
    //printf("N: %u\n",N);
    //printf("K: %u\n",K);
    //printf("D: %u\n",D);
    //printf("C: %u\n",chOut);

    kernel_conv2d_mlp_try01 <<<gridSize, blockSize>>>(
            gInput_i, gWeight_i, gOutput_o, B, N, K, D, chOut);
}*/