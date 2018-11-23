
kernel void kernel_split_3d_overdim2_float(
        global const float* __restrict__  g_i,
        global float* __restrict__  g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int new_dim2,
        const unsigned long worksize_x){

    unsigned int idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    //Padded because non-uniform work-group is not available in OCL1.2 (Trying to implement dynamic shared memory)
    if(idx < worksize_x ) {
        // threads are mapped into output tensor
        const unsigned long _len_output = dim0 * dim1 * new_dim2;
        unsigned int idx = (get_group_id(0) * get_local_size(0) + get_local_id(0));
        if (idx < _len_output) {
            unsigned int d0 = idx / (dim1 * new_dim2);
            unsigned int d1 = (idx % (dim1 * new_dim2)) / (new_dim2);
            unsigned int d2 = (idx % (new_dim2)) / 1;

            g_o[idx] = g_i[d0 * dim1 * dim2 + d1 * dim2 + d2];
        }
    }
}

/*
void split_3d_overdim2_float(
        const float* __restrict__  g_i,
        float* __restrict__  g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int new_dim2){
    unsigned int blockSize, gridSize;
    blockSize = 256;
    gridSize = ((dim0*dim1*new_dim2)+blockSize-1)/blockSize;
    //cudaDeviceSynchronize();
    kernel_split_3d_overdim2<float><<<gridSize,blockSize>>>(g_i,g_o,dim0,dim1,dim2,new_dim2);
    //cudaDeviceSynchronize();
}
*/