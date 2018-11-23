
// COPYRIGHT TO CHARLESQ34 @ GitHub : PointNet++
// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
kernel void kernel_group_point_gpu(
        global const float *  __restrict__ points,
        global const int * __restrict__ idx,
        global float * __restrict__ out,
        const int b,
        const int n,
        const int c,
        const int m,
        const int nsample) {
    //              get_group_id(uint dimindx)      and      blockIdx. [xyz]
    //              get_local_size(uint dimindx)    and      blockDim. [xyz]
    //              get_local_id(uint dimindx)      and      threadIdx.[xyz]
    //              get_num_groups(uint dimindx)    and      gridDim.  [xyz]
    int batch_index = get_group_id(0);
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = get_local_id(0);
    int stride = get_local_size(0);

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}
/*
void gather(
        const float* points,   // (b,n,c)
        const int *indices,    // (b,m,nsample)
        float* output,  // (b,m,nsample,c)
        int b,
        int n,
        int c,
        int m,
        int nsample){
    unsigned int blockSize = 256;

    //cudaDeviceSynchronize();
    kernel_group_point_gpu<<<b,blockSize>>>(b,n,c,m,nsample,points,indices,output);
    //cudaDeviceSynchronize();
}
 */