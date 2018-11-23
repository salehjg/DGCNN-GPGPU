
// COPYRIGHT TO CHARLESQ34 @ GitHub : PointNet++
// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
kernel void kernel_selection_sort_gpu(
        global const float * __restrict__ dist,
        global int * __restrict__ outi,
        global float * __restrict__ out,
        const int b,
        const int n,
        const int m,
        const int k) {
    //              get_group_id(uint dimindx)      and      blockIdx. [xyz]
    //              get_local_size(uint dimindx)    and      blockDim. [xyz]
    //              get_local_id(uint dimindx)      and      threadIdx.[xyz]
    //              get_num_groups(uint dimindx)    and      gridDim.  [xyz]
    int batch_index = get_group_id(0);
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = get_local_id(0);
    int stride = get_local_size(0);

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    global float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s;
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}

/*
void top_k(
        const float* __restrict__  distance_matrix,   // (b,m,n)
        int * __restrict__ output_indices,            // (b,m,k)
        float* output_values,           // (b,m,k)
        int b,
        int n,
        int m,
        int k){
    unsigned int blockSize = 256;

    float* tmpVal;
    int* tmpIndices;
    CHECK(cudaMalloc((float**)&tmpVal  , (b*m*n)*sizeof(float)));
    CHECK(cudaMalloc((int**)&tmpIndices, (b*m*n)*sizeof(int)));



    //cudaDeviceSynchronize();

    kernel_selection_sort_gpu<<<b,blockSize>>>(b,n,m,k,distance_matrix,tmpIndices,tmpVal);
    split_3d_overdim2_float(tmpVal, output_values,b,m,n,k);     //split BxMxN into BxMxK (float)
    split_3d_overdim2_integer(tmpIndices, output_indices,b,m,n,k);  //split BxMxN into BxMxK (integer)

    //cudaDeviceSynchronize();



    CHECK(cudaFree(tmpVal));
    CHECK(cudaFree(tmpIndices));
}*/