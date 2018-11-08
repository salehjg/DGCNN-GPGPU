//
// Created by saleh on 7/16/18.
//

#define BDIMX 16
#define BDIMY 16
#define IPAD 0



__kernel void transposeBatch_try01 (
        const __global float * __restrict__ in,
        __global float * __restrict__ out,
        const int dim0,
        const int dim1,
        const int dim2)
{
    __local float tile[(BDIMX + IPAD) * BDIMY];

    // https://stackoverflow.com/questions/45203444/cuda-to-opencl-what-is-the-equivalent-of-blockidx-x-blockidx-ygriddim-x
    //unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    //unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //              get_group_id(uint dimindx)      and      blockIdx. [xyz]
    //              get_local_size(uint dimindx)    and      blockDim. [xyz]
    //              get_local_id(uint dimindx)      and      threadIdx.[xyz]
    //              get_num_groups(uint dimindx)    and      gridDim.  [xyz]


    unsigned int ix = get_local_size(0) * get_group_id(0) + get_local_id(0);
    unsigned int iy = get_local_size(1) * get_group_id(1) + get_local_id(1);
    unsigned long batch_offset = dim1*dim2;

    if(ix <dim2 && iy<dim1){
        unsigned int ti = iy * dim2 + ix; //input is of shape  (nx, ny)
        unsigned int to = ix * dim1 + iy; //output is of shape (ny, nx)

        // load data from global memory to shared memory
        //unsigned int local_idx = threadIdx.y * (blockDim.x+ IPAD) + threadIdx.x;
        unsigned int local_idx = get_local_id(1) * (get_local_size(0)+ IPAD) + get_local_id(0);
        for(int b=0;b<dim0;b++){
            tile[local_idx] = in[b*batch_offset + ti];
            barrier(CLK_LOCAL_MEM_FENCE);
            out[b*batch_offset + to] = tile[local_idx];
        }

        /*if(ti==4 || to==4 || ti==5){
        printf("bidx:%d, bidy:%d, thidx:%d, thidy:%d, ix:%d, iy:%d, ti:%d, to:%d\n",
            blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,ix,iy,ti,to);
        }*/
    }
}

// This is the wrapper function that will be used in cpp files!
//void transpose(
//        const float* g_i,
//        float* g_o,
//        unsigned int dim0,
//        unsigned int dim1,
//        unsigned int dim2){
//
//    dim3 blocksize  (BDIMX, BDIMY);
//    dim3 gridsize  ((dim2 + blocksize.x - 1) / blocksize.x, (dim1 + blocksize.y - 1) / blocksize.y);
//
//    transposeBatch_try01<<<gridsize, blocksize,  * sizeof(float)>>>(g_i, g_o, dim0,dim1,dim2);
//}
