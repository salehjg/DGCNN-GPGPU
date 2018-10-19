//
// Created by saleh on 7/16/18.
//

#define BDIMX 16
#define BDIMY 16
#define IPAD 0



__global__ void transposeBatch_try01 (const float * __restrict__ in, float * __restrict__ out, const int dim0, const int dim1, const int dim2)
{
    // dynamic shared memory
    extern __shared__ float tile[];

    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned long batch_offset = dim1*dim2;

    if(ix <dim2 && iy<dim1){
        unsigned int ti = iy * dim2 + ix; //input is of shape  (nx, ny)
        unsigned int to = ix * dim1 + iy; //output is of shape (ny, nx)

        // load data from global memory to shared memory
        unsigned int local_idx = threadIdx.y * (blockDim.x+ IPAD) + threadIdx.x;
        for(int b=0;b<dim0;b++){
            tile[local_idx] = in[b*batch_offset + ti];
            __syncthreads();
            out[b*batch_offset + to] = tile[local_idx];
        }

        /*if(ti==4 || to==4 || ti==5){
        printf("bidx:%d, bidy:%d, thidx:%d, thidy:%d, ix:%d, iy:%d, ti:%d, to:%d\n",
            blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,ix,iy,ti,to);
        }*/
    }
}

// This is the wrapper function that will be used in cpp files!
void transpose(
        const float* g_i,
        float* g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2){

    dim3 blocksize  (BDIMX, BDIMY);
    dim3 gridsize  ((dim2 + blocksize.x - 1) / blocksize.x, (dim1 + blocksize.y - 1) / blocksize.y);

    transposeBatch_try01<<<gridsize, blocksize, (BDIMX + IPAD) * BDIMY * sizeof(float)>>>(g_i, g_o, dim0,dim1,dim2);
}
