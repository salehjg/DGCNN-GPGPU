//
// Created by saleh on 7/16/18.
//

#define BDIMX 16
#define BDIMY 16
#define IPAD 2


// COPYRIGHT "Professional CUDA C Programming BOOK - Chapter5"
// transposeSmemUnrollPadDyn refactored as transpose
__global__ void kernel_transpose(float *out, float *in, const int nx,const int ny)
{
    // static 1D shared memory with padding
    __shared__ float tile[BDIMY * (BDIMX * 2 + IPAD)];

    // coordinate in original matrix
    unsigned int ix = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    // linear global memory index for original matrix
    unsigned int ti = iy * nx + ix;

    // thread index in transposed block
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx / blockDim.y;
    unsigned int icol = bidx % blockDim.y;

    // coordinate in transposed matrix
    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = 2 * blockIdx.x * blockDim.x + irow;

    // linear global memory index for transposed matrix
    unsigned int to = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny)
    {
        // load two rows from global memory to shared memory
        unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) +
                               threadIdx.x;
        tile[row_idx]         = in[ti];
        tile[row_idx + BDIMX] = in[ti + BDIMX];

        // thread synchronization
        __syncthreads();

        // store two rows to global memory from two columns of shared memory
        unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
        out[to] = tile[col_idx];
        out[to + ny * BDIMX] = tile[col_idx + BDIMX];
    }
}


// COPYRIGHT "Professional CUDA C Programming BOOK - Chapter 3"
// transposeSmemUnrollPadDyn refactored as transpose
__global__ void kernel_transpose_naive(float *out, float *in, const int nx, const int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

// This is the wrapper function that will be used in cpp files!
void transpose(dim3 grid, dim3 block, float *out, float *in, const int nx, const int ny){
    kernel_transpose<<<grid, block>>>(out, in, nx, ny);
}