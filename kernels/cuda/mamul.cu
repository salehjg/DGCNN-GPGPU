// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <numeric>
#include <stdlib.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>


#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}



// C = AB
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y>
__global__ void kernel_batch_matmul(
        const float *  __restrict__ matA,
        const float *  __restrict__ matB,
        float *  __restrict__ matC,
        int dim0,
        int dim1A, int dim2A,
        int dim1B, int dim2B,
        int dim1C, int dim2C){
    extern __shared__ float smem[];

    const unsigned int len_subA = BLOCK_SIZE_Y * dim2A, len_subB = BLOCK_SIZE_X * dim1B; //len of sub matrices of A and B.
    const unsigned long
            len_A = dim0*dim1A*dim2A,
            len_B = dim0*dim1B*dim2B,
            len_C = dim0*dim1C*dim2C;
    const unsigned long
            len_A_signleBatch = dim1A*dim2A,
            len_B_signleBatch = dim1B*dim2B,
            len_C_signleBatch = dim1C*dim2C;
    const unsigned int BLOCKSIZE_P2 = BLOCK_SIZE_X*BLOCK_SIZE_Y;

    //smemA = smem + 0;
    //smemB = smem + len_subA;


    // Block index
    unsigned int bx = blockIdx.x; // mapped to the sub-matrices of output
    unsigned int by = blockIdx.y; // mapped to the sub-matrices of output
    unsigned int bz = blockIdx.z; // batch index

    // Thread index
    unsigned int  tx = threadIdx.x;
    unsigned int  ty = threadIdx.y;

    unsigned int  c_pos_x, c_pos_y;
    c_pos_x = bx*BLOCK_SIZE_X + tx;
    c_pos_y = by*BLOCK_SIZE_Y + ty;

    unsigned long gidx1,gidx2;
    unsigned int _d1,_d2;

    //printf("## bx:%u, by:%u, tx:%u, ty:%u, c_pos_x:%u, c_pos_y:%u\n",bx,by,tx,ty,c_pos_x,c_pos_y);


    unsigned long offsetA = (by * BLOCK_SIZE_Y) * dim2A;
    unsigned long offsetB = (bx * BLOCK_SIZE_X); //first row (d1=0)

    // Load sub matrices from global memory into shared memory

    unsigned long idxA, idxB;
    idxA = ty* BLOCK_SIZE_X + tx;
    idxB = ty* BLOCK_SIZE_X + tx;

    //printf("*** bx:%u, by:%u, tx:%u, ty:%u ,idxA:%ld, idxB:%ld\n",bx,by,tx,ty,idxA,idxB);

    while(idxA < len_subA){//Block-stride loop
        gidx1 = offsetA + idxA;
        if(idxA < len_subA && gidx1 < len_A) {
            smem[idxA] = matA[bz * len_A_signleBatch + gidx1];
            /*printf("bx:%u, by:%u, tx:%u, ty:%u ,idxA:%ld, gidx1:%ld\n",bx,by,tx,ty,idxA,gidx1);*/
        }else{
            smem[idxA] = 0;
        }
        idxA += BLOCKSIZE_P2;
    }

    ///TODO: It might be better to store transposed subMatB in shared memory to avoid shared memory read conflict.
    ///      But then we might get shared memory write conflict. (?)
    while(idxB < len_subB ){//Block-stride loop
        //gidx2 = offsetB + (bx*BLOCK_SIZE)*dim2B + (idxB % dim2B);
        _d2 = idxB%BLOCK_SIZE_X;
        _d1 = (idxB/BLOCK_SIZE_X);
        gidx2 = offsetB + _d1*dim2B + _d2;
        if(idxB < len_subB && _d1<dim1B && _d2<dim2B){
            smem[len_subA+idxB] = matB[bz * len_B_signleBatch +gidx2];
            /*printf("* bx:%u, by:%u ,tx:%u, ty:%u ,idxB:%ld, _d1:%d, _d2:%d, gidx2:%ld\n",bx,by,tx,ty,idxB,_d1,_d2,gidx2);*/
        }else{
            smem[len_subA+idxB] = 0;
        }
        idxB += BLOCKSIZE_P2;
    }





    __syncthreads();




    // Multiply and add each result to produce output element of current thread in the thread block.
    if(c_pos_x<dim2C && c_pos_y<dim1C){
        float output_element = 0.0f;

        //dim2A=dim1B is common equal dimension of 2 matrices  --- block-stride loop
        for (int k = 0; k < dim2A; k++) {
            output_element += smem[ty*dim2A+k] * smem[len_subA+ k*BLOCK_SIZE_X + tx];
            /*printf("###bz:%d, c_pos_x:%d, c_pos_y:%d, smem[%d]=%f, smem[%d]=%f\n",
                    bz,c_pos_x,c_pos_y,
                    ty*dim2A+k,smem[ty*dim2A+k],
                    len_subA+ k*BLOCK_SIZE+tx,smem[len_subA+ k*BLOCK_SIZE+tx]);*/
        }

        ///TODO: Check matC index to not to exceed the len of matC!
        matC[bz * len_C_signleBatch + c_pos_y*dim2C + c_pos_x] = output_element;

    }



}

void batch_matmul(
        const float * matA, //row-major device ptr (batch, hA, wA) == (dim0A,  dim1A  , *dim2A* )
        const float * matB, //row-major device ptr (batch, hB, wB) == (dim0B, *dim1B* ,  dim2B  )
        float * matC,		//row-major device ptr (batch, hB, wB) == (dim0B,  dim1A  ,  dim2B  )
        int dim0A, int dim1A, int dim2A,
        int dim0B, int dim1B, int dim2B){
    if(dim2A != dim1B){printf("ERR@batched_matmul: BAD SHAPE.\n"); return;}
    if(dim0B != dim0A){printf("ERR@batched_matmul: BAD BATCH SIZES.\n"); return;}

    const int BLOCK_DIM_X = 8;
    const int BLOCK_DIM_Y = 4;

    dim3 blocksize(BLOCK_DIM_X,BLOCK_DIM_Y,1);
    dim3 gridsize(0,0,0);
    gridsize.x = (dim2B + BLOCK_DIM_X-1)/BLOCK_DIM_X;
    gridsize.y = (dim1A + BLOCK_DIM_Y-1)/BLOCK_DIM_Y;
    gridsize.z = (dim0A);
    unsigned long sharedmemsize = (BLOCK_DIM_Y*dim2A + BLOCK_DIM_X* dim1B)*sizeof(float);
    //printf("@batched_matmul:\n");
    //printf("\tBLOCK:(%d, %d)\n",blocksize.x,blocksize.y);
    //printf("\t GRID:(%d, %d, %d)\n",gridsize.x,gridsize.y,gridsize.z);
    //printf("\t SHARED: %d Bytes\n",sharedmemsize);

    if(BLOCK_DIM_X==8 && BLOCK_DIM_Y==4){
        kernel_batch_matmul<8,4> <<<gridsize, blocksize, sharedmemsize>>>(
                matA,
                        matB,
                        matC,
                        dim0A,

                        dim1A, //hA
                        dim2A, //wA

                        dim1B, //hA
                        dim2B, //wA

                        dim1A,
                        dim2B);
        CudaCheckError();
    }else{
        printf("ERR@batched_matmul: UNDEFINED BLOCK_DIM.\n"); return;
    }

}