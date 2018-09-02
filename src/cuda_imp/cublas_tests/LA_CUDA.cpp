//
// Created by saleh on 6/17/18.
//

#include "LA_CUDA.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>


LA_CUDA::LA_CUDA(){

}


/// \param handle
/// \param matA input left, column-major
/// \param matB input right, column-major
/// \param matR output will be set to this matrix as " alpha * matA x matB + beta*matR "
/// \param batch_size
/// \param m output's dim(matR)
/// \param n output's dim(matR)
/// \param k
/// \param alpha AxB's coef
/// \param beta matR's coef
/// \param transposeA if matA is row-major,this should be true, otherwise should be false
/// \param transposeB if matB is row-major,this should be true, otherwise should be false
/// \return
int LA_CUDA::MatMul_OLD(
        cublasHandle_t handle,
        float** matA, // "batch_size" pointers of mxk col-major matrix on device mem ptr.
        float** matB, // "batch_size" pointers of kxn col-major matrix on device mem ptr.
        float** matR, // "batch_size" pointers of mxn col-major matrix on device mem ptr.
        int batch_size,
        int m,
        int n,
        int k,
        float alpha=1.0f,
        float beta=0.0f,
        bool transposeA=false,
        bool transposeB=false)
{  //cublasSgemm : C = alpha * OP(A) * OP(B) + beta * C

    cublasStatus_t stat ;
    cudaError_t cuda_stat;
    int ldA,ldB,ldR;
    ldA = m; //leading dim of col-major 2d matrix which is of shape mxk
    ldB = k; //leading dim of col-major 2d matrix which is of shape kxn
    ldR = m; //leading dim of col-major 2d matrix which is of shape mxn

    //------------------------------------------------------------------------------------------------------------------
    /*******************************************************************************************************************
     * https://stackoverflow.com/questions/50972336/getting-pointers-to-specific-elements-of-a-1d-contiguous-array-on-the-device
     * https://stackoverflow.com/questions/16376804/clarification-of-the-leading-dimension-in-cublas-when-transposing/16377121
     *  The leading dimension always refers to the length of the first dimension of the array. The data order flags
     *  (normal, transpose, conjugate) only indicate to BLAS how the data within the array is stored. They have no
     *  effect on the array itself, which is always column major ordered and requires an LDA value for indexing in 2D.
     *  So whether the matrix data is stored in transposed form or not, an m x n array always has LDA>=m.
     *******************************************************************************************************************/
    stat =
    cublasSgemmBatched(
            handle,
            transposeA?CUBLAS_OP_T:CUBLAS_OP_N, // means that we are feeding column-major matrices into cuBLAS.
            transposeB?CUBLAS_OP_T:CUBLAS_OP_N,
            m,n,k,
            &alpha,
            (const float **)matA,
            ldA,
            (const float **)matB,
            ldB,
            &beta,
            matR,
            ldR,
            batch_size);
    assert(stat == CUBLAS_STATUS_SUCCESS);

    //------------------------------------------------------------------------------------------------------------------
}

int LA_CUDA::MatMul(
        cublasHandle_t handle,
        TensorF matA, // "batch_size" pointers of mxk col-major matrix on device mem ptr.
        TensorF matB, // "batch_size" pointers of kxn col-major matrix on device mem ptr.
        TensorF matR, // "batch_size" pointers of mxn col-major matrix on device mem ptr.
        float alpha=1.0f,
        float beta=0.0f,
        bool transposeA=false,
        bool transposeB=false)
{   // cublasSgemm : C = alpha * OP(A) * OP(B) + beta * C

    if( matA.shape[0] != matB.shape[0] ){
        // unequal batch sizes!
        throw exception();
    }
    if( matA.shape[2] != matB.shape[1] ){
        // unequal shared multiply dim!
        throw exception();
    }
    if(!matR.initialized){
        matR.Init({matA.shape[0], matA.shape[1], matB.shape[2]});
    }

    cublasStatus_t stat ;
    cudaError_t cuda_stat;
    int ldA,ldB,ldR;
    int m,n,k,batchsize;

    m = matA.shape[1];
    k = matA.shape[2];
    n = matB.shape[2];
    batchsize = matA.shape[0];

    ldA = m; //leading dim of col-major 2d matrix which is of shape mxk
    ldB = k; //leading dim of col-major 2d matrix which is of shape kxn
    ldR = m; //leading dim of col-major 2d matrix which is of shape mxn

    //------------------------------------------------------------------------------------------------------------------
    /*******************************************************************************************************************
     * https://stackoverflow.com/questions/50972336/getting-pointers-to-specific-elements-of-a-1d-contiguous-array-on-the-device
     * https://stackoverflow.com/questions/16376804/clarification-of-the-leading-dimension-in-cublas-when-transposing/16377121
     *  The leading dimension always refers to the length of the first dimension of the array. The data order flags
     *  (normal, transpose, conjugate) only indicate to BLAS how the data within the array is stored. They have no
     *  effect on the array itself, which is always column major ordered and requires an LDA value for indexing in 2D.
     *  So whether the matrix data is stored in transposed form or not, an m x n array always has LDA>=m.
     *******************************************************************************************************************/
    stat =
            cublasSgemmBatched(
                    handle,
                    transposeA?CUBLAS_OP_T:CUBLAS_OP_N, // means that we are feeding column-major matrices into cuBLAS.
                    transposeB?CUBLAS_OP_T:CUBLAS_OP_N,
                    m,n,k,
                    &alpha,
                    (const float **)matA.deviceBatchPtr,
                    ldA,
                    (const float **)matB.deviceBatchPtr,
                    ldB,
                    &beta,
                    matR.deviceBatchPtr,
                    ldR,
                    batchsize);
    assert(stat == CUBLAS_STATUS_SUCCESS);

    //------------------------------------------------------------------------------------------------------------------
}