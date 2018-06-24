//
// Created by saleh on 6/17/18.
//

#ifndef DEEPPOINTV1_LA_CUDA_H
#define DEEPPOINTV1_LA_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>




class LA_CUDA {
public:
    LA_CUDA();
    int MatMul(
            cublasHandle_t handle,
            float** matA, // "batch_size" pointers of mxk col-major matrix on device mem ptr.
            float** matB, // "batch_size" pointers of kxn col-major matrix on device mem ptr.
            float** matR, // "batch_size" pointers of mxn col-major matrix on device mem ptr.
            int batch_size,
            int m,
            int n,
            int k,
            float alpha,
            float beta,
            bool transposeA,
            bool transposeB);

    void GPU_Multi(float **M, float **N, float **P, size_t pr, size_t pc, size_t mc, size_t num_mat, float alpha, float beta);

private:
    /*
    template<typename T> T** ConvertMatrixToBatches_ColMajor(
            T* matrix,
            int batch,
            int slice_len);
    template<typename T> T* ConvertBatchesToMatrix_ColMajor(
            T** batches,
            int batch,
            int slice_len);*/
};


#endif //DEEPPOINTV1_LA_CUDA_H
