//
// Created by saleh on 6/25/18.
//

#ifndef DEEPPOINfloatV1_CUDAMEMHELPER_H
#define DEEPPOINfloatV1_CUDAMEMHELPER_H

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <cstring>
#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

#ifndef cudaCheckErrors
    #define cudaCheckErrors(msg) \
        do { \
            cudaError_t __err = cudaGetLastError(); \
            if (__err != cudaSuccess) { \
                fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                    msg, cudaGetErrorString(__err), \
                    __FILE__, __LINE__); \
                fprintf(stderr, "*** FAILED - ABORfloatING\n"); \
            } \
        } while (0)
#endif

class CudaMemHelper {
public:

    static float** ConvertDevice1D_to_Device2D(float* input_batched_1d,int batchsize,int perBatchLen);
    static float** ConvertDevice1D_to_Device2D(float* input_batched_1d,std::vector<int> shape);
    
    static float** ConvertHost1D_to_Device2D(float* input_batched_1d, float** outDevice1D,int batchsize,int perBatchLen);
    static float** ConvertHost1D_to_Device2D(float* input_batched_1d, float** outDevice1D,std::vector<int> shape);
};


#endif //DEEPPOINfloatV1_CUDAMEMHELPER_H
