//
// Created by saleh on 8/23/18.
//

#include "../../inc/cuda_imp/CudaTensorF.h"
#include "../../inc/cuda_imp/CudaTensorI.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <cassert>
#include <stdio.h>
#include <cstring>
#include <cstdio>
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

CudaTensorI::CudaTensorI(){
    initialized = false;
    platform = PLATFORMS::DEFAULT;
}

CudaTensorI::CudaTensorI(std::vector<unsigned int> shape){
    Init(shape);
}

CudaTensorI::CudaTensorI( std::vector<unsigned int> shape,int *buff){
    Init(shape,buff);
}


void CudaTensorI::Init(std::vector<unsigned int> shape) {
    cudaError_t cuda_stat;
    if(initialized){
        cuda_stat = cudaFree(_buff);
        assert(cuda_stat==cudaSuccess);
    }
    this->shape = shape;
    this->rank = (int)shape.size()-1;
    this->initialized = true;
    unsigned long len = getLengthBytes();
    platform = PLATFORMS::GPU_CUDA;

    cuda_stat = cudaMalloc((void**)&_buff,len);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("Init@CudaTensorI: ERR01");
}

void CudaTensorI::Init(std::vector<unsigned int> shape, int* buff){
    cudaError_t cuda_stat;
    if(initialized){
        cuda_stat = cudaFree(_buff);
        assert(cuda_stat==cudaSuccess);
    }
    this->shape = shape;
    this->rank = (int)shape.size()-1;
    this->initialized = true;
    this->_buff = buff;
    platform = PLATFORMS::GPU_CUDA;
    cudaCheckErrors("Init@CudaTensorI: ERR02");
}

void CudaTensorI::InitWithHostData(std::vector<unsigned int> shape, int *hostBuff) {
    cudaError_t cuda_stat;
    if(initialized){
        cuda_stat = cudaFree(_buff);
        assert(cuda_stat==cudaSuccess);
    }
    this->shape = shape;
    this->rank = (int)shape.size()-1;
    this->initialized = true;
    platform = PLATFORMS::GPU_CUDA;

    unsigned long len = getLengthBytes();

    cuda_stat = cudaMalloc((void**)this->_buff, len);
    assert(cuda_stat==cudaSuccess);
    cuda_stat =cudaMemcpy(this->_buff, hostBuff, len, cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);

    cudaCheckErrors("InitWithHostData@CudaTensorI: ERR03");
}

CudaTensorI::~CudaTensorI() {
    /* https://stackoverflow.com/questions/17923370/override-identifier-after-destructor-in-c11
     * Even though destructors are not inherited, a destructor in a derived class
     * overrides a base class destructor declared virtual; see 12.4 and 12.5. */
    cudaError_t cuda_stat;
    if(initialized){
        cuda_stat = cudaFree(_buff);
        assert(cuda_stat==cudaSuccess);
    }
    cudaCheckErrors("~CudaTensorI@CudaTensorI: ERR04");
}

TensorI* CudaTensorI::TransferToHost() {
    TensorI* rsltTn;
    cudaError_t cuda_stat;
    int* hostBuff = new int[getLength()];
    cuda_stat =cudaMemcpy(hostBuff, this->_buff, getLengthBytes(), cudaMemcpyDeviceToHost);
    assert(cuda_stat==cudaSuccess);
    rsltTn = new TensorI(getShape(),hostBuff);
    return rsltTn;
}

int** CudaTensorI::ConvertDevice1D_to_Device2D(int* input_batched_1d,int batchsize,int perBatchLen){
    int B = batchsize;
    cudaError_t cuda_stat;


    int** input_ptr_device;
    int** input_ptr_host = new int*[B];

    //create device side pointer list
    cuda_stat = cudaMalloc((void**)&input_ptr_device, B*sizeof(int *));
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR5");

    for(int b=0;b<B;b++){
        input_ptr_host[b] = input_batched_1d + b*perBatchLen;
    }

    cuda_stat = cudaMemcpy(input_ptr_device, input_ptr_host, B*sizeof(int *), cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR6");
    return input_ptr_device;
}

int** CudaTensorI::ConvertHost1D_to_Device2D(int* input_batched_1d, int** outDevice1D, int batchsize,int perBatchLen){
    int B = batchsize;
    cudaError_t cuda_stat;


    cuda_stat = cudaMalloc((void**)outDevice1D, B*perBatchLen*sizeof(int));
    assert(cuda_stat==cudaSuccess);

    //copy cpu batched buffers to gpu batched buffers
    cuda_stat = cudaMemcpy(*outDevice1D,input_batched_1d,B*perBatchLen*sizeof(int),cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR7");

    return ConvertDevice1D_to_Device2D(*outDevice1D,B,perBatchLen);
}

int** CudaTensorI::ConvertDevice1D_to_Device2D(int* input_batched_1d,std::vector<int> shape){
    if(shape.size()<2) throw std::exception();
    int B = shape[0];
    int Len = 1;
    for(int i = 1;i<shape.size();i++){
        Len = Len * shape[i];
    }
    ConvertDevice1D_to_Device2D(input_batched_1d,B,Len);
}

int** CudaTensorI::ConvertHost1D_to_Device2D(int* input_batched_1d, int** outDevice1D,std::vector<int> shape){
    if(shape.size()<2) throw std::exception();
    int B = shape[0];
    int Len = 1;
    for(int i = 1;i<shape.size();i++){
        Len = Len * shape[i];
    }
    ConvertHost1D_to_Device2D(input_batched_1d,outDevice1D,B,Len);
}
