//
// Created by saleh on 8/23/18.
//

#include "../../inc/cuda_imp/CudaTensorF.h"
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

CudaTensorF::CudaTensorF(){
    initialized = false;
    platform = PLATFORMS::DEFAULT;
}

CudaTensorF::CudaTensorF(std::vector<unsigned int> shape){
    Init(shape);
}

CudaTensorF::CudaTensorF( std::vector<unsigned int> shape,float *buff){
    Init(shape,buff);
}


void CudaTensorF::Init(std::vector<unsigned int> shape) {
    cudaError_t cuda_stat;
    if(initialized){
        std::cout<<"--- CudaTensorF: buffer deleted.\n";
        cuda_stat = cudaFree(_buff);
        assert(cuda_stat==cudaSuccess);
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    unsigned long len = getLengthBytes();
    platform = PLATFORMS::GPU_CUDA;

    cuda_stat = cudaMalloc((void**)&_buff,len);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("Init@CudaTensorF: ERR01");
}

void CudaTensorF::Init(std::vector<unsigned int> shape, float* buff){
    cudaError_t cuda_stat;
    if(initialized){
        std::cout<<"--- CudaTensorF: buffer deleted.\n";
        cuda_stat = cudaFree(_buff);
        assert(cuda_stat==cudaSuccess);
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    this->_buff = buff;
    platform = PLATFORMS::GPU_CUDA;
    cudaCheckErrors("Init@CudaTensorF: ERR02");
}

void CudaTensorF::InitWithHostData(std::vector<unsigned int> shape, float *hostBuff) {
    cudaError_t cuda_stat;
    if(initialized){
        std::cout<<"--- CudaTensorF: buffer deleted.\n";
        cuda_stat = cudaFree(_buff);
        assert(cuda_stat==cudaSuccess);
    }
    this->shape = shape;
    this->rank = (int)shape.size();
    this->initialized = true;
    platform = PLATFORMS::GPU_CUDA;

    unsigned long len = getLengthBytes();

    cuda_stat = cudaMalloc((void**)&this->_buff, len);
    assert(cuda_stat==cudaSuccess);
    cuda_stat =cudaMemcpy(this->_buff, hostBuff, len, cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);

    cudaCheckErrors("InitWithHostData@CudaTensorF: ERR03");
}

CudaTensorF::~CudaTensorF() {
    /* https://stackoverflow.com/questions/17923370/override-identifier-after-destructor-in-c11
     * Even though destructors are not inherited, a destructor in a derived class
     * overrides a base class destructor declared virtual; see 12.4 and 12.5. */
    cudaError_t cuda_stat;
    if(initialized){
        std::cout<<"--- CudaTensorF: buffer deleted.\n";
        cuda_stat = cudaFree(_buff);
        assert(cuda_stat==cudaSuccess);
    }
    cudaCheckErrors("~CudaTensorF@CudaTensorF: ERR04");
}

TensorF* CudaTensorF::TransferToHost() {
    TensorF* rsltTn;
    cudaError_t cuda_stat;
    float* hostBuff = new float[getLength()];
    cuda_stat =cudaMemcpy(hostBuff, this->_buff, getLengthBytes(), cudaMemcpyDeviceToHost);
    assert(cuda_stat==cudaSuccess);
    rsltTn = new TensorF(getShape(),hostBuff);
    return rsltTn;
}

float** CudaTensorF::ConvertDevice1D_to_Device2D(float* input_batched_1d,int batchsize,int perBatchLen){
    int B = batchsize;
    cudaError_t cuda_stat;


    float** input_ptr_device;
    float** input_ptr_host = new float*[B];

    //create device side pointer list
    cuda_stat = cudaMalloc((void**)&input_ptr_device, B*sizeof(float *));
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR5");

    for(int b=0;b<B;b++){
        input_ptr_host[b] = input_batched_1d + b*perBatchLen;
    }

    cuda_stat = cudaMemcpy(input_ptr_device, input_ptr_host, B*sizeof(float *), cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR6");
    return input_ptr_device;
}

float** CudaTensorF::ConvertHost1D_to_Device2D(float* input_batched_1d, float** outDevice1D, int batchsize,int perBatchLen){
    int B = batchsize;
    cudaError_t cuda_stat;


    cuda_stat = cudaMalloc((void**)outDevice1D, B*perBatchLen*sizeof(float));
    assert(cuda_stat==cudaSuccess);

    //copy cpu batched buffers to gpu batched buffers
    cuda_stat = cudaMemcpy(*outDevice1D,input_batched_1d,B*perBatchLen*sizeof(float),cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR7");

    return ConvertDevice1D_to_Device2D(*outDevice1D,B,perBatchLen);
}

float** CudaTensorF::ConvertDevice1D_to_Device2D(float* input_batched_1d,std::vector<int> shape){
    if(shape.size()<2) throw std::exception();
    int B = shape[0];
    int Len = 1;
    for(int i = 1;i<shape.size();i++){
        Len = Len * shape[i];
    }
    return ConvertDevice1D_to_Device2D(input_batched_1d,B,Len);
}

float** CudaTensorF::ConvertHost1D_to_Device2D(float* input_batched_1d, float** outDevice1D,std::vector<int> shape){
    if(shape.size()<2) throw std::exception();
    int B = shape[0];
    int Len = 1;
    for(int i = 1;i<shape.size();i++){
        Len = Len * shape[i];
    }
    return ConvertHost1D_to_Device2D(input_batched_1d,outDevice1D,B,Len);
}
