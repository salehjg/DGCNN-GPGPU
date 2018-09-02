//
// Created by saleh on 6/25/18.
//

#include "TensorF_OLD.h"
#include "CudaMemHelper.h"
using namespace std;

TensorF::TensorF(){

}

void TensorF::InitWithDeviceData(float* deviceBuffer, vector<int> bufferShape){
    this->deviceBuffer = deviceBuffer;
    this->shape = bufferShape;
    this->rank = (int)bufferShape.size()-1;
    this->initialized=true;
    ///TODO: assign deviceBatchPtr using new deviceBuffer
}

void TensorF::InitWithDeviceData(float* deviceBuffer, float** deviceBatchPtr, vector<int> bufferShape){
    this->deviceBuffer = deviceBuffer;
    this->deviceBatchPtr = deviceBatchPtr;
    this->shape = bufferShape;
    this->rank = (int)bufferShape.size()-1;
    this->initialized=true;
}

void TensorF::InitWithHostData(float* hostBuffer, std::vector<int> bufferShape){
    this->deviceBatchPtr = CudaMemHelper::ConvertHost1D_to_Device2D(hostBuffer,&(this->deviceBuffer),bufferShape);
    this->shape = bufferShape;
    this->rank = (int)bufferShape.size()-1;
    this->initialized=true;
}

void TensorF::Init(std::vector<int> bufferShape){
    cudaError_t cuda_stat;

    this->shape = bufferShape;
    this->rank = (int)bufferShape.size()-1;
    this->initialized = true;
    int B = shape[0];
    int Len = 1;
    for(int i = 1;i<shape.size();i++){
        Len = Len * shape[i];
    }

    cuda_stat = cudaMalloc((void**)&this->deviceBuffer,Len);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR0");

    this->deviceBatchPtr = CudaMemHelper::ConvertDevice1D_to_Device2D(this->deviceBuffer,B,Len);
}

float* TensorF::TransferDeviceBufferToHost() {
    cudaError_t cuda_stat;
    if (!this->initialized) throw exception();
    int B = shape[0];
    int Len = 1;
    for (int i = 1; i < shape.size(); i++) {
        Len = Len * shape[i];
    }

    float *hostbuff = new float[B * Len];

    cuda_stat = cudaMemcpy(hostbuff, this->deviceBuffer, B * Len*sizeof(float), cudaMemcpyDeviceToHost);
    assert(cuda_stat == cudaSuccess);
    cudaCheckErrors("ERROR0");

    return hostbuff;
}


