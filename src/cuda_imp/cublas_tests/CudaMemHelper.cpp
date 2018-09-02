//
// Created by saleh on 6/25/18.
//

#include "CudaMemHelper.h"


float** CudaMemHelper::ConvertDevice1D_to_Device2D(float* input_batched_1d,int batchsize,int perBatchLen){
    int B = batchsize;
    cudaError_t cuda_stat;


    float** input_ptr_device;
    float** input_ptr_host = new float*[B];

    //create device side pointer list
    cuda_stat = cudaMalloc((void**)&input_ptr_device, B*sizeof(float *));
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR1");

    for(int b=0;b<B;b++){
        input_ptr_host[b] = input_batched_1d + b*perBatchLen;
    }

    cudaMemcpy(input_ptr_device, input_ptr_host, B*sizeof(float *), cudaMemcpyHostToDevice);
    cudaCheckErrors("ERROR4");
    return input_ptr_device;
}


float** CudaMemHelper::ConvertHost1D_to_Device2D(float* input_batched_1d, float** outDevice1D, int batchsize,int perBatchLen){
    int B = batchsize;
    cudaError_t cuda_stat;


    cuda_stat = cudaMalloc((void**)outDevice1D, B*perBatchLen*sizeof(float));
    assert(cuda_stat==cudaSuccess);

    //copy cpu batched buffers to gpu batched buffers
    cuda_stat =cudaMemcpy(*outDevice1D,input_batched_1d,B*perBatchLen*sizeof(float),cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR3");

    return ConvertDevice1D_to_Device2D(*outDevice1D,B,perBatchLen);
}


float** CudaMemHelper::ConvertDevice1D_to_Device2D(float* input_batched_1d,std::vector<int> shape){
    if(shape.size()<2) throw std::exception();
    int B = shape[0];
    int Len = 1;
    for(int i = 1;i<shape.size();i++){
        Len = Len * shape[i];
    }
    CudaMemHelper::ConvertDevice1D_to_Device2D(input_batched_1d,B,Len);
}


float** CudaMemHelper::ConvertHost1D_to_Device2D(float* input_batched_1d, float** outDevice1D,std::vector<int> shape){
    if(shape.size()<2) throw std::exception();
    int B = shape[0];
    int Len = 1;
    for(int i = 1;i<shape.size();i++){
        Len = Len * shape[i];
    }
    CudaMemHelper::ConvertHost1D_to_Device2D(input_batched_1d,outDevice1D,B,Len);
}

