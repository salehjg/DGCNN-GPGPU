//
// Created by saleh on 8/23/18.
//
#include <cuda_runtime.h>
#include "../inc/TensorF.h"
#include "../inc/cuda_imp/CudaTensorF.h"
#include <iostream>

using namespace std;

int main(){
    TensorF *tnc01, *tnc02;
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    cudaDeviceSynchronize();

    tnc01 = new CudaTensorF({10, 10, 10});
    cout << "Tensor tnc01 len: "<< tnc01->getLength() << endl;
    cout << "Tensor tnc01 lenBytes: "<< tnc01->getLengthBytes() << endl;
    TensorF* trnsfrd = ((CudaTensorF*)tnc01)->TransferToHost();
    cout << "Tensor trnsfrd len: "<< trnsfrd->getLength() << endl;

    tnc02 = new TensorF({10, 10, 10});
    cout << "Tensor tnc02 len: "<< tnc02->getLength() << endl;
    cout << "Tensor tnc02 lenBytes: "<< tnc02->getLengthBytes() << endl;



}