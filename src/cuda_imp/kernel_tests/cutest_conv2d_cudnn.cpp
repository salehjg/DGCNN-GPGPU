//
// Created by saleh on 7/23/18.
//

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <CudaTensorF.h>
#include <TensorF.h>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include "cpu_imp/CpuImplementation.h"
#include "WorkScheduler.h"
#include "WeightsLoader.h"

using namespace std;

extern
void conv2d_cudnn_try01(
        float* ginput_i,
        float* gweight_i,
        float* goutput_o,
        unsigned int B,
        unsigned int N,
        unsigned int K,
        unsigned int D,
        unsigned int ChOut);

float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

TensorF* getTestTensor(int rank){
    assert(rank!=0);
    vector<unsigned int> shape = {5,1024,20,6};
    TensorF *testTn = new TensorF(shape);
    unsigned long _len = testTn->getLength();
    for (unsigned long i = 0; i < _len; i++) {
        testTn->_buff[i] = 1.5f ;//+ float_rand(0,0.10f);
    }
    return testTn;
}

void Test_01()
{
    WeightsLoader* weightsLoader = new WeightsLoader({PLATFORMS::CPU,PLATFORMS::GPU_CUDA});

    // set up device
    int dev = 0;
    double iStart, iElaps;
    CpuImplementation cpuImplementation;
    WorkScheduler workScheduler;
    cout << "-------------------------------------------------------" << endl;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    weightsLoader->LoadFromDisk("/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                                "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights/",

                                "/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                                "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights/filelist.txt");


    TensorF* inputTn_CPU    = getTestTensor(4);
    TensorF* weightTn_CPU   = weightsLoader->AccessWeights(PLATFORMS::CPU,"dgcnn1.weights.npy");
    TensorF* biasTn_CPU   = weightsLoader->AccessWeights(PLATFORMS::CPU,"empty_64bias.npy");
    TensorF* ouputTn_CPU;

    CudaTensorF* inputTn_GPU  =
            new CudaTensorF(); inputTn_GPU->InitWithHostData(inputTn_CPU->getShape(),inputTn_CPU->_buff);
    TensorF* weightTn_GPU = weightsLoader->AccessWeights(PLATFORMS::GPU_CUDA,"dgcnn1.weights.npy");
    CudaTensorF* ouputTn_GPU = new CudaTensorF({inputTn_CPU->getShape()[0],
                                                inputTn_CPU->getShape()[1],
                                                inputTn_CPU->getShape()[2],
                                                weightTn_CPU->getShape()[3] });


    ouputTn_CPU = cpuImplementation.Conv2D(workScheduler,inputTn_CPU,weightTn_CPU,biasTn_CPU);





    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    ///TODO: ADD KERNEL LAUNCH HERE
    conv2d_cudnn_try01(
            inputTn_GPU->_buff,
            weightTn_GPU->_buff,
            ouputTn_GPU->_buff,
            inputTn_GPU->getShape()[0],
            inputTn_GPU->getShape()[1],
            inputTn_GPU->getShape()[2],
            inputTn_GPU->getShape()[3],
            weightTn_GPU->getShape()[3]);
    
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;


    

    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout    << "Matches: "
            << cpuImplementation.CompareTensors(
                    workScheduler,
                    ouputTn_CPU,
                    ((CudaTensorF*)ouputTn_GPU)->TransferToHost()
                    )
            << endl;

    // reset device
    CHECK(cudaDeviceReset());
}
  
int main(){
    Test_01();
}