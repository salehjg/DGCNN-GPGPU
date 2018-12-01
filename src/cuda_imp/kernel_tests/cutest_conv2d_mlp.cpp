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
void conv2d_mlp_try02(
        const float* __restrict__ gInput_i,
        const float* __restrict__ gWeight_i,
        float* gOutput_o,
        unsigned int B,
        unsigned int N,
        unsigned int K,
        unsigned int D,
        unsigned int chOut);

extern
void mat_ops_try01(
        float* g_iA,
        float* g_iB,
        float* g_o,
        unsigned int rankA,
        unsigned int dim0A,
        unsigned int dim1A,
        unsigned int dim2A,
        unsigned int dim3A,
        unsigned int rankB,
        unsigned int dim0B,
        unsigned int dim1B,
        unsigned int dim2B,
        unsigned int dim3B,
        int operationMode);



float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

TensorF* getTestTensor(int mode){
    if(mode==0){
        vector<unsigned int> shape = {5,1024,20,128};
        TensorF *testTn = new TensorF(shape);
        unsigned long _len = testTn->getLength();
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = 1.0f + float_rand(0,0.50f);
        }
        return testTn;
    }
    if(mode==1){
        vector<unsigned int> shape = {1,1,128,256};
        TensorF *testTn = new TensorF(shape);
        unsigned long _len = testTn->getLength();
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = 2.0f + float_rand(0,0.50f);
        }
        return testTn;
    }
    if(mode==2){
        vector<unsigned int> shape = {256};
        TensorF *testTn = new TensorF(shape);
        unsigned long _len = testTn->getLength();
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = 0.0f + float_rand(0,1.0f);
        }
        return testTn;
    }
}

void printTensorContent(TensorF * tn){
    unsigned int dim0,dim1,dim2,dim3;
    unsigned int _dim0,_dim1,_dim2,_dim3;
    dim0 = tn->getShape()[0];
    dim1 = tn->getShape()[1];
    dim2 = tn->getShape()[2];
    dim3 = tn->getShape()[3];
    _dim0 = tn->getShape()[0];
    _dim1 = tn->getShape()[1];
    _dim2 = tn->getShape()[2];
    _dim3 = tn->getShape()[3];

    if(tn->getLength() > 1000){
        _dim0 = 1;
        _dim1 = 1;
        _dim2 = 1;
        //_dim3 = 1;
    }

    float val;
    unsigned long indx;
    for(int d0=0;d0<_dim0;d0++){
    for(int d1=0;d1<_dim1;d1++){
    for(int d2=0;d2<_dim2;d2++){
    for(int d3=0;d3<_dim3;d3++){
        indx = d0*dim1*dim2*dim3+
               d1*dim2*dim3+
               d2*dim3+
               d3;
        val = tn->_buff[indx];
        std::cout<<"("<<d0<<", "<<d1<<", "<<d2<<", "<<d3 <<", "<< ")" << ": "<< val<<endl;
    }}}}
}

void Test_try01()
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


    TensorF* inputTn_CPU    = getTestTensor(0);
    //TensorF* weightTn_CPU   = weightsLoader->AccessWeights(PLATFORMS::CPU,"dgcnn1.weights.npy");
    TensorF* weightTn_CPU   = getTestTensor(1);

    //TensorF* biasTn_CPU   = weightsLoader->AccessWeights(PLATFORMS::CPU,"empty_64bias.npy");
    TensorF* biasTn_CPU   = getTestTensor(2);
    TensorF* ouputTn_CPU;

    CudaTensorF* inputTn_GPU  = new CudaTensorF(); inputTn_GPU->InitWithHostData(inputTn_CPU->getShape(),inputTn_CPU->_buff);
    //TensorF* weightTn_GPU = weightsLoader->AccessWeights(PLATFORMS::GPU_CUDA,"dgcnn1.weights.npy");
    CudaTensorF* weightTn_GPU = new CudaTensorF(); weightTn_GPU->InitWithHostData(weightTn_CPU->getShape(),weightTn_CPU->_buff);
    CudaTensorF* biasTn_GPU = new CudaTensorF(); biasTn_GPU->InitWithHostData(biasTn_CPU->getShape(),biasTn_CPU->_buff);

    CudaTensorF* tmpBuff_GPU = new CudaTensorF({inputTn_CPU->getShape()[0],
                                                inputTn_CPU->getShape()[1],
                                                inputTn_CPU->getShape()[2],
                                                weightTn_CPU->getShape()[3] });

    CudaTensorF* ouputTn_GPU = new CudaTensorF({inputTn_CPU->getShape()[0],
                                                inputTn_CPU->getShape()[1],
                                                inputTn_CPU->getShape()[2],
                                                weightTn_CPU->getShape()[3] });


    ouputTn_CPU = cpuImplementation.Conv2D(workScheduler,inputTn_CPU,weightTn_CPU,biasTn_CPU);





    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    ///TODO: ADD KERNEL LAUNCH HERE
    conv2d_mlp_try02(
            inputTn_GPU->_buff,
            weightTn_GPU->_buff,
            tmpBuff_GPU->_buff,
            inputTn_GPU->getShape()[0],
            inputTn_GPU->getShape()[1],
            inputTn_GPU->getShape()[2],
            inputTn_GPU->getShape()[3],
            weightTn_GPU->getShape()[3]);

    mat_ops_try01(
            tmpBuff_GPU->_buff,
            biasTn_GPU->_buff,
            ouputTn_GPU->_buff,
            tmpBuff_GPU->getRank(),
            tmpBuff_GPU->getShape()[0],
            tmpBuff_GPU->getShape()[1],
            tmpBuff_GPU->getShape()[2],
            tmpBuff_GPU->getShape()[3],
            biasTn_GPU->getRank(),
            biasTn_GPU->getShape()[0],
            biasTn_GPU->getShape()[1],
            biasTn_GPU->getShape()[2],
            biasTn_GPU->getShape()[3],
            0);

    iElaps = seconds() - iStart;

    CHECK(cudaDeviceSynchronize());



    TensorF* outputTn_GPU = ((CudaTensorF*)ouputTn_GPU)->TransferToHost();

    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout    << "Matches: "
            << cpuImplementation.CompareTensors(
                    workScheduler,
                    ouputTn_CPU,
                    outputTn_GPU)
            << endl;
    cout << "\n\n\n\nContent of CPU result: "<< endl;
    printTensorContent(ouputTn_CPU);
    cout << "\n\n\n\nContent of GPU result: "<< endl;
    printTensorContent(outputTn_GPU);
    // reset device
    CHECK(cudaDeviceReset());
}

int main(){
    Test_try01();
}