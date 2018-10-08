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
void transpose(
        const float* g_i,
        float* g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2);



float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

TensorF* getTestTensor(int mode){
    if(mode==0){
        vector<unsigned int> shape = {5,32,32};
        TensorF *testTn = new TensorF(shape);
        unsigned long _len = testTn->getLength();
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = 1.0f + float_rand(0,0.50f);
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
    _dim0 = tn->getShape()[0];
    _dim1 = tn->getShape()[1];
    _dim2 = tn->getShape()[2];

    if(tn->getLength() > 1000){
        _dim0 = 1;
        _dim1 = 1;
    }

    float val;
    unsigned long indx;
    for(int d0=0;d0<_dim0;d0++){
        for(int d1=0;d1<_dim1;d1++){
            for(int d2=0;d2<_dim2;d2++){

                indx = d0*dim1*dim2+
                       d1*dim2+
                       d2;
                val = tn->_buff[indx];
                std::cout<<"("<<d0<<", "<<d1<<", "<<d2<<")" << ": "<< val<<endl;
            }}}
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



    TensorF* inputTn_CPU    = getTestTensor(0);
    TensorF* ouputTn_CPU;

    CudaTensorF* inputTn_GPU  = new CudaTensorF(); inputTn_GPU->InitWithHostData(inputTn_CPU->getShape(),inputTn_CPU->_buff);
    CudaTensorF* ouputTn_GPU = new CudaTensorF({inputTn_CPU->getShape()[0],
                                                inputTn_CPU->getShape()[2],
                                                inputTn_CPU->getShape()[1] });


    ouputTn_CPU = cpuImplementation.Transpose(workScheduler,inputTn_CPU);





    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    ///TODO: ADD KERNEL LAUNCH HERE
    transpose(
            inputTn_GPU->_buff,
            ouputTn_GPU->_buff,
            inputTn_GPU->getShape()[0],
            inputTn_GPU->getShape()[1],
            inputTn_GPU->getShape()[2]);

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