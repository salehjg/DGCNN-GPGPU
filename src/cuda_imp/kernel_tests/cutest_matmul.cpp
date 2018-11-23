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
void batch_matmul(
        const float * matA, //row-major device ptr (batch, hA, wA) == (dim0A,  dim1A  , *dim2A* )
        const float * matB, //row-major device ptr (batch, hB, wB) == (dim0B, *dim1B* ,  dim2B  )
        float * matC,		//row-major device ptr (batch, hB, wB) == (dim0B,  dim1A  ,  dim2B  )
        int dim0A, int dim1A, int dim2A,
        int dim0B, int dim1B, int dim2B);



float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

TensorF* getTestTensor(int mode){
    if(mode==0){
        vector<unsigned int> shape = {10,50,20};
        TensorF *testTn = new TensorF(shape);
        unsigned long _len = testTn->getLength();
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = i %10 ;//+ float_rand(0,2.50f);
        }
        return testTn;
    }
    if(mode==1){
        vector<unsigned int> shape = {10,20,60};
        TensorF *testTn = new TensorF(shape);
        unsigned long _len = testTn->getLength();
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = i ;//+ float_rand(0,2.50f);
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



    TensorF* inputTnA_CPU    = getTestTensor(0);
    TensorF* inputTnB_CPU    = getTestTensor(1);
    TensorF* ouputTn_CPU;

    CudaTensorF* inputTnA_GPU  = new CudaTensorF();
    inputTnA_GPU->InitWithHostData(inputTnA_CPU->getShape(),inputTnA_CPU->_buff);

    CudaTensorF* inputTnB_GPU  = new CudaTensorF();
    inputTnB_GPU->InitWithHostData(inputTnB_CPU->getShape(),inputTnB_CPU->_buff);

    CudaTensorF* ouputTn_GPU = new CudaTensorF({inputTnA_CPU->getShape()[0],
                                                inputTnA_CPU->getShape()[1],
                                                inputTnB_CPU->getShape()[2] });

    cout<< "A Dim0:"<< inputTnA_GPU->getShape()[0]<<endl;
    cout<< "A Dim1:"<< inputTnA_GPU->getShape()[1]<<endl;
    cout<< "A Dim2:"<< inputTnA_GPU->getShape()[2]<<endl<<endl;

    cout<< "B Dim0:"<< inputTnB_GPU->getShape()[0]<<endl;
    cout<< "B Dim1:"<< inputTnB_GPU->getShape()[1]<<endl;
    cout<< "B Dim2:"<< inputTnB_GPU->getShape()[2]<<endl<<endl;

    cout<< "Out Dim0:"<< ouputTn_GPU->getShape()[0]<<endl;
    cout<< "Out Dim1:"<< ouputTn_GPU->getShape()[1]<<endl;
    cout<< "Out Dim2:"<< ouputTn_GPU->getShape()[2]<<endl<<endl;



    ouputTn_CPU = cpuImplementation.MatMul(workScheduler,inputTnA_CPU,inputTnB_CPU);
    cout << "CPU Result Shape: "        << "" <<
            ouputTn_CPU->getShape()[0]  << "x" <<
            ouputTn_CPU->getShape()[1]  << "x" <<
            ouputTn_CPU->getShape()[2]  << " " << endl;

    cout << "GPU Result Shape: "        << "" <<
         ouputTn_GPU->getShape()[0]  << "x" <<
         ouputTn_GPU->getShape()[1]  << "x" <<
         ouputTn_GPU->getShape()[2]  << "\n" << endl;


    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    ///TODO: ADD KERNEL LAUNCH HERE
    batch_matmul(
            inputTnA_GPU->_buff,
            inputTnB_GPU->_buff,
            ouputTn_GPU->_buff,

            inputTnA_GPU->getShape()[0],
            inputTnA_GPU->getShape()[1],
            inputTnA_GPU->getShape()[2],

            inputTnB_GPU->getShape()[0],
            inputTnB_GPU->getShape()[1],
            inputTnB_GPU->getShape()[2]
            );

    iElaps = seconds() - iStart;

    CHECK(cudaDeviceSynchronize());



    TensorF* outputTn_GPU = ((CudaTensorF*)ouputTn_GPU)->TransferToHost();


    cout << "\n\n\n\nContent of CPU result: "<< endl;
    printTensorContent(ouputTn_CPU);
    cout << "\n\n\n\nContent of GPU result: "<< endl;
    printTensorContent(outputTn_GPU);

    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout    << "Matches: "
            << cpuImplementation.CompareTensors(
                    workScheduler,
                    ouputTn_CPU,
                    outputTn_GPU)
            << endl;

    // reset device
    CHECK(cudaDeviceReset());
}

int main(){
    Test_try01();
}