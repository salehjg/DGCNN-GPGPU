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

using namespace std;

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

TensorF* getTestTensor(int rank){
    assert(rank!=0);
    vector<unsigned int> shape;
    for (int i = 0; i < rank; i++) {
        shape.push_back(3);
    }
    TensorF *testTn = new TensorF(shape);
    unsigned long _len = testTn->getLength();
    for (unsigned long i = 0; i < _len; i++) {
        testTn->_buff[i] = 1.5f ;//+ float_rand(0,0.10f);
    }
    return testTn;
}

void Test_01(int mode, int rankA, int rankB, bool isB_scalar)
{
    // set up device
    int dev = 0;
    double iStart, iElaps;
    CpuImplementation cpuImplementation;
    WorkScheduler workScheduler;
    cout << "-------------------------------------------------------" << endl;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));



    TensorF* TnA = getTestTensor(rankA);
    TensorF* TnB; 
    TensorF* TnO; 
    if(!isB_scalar){
        TnB = getTestTensor(rankB);}
    else{
        float *scalarVal = new float[1];
        scalarVal[0] = 10.0f;
        TnB = new TensorF({1},scalarVal);
    }
    
    
    switch(mode){
        case 0:{ //Add
            if(rankA==rankB){
                TnO = cpuImplementation.MatAdd(workScheduler,TnA,TnB);
            }else{
                if(isB_scalar){
                    TnO = cpuImplementation.MatAddTiled(workScheduler,TnA,10.0f);
                }else{
                    TnO = cpuImplementation.MatAddTiled(workScheduler,TnA,TnB);
                }
            }

            break;
        }
        case 1:{ //Sub
            if(rankA==rankB){
                TnO = cpuImplementation.MatSub(workScheduler,TnA,TnB);
            }else {
                if (isB_scalar) {
                    TnO = cpuImplementation.MatSubTiled(workScheduler, TnA, 10.0f);
                } else {
                    TnO = cpuImplementation.MatSubTiled(workScheduler, TnA, TnB);
                }
            }
            break;
        }
        case 2:{
            if(rankA==rankB){
                TnO = cpuImplementation.Multiply(workScheduler,TnA,TnB);
            }else {
                if (isB_scalar) {
                    TnO = cpuImplementation.MatMul(workScheduler, TnA, 10.0f);
                } else {
                    TnO = cpuImplementation.MultiplyTiled(workScheduler, TnA, TnB);
                }
            }
            break;
        }
        case 3:{
            if(rankA==rankB){
                TnO = cpuImplementation.Divide(workScheduler,TnA,TnB);
            }else{
                if(isB_scalar){
                    TnO = cpuImplementation.MatMul(workScheduler,TnA,1.0f/10.0f);
                }else{
                    TnO = cpuImplementation.DivideTiled(workScheduler,TnA,TnB);
                }
            }
            break;
        }
    }



    CudaTensorF* TnA_gpu = new CudaTensorF();
    CudaTensorF* TnB_gpu = new CudaTensorF();
    CudaTensorF* TnO_gpu = new CudaTensorF(TnA->getShape());
    
    TnA_gpu->InitWithHostData(TnA->getShape(),TnA->_buff);
    TnB_gpu->InitWithHostData(TnB->getShape(),TnB->_buff);



    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    while(TnA_gpu->getRank()<4){
        TnA_gpu->ExpandDimZero();
    }
    unsigned int dim0B,dim1B,dim2B,dim3B;
    if(TnB_gpu->getRank()==4){
        dim0B=TnB_gpu->getShape()[0];dim1B=TnB_gpu->getShape()[1];
        dim2B=TnB_gpu->getShape()[2];dim3B=TnB_gpu->getShape()[3];
    }
    if(TnB_gpu->getRank()==3){
        dim0B=0                     ;dim1B=TnB_gpu->getShape()[0];
        dim2B=TnB_gpu->getShape()[1];dim3B=TnB_gpu->getShape()[2];
    }
    if(TnB_gpu->getRank()==2){
        dim0B=0;dim1B=0;
        dim2B=TnB_gpu->getShape()[0];dim3B=TnB_gpu->getShape()[1];
    }
    if(TnB_gpu->getRank()==1 && !isB_scalar){
        dim0B=0;dim1B=0;
        dim2B=0;dim3B=TnB_gpu->getShape()[0];
    }
    if(isB_scalar){
        dim0B=0;dim1B=0;
        dim2B=0;dim3B=1; //and rank should be 1 which already is
    }
    CHECK(cudaDeviceSynchronize());

    mat_ops_try01(
            TnA_gpu->_buff,
            TnB_gpu->_buff,
            TnO_gpu->_buff,
            TnA_gpu->getRank(),
            TnA_gpu->getShape()[0],
            TnA_gpu->getShape()[1],
            TnA_gpu->getShape()[2],
            TnA_gpu->getShape()[3],
            TnB_gpu->getRank(),
            dim0B,
            dim1B,
            dim2B,
            dim3B,
            mode);
    
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    TensorF *TnO_result = TnO_gpu->TransferToHost();
    
    

    cout << "Test: Mode="<<mode <<" rankA: "<<rankA << " rankB: "<<rankB << " isB_Scalar: "<< isB_scalar <<endl;
    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout << "Matches: " << cpuImplementation.CompareTensors(workScheduler,TnO,TnO_result) << endl;

    // reset device
    CHECK(cudaDeviceReset());
}
  
int main(){
    int mode=0;
    //for(mode=0;mode<4;mode++)
    {
        Test_01(mode,4,4, false);
        //Test_01(mode,4,3, false);
        //Test_01(mode,4,2, false);
        Test_01(mode,4,1, false);
        //Test_01(mode,4,1, true);
        // --------------------------
        //Test_01(mode,3,3, false);
        //Test_01(mode,3,2, false);
        //Test_01(mode,3,1, false);
        //Test_01(mode,3,1, true);
        // --------------------------
        Test_01(mode,2,2, false);
        //Test_01(mode,2,1, false);
        Test_01(mode,2,1, true);
        // --------------------------
        Test_01(mode,1,1, false);
        //Test_01(mode,1,1, true);
    }
}