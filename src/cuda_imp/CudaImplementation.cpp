//
// Created by saleh on 8/22/18.
//

#include "../../inc/cuda_imp/CudaImplementation.h"
#include <cuda_runtime.h>
#include "common.h"
#include <iostream>
#include <cuda_imp/CudaMemHelper.h>

using namespace std;

CudaImplementation::CudaImplementation(int aa) {
    a = aa;

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    cout<<"Selected CUDA Device: "<< dev<<", "<< deviceProp.name<<endl;
    CHECK(cudaSetDevice(dev));
    CHECK(cudaDeviceReset());
    CHECK(cudaDeviceSynchronize());
}

















void CudaImplementation::PrintInfo(
        string opName,
        const string &setting1, int val1,
        const string &setting2, int val2,
        const string &setting3, float val3,
        vector<unsigned int> shape1,
        vector<unsigned int> shape2,
        vector<bool> comb){

    string finalStr ;
    if(!setting1.empty() && !setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting1+ "=" + to_string(val1)+ ", " + setting2+ "=" + to_string(val2);
    }else if(!setting1.empty() && setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting1+ "=" + to_string(val1);
    }else if(setting1.empty() && !setting2.empty()){
        finalStr = "\t\t** " + opName + ": " + setting2+ "=" + to_string(val2);
    }else if(setting1.empty() && setting2.empty()){
        finalStr = "\t\t** " + opName + ": " ;
    }

    if(!setting3.empty()){
        finalStr += ", " + setting3 + ": " + to_string(val3);
    }

    if(!shape1.empty()){
        finalStr += ", Shape1=";
        for(unsigned int i : shape1){ finalStr += to_string(i) + "x"; }
        finalStr += ", ";
    }
    if(!shape2.empty()){
        finalStr += ", Shape2=";
        for(unsigned int i : shape2){ finalStr += to_string(i) + "x"; }
        finalStr += ", ";
    }
    if(!comb.empty()){
        finalStr += ", Combination=";
        for(bool i : comb){ finalStr += to_string(i) + "-"; }
        finalStr += ", ";
    }
    finalStr+="\n";
    //cout<<finalStr;
}

TensorF* CudaImplementation::Transpose(WorkScheduler scheduler, TensorF *batchedMat){
    PrintInfo("Transpose","",0,"",0,"",0,batchedMat->getShape(),{});

}

TensorF* CudaImplementation::MatMul(WorkScheduler scheduler,
                                   TensorF* batchedMat1, TensorF* batchedMat2){
    PrintInfo("MatMul","",0,"",0,"",0,batchedMat1->getShape(),batchedMat2->getShape());

}

TensorF* CudaImplementation::MatMul(WorkScheduler scheduler, TensorF* batchedMat, float scalar){
    PrintInfo("MatMul_Scalar","",0,"",0,"scalarVal",scalar,batchedMat->getShape(),{});

}

TensorF* CudaImplementation::Square(WorkScheduler scheduler, TensorF* batchedMat){
    PrintInfo("Square","",0,"",0,"",0,batchedMat->getShape(),{});

}

TensorF* CudaImplementation::ReduceSum(WorkScheduler scheduler,
                                      TensorF* inputTn,
                                      bool over_axis0,
                                      bool over_axis1,
                                      bool over_axis2){
    PrintInfo("ReduceSum","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2});
    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];

    CudaTensorF* rsltTn ;
    if(inputTn->getRank()==3 &&  !over_axis0 && !over_axis1 && over_axis2)rsltTn= new CudaTensorF({_dim0,_dim1});
    if(inputTn->getRank()==2 &&  !over_axis0 && over_axis1 )rsltTn= new CudaTensorF({_dim0});

    CHECK(cudaDeviceSynchronize());
    reduce_sum_3d_try03(
            inputTn->_buff,
            rsltTn->_buff,
            _dim0,
            _dim1,
            _dim2,
            over_axis0,
            over_axis1,
            over_axis2);
    CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("ReduceSum@CudaImplementation: KERNEL_ERROR");
    return rsltTn;
}

///[axis0,axis1,axis2,axis3] //No batch op, uses data as is
TensorF* CudaImplementation::ReduceSum4D(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        bool over_axis0,
                                        bool over_axis1,
                                        bool over_axis2,
                                        bool over_axis3){

    PrintInfo("ReduceSum4D","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2,over_axis3});
    assert(over_axis0&&over_axis1&&over_axis2&&!over_axis3); // TTTF ONLY

    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    CudaTensorF* rsltTn = new CudaTensorF({_dim3}); // TTTF

    CHECK(cudaDeviceSynchronize());
    reduce_sum_4d_try04(
            inputTn->_buff,
            rsltTn->_buff,
            _dim0,
            _dim1,
            _dim2,
            _dim3,
            over_axis0,
            over_axis1,
            over_axis2,
            over_axis3);
    CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("ReduceSum4D@CudaImplementation: KERNEL_ERROR");

    return rsltTn;

}

TensorF* CudaImplementation::Mean(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3){

    PrintInfo("Mean","",0,"",0,"",0,inputTn->getShape(),{},{mean_axis0,mean_axis1,mean_axis2,mean_axis3});
    unsigned long _dim0,_dim1,_dim2,_dim3;
    bool _mean_axis0, _mean_axis1, _mean_axis2, _mean_axis3;
    assert(inputTn->getRank()==2 || inputTn->getRank()==4);
    assert(
            (mean_axis0 && mean_axis1 && mean_axis2 && !mean_axis3 && inputTn->getRank()==4) ||
            (mean_axis0 && !mean_axis1 && !mean_axis2 && !mean_axis3 && inputTn->getRank()==2)
    );
    if(inputTn->getRank()==4){
        _dim0 = inputTn->getShape()[0];
        _dim1 = inputTn->getShape()[1];
        _dim2 = inputTn->getShape()[2];
        _dim3 = inputTn->getShape()[3];
        _mean_axis0 = mean_axis0;
        _mean_axis1 = mean_axis1;
        _mean_axis2 = mean_axis2;
        _mean_axis3 = mean_axis3;
    }else if (inputTn->getRank()==2){
        _dim0 = 1;
        _dim1 = 1;
        _dim2 = inputTn->getShape()[0];
        _dim3 = inputTn->getShape()[1];
        _mean_axis0 = true;
        _mean_axis1 = true;
        _mean_axis2 = true;
        _mean_axis3 = false;
    }
    CudaTensorF* rsltTn = new CudaTensorF({(unsigned int)_dim3});
    CHECK(cudaDeviceSynchronize());
    reduce_mean_4d_try02(
            inputTn->_buff,
            rsltTn->_buff,
            _dim0,
            _dim1,
            _dim2,
            _dim3,
            _mean_axis0,
            _mean_axis1,
            _mean_axis2,
            _mean_axis3);
    CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("Mean@CudaImplementation: KERNEL_ERROR");
    return rsltTn;
}

TensorF* CudaImplementation::Variance(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3){
    PrintInfo("Variance","",0,"",0,"",0,inputTn->getShape(),{},{variance_axis0,variance_axis1,variance_axis2,variance_axis3});

}

TensorF* CudaImplementation::MatAdd(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("MatAdd","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

}

TensorF* CudaImplementation::MatSub(
        WorkScheduler scheduler,
        TensorF* inputTn1,
        TensorF* inputTn2){
    PrintInfo("MatSub","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

}

TensorF* CudaImplementation::MatAddTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2){
    PrintInfo("MatAddTiled","",0,"",0,"",0,inputTn1->getShape(),inputSmallTn2->getShape(),{});

}

TensorF* CudaImplementation::MatSubTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2){
    PrintInfo("MatSubTiled","",0,"",0,"",0,inputTn1->getShape(),inputSmallTn2->getShape(),{});

}

TensorF* CudaImplementation::MatAddTiled(WorkScheduler scheduler, TensorF* inputTn1, float scalar){
    PrintInfo("MatAddTiled","",0,"",0,"scalar",scalar,inputTn1->getShape(),{},{});

}

TensorF* CudaImplementation::MatSubTiled(WorkScheduler scheduler, TensorF* inputTn1, float scalar){
    PrintInfo("MatSubTiled","",0,"",0,"scalar",scalar,inputTn1->getShape(),{},{});

}

TensorF* CudaImplementation::Sqrt(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("MatSubTiled","",0,"",0,"",0,inputTn->getShape(),{},{});

}

TensorF* CudaImplementation::Multiply(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("Multiply","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

}

TensorF* CudaImplementation::Divide(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("Divide","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

}

TensorF* CudaImplementation::MultiplyTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("MultiplyTiled","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

}

TensorF* CudaImplementation::DivideTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    PrintInfo("DivideTiled","",0,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

}

///concat 2 matrices
/// [matA, matB]
TensorF* CudaImplementation::Concat2(
        WorkScheduler scheduler,
        TensorF* inputTn1,
        TensorF* inputTn2,
        int concatDim){
    PrintInfo("Concat2","concatDim",concatDim,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});
    unsigned int dimA0,dimA1,dimA2,dimA3;
    unsigned int dimB0,dimB1,dimB2,dimB3;
    dimA0 = inputTn1->getShape()[0]; dimB0 = inputTn2->getShape()[0];
    dimA1 = inputTn1->getShape()[1]; dimB1 = inputTn2->getShape()[1];
    dimA2 = inputTn1->getShape()[2]; dimB2 = inputTn2->getShape()[2];
    dimA3 = inputTn1->getShape()[3]; dimB3 = inputTn2->getShape()[3];
    CudaTensorF* rsltTn = new CudaTensorF({dimA0,dimA1,dimA2,dimA3+dimB3});

    CHECK(cudaDeviceSynchronize());

    concat_try01(
            inputTn1->_buff,
            inputTn2->_buff,
            rsltTn->_buff,
            dimA0,dimA1,dimA2,dimA3,
            dimB0,dimB1,dimB2,dimB3,
            (unsigned int)concatDim);

    CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("Concat2@CudaImplementation: KERNEL_ERROR");
    return rsltTn;
}

TensorF* CudaImplementation::ReduceMax(
        WorkScheduler scheduler,
        TensorF* inputTn,
        int reductionDim){
    PrintInfo("ReduceMax","reductionDim",reductionDim,"",0,"",0,inputTn->getShape(),{},{});

    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    CudaTensorF* rsltTn = nullptr;
    if(inputTn->getRank()==4 &&  reductionDim==1)rsltTn= new CudaTensorF({_dim0,_dim2,_dim3});
    if(inputTn->getRank()==4 &&  reductionDim==2)rsltTn= new CudaTensorF({_dim0,_dim1,_dim3});

    CHECK(cudaDeviceSynchronize());
    if(reductionDim==2){
        reduce_max_4d_try01(
                inputTn->_buff,
                rsltTn->_buff,
                _dim0,
                _dim1,
                _dim2,
                _dim3,
                reductionDim==0,
                reductionDim==1,
                reductionDim==2,
                reductionDim==3);
    }
    if(reductionDim==1){
        // for all reductionDim=1 : dim2 is always 1(i.e. tensor is 3 dimensional)
        reduce_max_3d_try01(
                inputTn->_buff,
                rsltTn->_buff,
                _dim0,
                _dim1,
                _dim3,
                reductionDim==0,
                reductionDim==1,
                reductionDim==2);
    }
    CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("ReduceMax@CudaImplementation: KERNEL_ERROR");
    return rsltTn;


}

TensorI* CudaImplementation::TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k){
    PrintInfo("TopK","axis",axis,"k",k,"",0,batchedMat->getShape(),{},{});

}

TensorF* CudaImplementation::Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    PrintInfo("Gather","indices_axis",indices_axis,"",0,"",0,inputTn->getShape(),indices->getShape(),{});

}

TensorF* CudaImplementation::Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    PrintInfo("Conv2D","overrideDim2",overrideDim2,"",0,"",0,inputTn->getShape(),weights->getShape(),{});

}

TensorF* CudaImplementation::ReLU(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("ReLU","",0,"",0,"",0,inputTn->getShape(),{},{});

}

TensorF* CudaImplementation::Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount) {
    //Makes new tensor with same rank as inputTn's with tileAxis, tileCount times multiplied
    //tileAxis is in respect to the input tensor's axes.
    //----------------------------------------------------------------------------------------
    // inputTn       rsltTn         tileAxis        inputTn's Rank
    // BxNx1xD ----> BxNxKxD        2               4
    // BxNx1   ----> BxNxK          2               3
    // Bx1xN   ----> BxKxN          1               3
    // 1xD     ----> KxD            0               2

    PrintInfo("Tile","tileAxis",tileAxis,"tileCount",tileCount,"",0,inputTn->getShape(),{},{});

    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    CudaTensorF* rsltTn = nullptr;
    if(inputTn->getRank()==4 &&  tileAxis==2){
        rsltTn= new CudaTensorF({_dim0,_dim1,(unsigned int)tileCount,_dim3});
    }
    if(inputTn->getRank()==3 &&  tileAxis==1){
        rsltTn= new CudaTensorF({_dim0,(unsigned int)tileCount,_dim2});
    }
    if(inputTn->getRank()==3 &&  tileAxis==2){
        rsltTn= new CudaTensorF({_dim0,_dim1,(unsigned int)tileCount});
    }


    CHECK(cudaDeviceSynchronize());

    tile_try03(
            inputTn->_buff,
            rsltTn->_buff,
            _dim0,
            _dim1,
            _dim2,
            inputTn->getRank()<4 ? 0:_dim3,
            inputTn->getRank(),
            tileAxis,
            tileCount
            );

    CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("Tile@CudaImplementation: KERNEL_ERROR");
    return rsltTn;
}

void CudaImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorF* inputTn,
        string npy_dir){

}

void CudaImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorI* inputTn,
        string npy_dir){

}

bool CudaImplementation::CompareTensors(WorkScheduler scheduler,TensorF *inputTn1, TensorF *inputTn2) {
    return false;
}