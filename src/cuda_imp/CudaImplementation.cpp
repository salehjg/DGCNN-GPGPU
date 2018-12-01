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
    assert(batchedMat->getRank()==2 || batchedMat->getRank()==3);
    int rankDiff = 3 - batchedMat->getRank();
    if(rankDiff) batchedMat->ExpandDimZero();

    unsigned int dim0,dim1,dim2;
    dim0 = batchedMat->getShape()[0];
    dim1 = batchedMat->getShape()[1];
    dim2 = batchedMat->getShape()[2];

    CudaTensorF *rsltTn = new CudaTensorF({dim0,dim2,dim1});

    //CHECK(cudaDeviceSynchronize());

    transpose(
            batchedMat->_buff,
            rsltTn->_buff,
            dim0,
            dim1,
            dim2);

    //CHECK(cudaDeviceSynchronize());

    if(rankDiff) batchedMat->SqueezeDimZero();

    return rsltTn;
}

TensorF* CudaImplementation::MatMul(WorkScheduler scheduler,
                                   TensorF* batchedMat1, TensorF* batchedMat2){
    PrintInfo("MatMul","",0,"",0,"",0,batchedMat1->getShape(),batchedMat2->getShape());
    assert(batchedMat1->getRank()==3 || batchedMat1->getRank()==2);
    assert(batchedMat2->getRank()==3 || batchedMat2->getRank()==2);
    assert(batchedMat2->getRank()==batchedMat2->getRank());

    unsigned int dim0A,dim1A,dim2A,dim0B,dim1B,dim2B;
    int rankDiff;

    rankDiff = 3 - batchedMat1->getRank();
    for(int i=0;i<rankDiff;i++){
        batchedMat1->ExpandDimZero();
        batchedMat2->ExpandDimZero();
    }

    dim0A = batchedMat1->getShape()[0]; // batch size
    dim1A = batchedMat1->getShape()[1]; // height of matrix
    dim2A = batchedMat1->getShape()[2]; // width of matrix

    dim0B = batchedMat2->getShape()[0]; // batch size
    dim1B = batchedMat2->getShape()[1]; // height of matrix
    dim2B = batchedMat2->getShape()[2]; // width of matrix

    //Width of A should be equal to the Height of B. (dim2A = dim1B)
    assert(dim0A == dim0B);
    assert(dim2A == dim1B);

    CudaTensorF*rsltTn = new CudaTensorF({dim0A,dim1A, dim2B});
    //CHECK(cudaDeviceSynchronize());
    batch_matmul(
             batchedMat1->_buff,
             batchedMat2->_buff,
             rsltTn->_buff,
             batchedMat1->getShape()[0],
             batchedMat1->getShape()[1],
             batchedMat1->getShape()[2],
             batchedMat2->getShape()[0],
             batchedMat2->getShape()[1],
             batchedMat2->getShape()[2]);
    //CHECK(cudaDeviceSynchronize());

    for(int i=0;i<rankDiff;i++){
        batchedMat1->SqueezeDimZero();
        batchedMat2->SqueezeDimZero();
        rsltTn->SqueezeDimZero();
    }

    return rsltTn;

}

/*
TensorF* CudaImplementation::MatMul(WorkScheduler scheduler, TensorF* batchedMat, float scalar){
    PrintInfo("MatMul_Scalar","",0,"",0,"scalarVal",scalar,batchedMat->getShape(),{});

}
*/

TensorF* CudaImplementation::Square(WorkScheduler scheduler, TensorF* batchedMat){
    PrintInfo("Square","",0,"",0,"",0,batchedMat->getShape(),{});
    assert(batchedMat->getLength()!=0);
    CudaTensorF* rsltTn = new CudaTensorF(batchedMat->getShape());

    //CHECK(cudaDeviceSynchronize());
    square(batchedMat->_buff,rsltTn->_buff,batchedMat->getLength());
    //CHECK(cudaDeviceSynchronize());

    return rsltTn;
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
    if(inputTn->getRank()==3 &&  !over_axis0 && over_axis1 && !over_axis2)rsltTn= new CudaTensorF({_dim0,_dim2});
    if(inputTn->getRank()==3 &&  over_axis0 && !over_axis1 && !over_axis2)rsltTn= new CudaTensorF({_dim1,_dim2});

    if(inputTn->getRank()==2 &&  !over_axis0 && over_axis1 )rsltTn= new CudaTensorF({_dim0});

    //CHECK(cudaDeviceSynchronize());
    reduce_sum_3d_try03(
            inputTn->_buff,
            rsltTn->_buff,
            _dim0,
            _dim1,
            _dim2,
            over_axis0,
            over_axis1,
            over_axis2);
    //CHECK(cudaDeviceSynchronize());

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

    //CHECK(cudaDeviceSynchronize());
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
    //CHECK(cudaDeviceSynchronize());

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
    //CHECK(cudaDeviceSynchronize());
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
    //CHECK(cudaDeviceSynchronize());

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
    unsigned long _dim0,_dim1,_dim2,_dim3;
    bool _variance_axis0, _variance_axis1, _variance_axis2, _variance_axis3;
    assert(inputTn->getRank()==2 || inputTn->getRank()==4);
    assert(
            (variance_axis0 && variance_axis1 && variance_axis2 && !variance_axis3 && inputTn->getRank()==4) ||
            (variance_axis0 && !variance_axis1 && !variance_axis2 && !variance_axis3 && inputTn->getRank()==2)
    );
    if(inputTn->getRank()==4){
        _dim0 = inputTn->getShape()[0];
        _dim1 = inputTn->getShape()[1];
        _dim2 = inputTn->getShape()[2];
        _dim3 = inputTn->getShape()[3];
        _variance_axis0 = variance_axis0;
        _variance_axis1 = variance_axis1;
        _variance_axis2 = variance_axis2;
        _variance_axis3 = variance_axis3;
    }else if (inputTn->getRank()==2){
        _dim0 = 1;
        _dim1 = 1;
        _dim2 = inputTn->getShape()[0];
        _dim3 = inputTn->getShape()[1];
        _variance_axis0 = true;
        _variance_axis1 = true;
        _variance_axis2 = true;
        _variance_axis3 = false;
    }
    CudaTensorF* rsltTn = new CudaTensorF({(unsigned int)_dim3});
    //CHECK(cudaDeviceSynchronize());
    reduce_variance_4d_try01(
            inputTn->_buff,
            rsltTn->_buff,
            _dim0,
            _dim1,
            _dim2,
            _dim3,
            _variance_axis0,
            _variance_axis1,
            _variance_axis2,
            _variance_axis3);
    //CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("Mean@CudaImplementation: KERNEL_ERROR");
    return rsltTn;
}

/*
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
*/

TensorF* CudaImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});


    int rankDiff;

    if(!(inputTn1->getRank()<=4 && inputTn1->getRank()>=1 && inputTn2->getRank()<=4 && inputTn2->getRank()>=1 )){
        cout<<"MatOps: ERROR_BAD_TENSOR_RANK-E1"<<endl;
        return nullptr;
    }

    if(inputTn1->getRank() < inputTn2->getRank()){
        cout<<"MatOps: ERROR_BAD_TENSOR_RANK-E2"<<endl;
        return nullptr;
    }

    //forcing inputTn1 to be of rank 4. (always)
    rankDiff = 4- inputTn1->getRank();
    while(inputTn1->getRank()<4){
        inputTn1->ExpandDimZero();
    }

    unsigned long indxS1;
    unsigned long indxS2;
    unsigned int dim0, dim1, dim2, dim3;
    unsigned int dim0B, dim1B, dim2B, dim3B;
    int dim0B_IsNotZero, dim1B_IsNotZero, dim2B_IsNotZero, dim3B_IsNotZero;

    TensorF* rsltTn = new CudaTensorF( inputTn1->getShape() );

    dim0 = inputTn1->getShape()[0];
    dim1 = inputTn1->getShape()[1];
    dim2 = inputTn1->getShape()[2];
    dim3 = inputTn1->getShape()[3];

    if(inputTn2->getRank()==4){
        dim0B=inputTn2->getShape()[0];
        dim1B=inputTn2->getShape()[1];
        dim2B=inputTn2->getShape()[2];
        dim3B=inputTn2->getShape()[3];
    }
    if(inputTn2->getRank()==3){
        dim0B=0                     ;
        dim1B=inputTn2->getShape()[0];
        dim2B=inputTn2->getShape()[1];
        dim3B=inputTn2->getShape()[2];
    }
    if(inputTn2->getRank()==2){
        dim0B=0;
        dim1B=0;
        dim2B=inputTn2->getShape()[0];
        dim3B=inputTn2->getShape()[1];
    }
    if(inputTn2->getRank()==1 && inputTn2->getShape()[0]!=1){
        dim0B=0;
        dim1B=0;
        dim2B=0;
        dim3B=inputTn2->getShape()[0];
    }
    if(inputTn2->getShape()[0]==1){
        dim0B=0;
        dim1B=0;
        dim2B=0;
        dim3B=1; //and rank should be 1 which already is
    }


    mat_ops_try01(
            inputTn1->_buff,
            inputTn2->_buff,
            rsltTn->_buff,
            inputTn1->getRank(),
            dim0,
            dim1,
            dim2,
            dim3,
            inputTn2->getRank(),
            dim0B,
            dim1B,
            dim2B,
            dim3B,
            mode==MAT_OPS::ADD ? 0 :
            mode==MAT_OPS::SUB ? 1 :
            mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
            3);

    for(int i =0;i<rankDiff;i++){
        inputTn1->SqueezeDimZero();
        rsltTn->SqueezeDimZero();
    }

    return rsltTn;
}

TensorF* CudaImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),{},{});
    float* val = new float[1]; val[0] = scalar;
    CudaTensorF* tmpTn = new CudaTensorF();
    tmpTn->InitWithHostData({1},val);
    MatOps(scheduler,inputTn1,tmpTn,mode);
}

TensorF* CudaImplementation::Sqrt(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("MatSubTiled","",0,"",0,"",0,inputTn->getShape(),{},{});

    CudaTensorF *rsltTn = new CudaTensorF(inputTn->getShape());

    //CHECK(cudaDeviceSynchronize());
    sqrt_float(inputTn->_buff,rsltTn->_buff,inputTn->getLength());
    //CHECK(cudaDeviceSynchronize());

    return rsltTn;
}

/*
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
*/

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

    //CHECK(cudaDeviceSynchronize());

    concat_try01(
            inputTn1->_buff,
            inputTn2->_buff,
            rsltTn->_buff,
            dimA0,dimA1,dimA2,dimA3,
            dimB0,dimB1,dimB2,dimB3,
            (unsigned int)concatDim);

    //CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("Concat2@CudaImplementation: KERNEL_ERROR");
    return rsltTn;
}

TensorF* CudaImplementation::ReduceMax(
        WorkScheduler scheduler,
        TensorF* inputTn,
        int reductionDim){
    PrintInfo("ReduceMax","reductionDim",reductionDim,"",0,"",0,inputTn->getShape(),{},{});
    assert(inputTn->getRank()==3 || inputTn->getRank()==4);

    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    CudaTensorF* rsltTn = nullptr;
    if(inputTn->getRank()==4 &&  reductionDim==1)rsltTn= new CudaTensorF({_dim0,_dim2,_dim3});
    if(inputTn->getRank()==4 &&  reductionDim==2)rsltTn= new CudaTensorF({_dim0,_dim1,_dim3});

    //CHECK(cudaDeviceSynchronize());
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
    //CHECK(cudaDeviceSynchronize());

    cudaCheckErrors("ReduceMax@CudaImplementation: KERNEL_ERROR");
    return rsltTn;
}

TensorI* CudaImplementation::TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k){
    PrintInfo("TopK","axis",axis,"k",k,"",0,batchedMat->getShape(),{},{});
    assert(batchedMat->getRank()==3);
    unsigned int b,m,n;
    b = batchedMat->getShape()[0];
    m = batchedMat->getShape()[1];
    n = batchedMat->getShape()[2];

    CudaTensorI *rsltIndicesTn = new CudaTensorI({
                                                        b,
                                                        m,
                                                        (unsigned int)k
                                                });
    CudaTensorF *rsltValTn = new CudaTensorF({
                                                         b,
                                                         m,
                                                         (unsigned int)k
                                                 });

    top_k(batchedMat->_buff,rsltIndicesTn->_buff,rsltValTn->_buff,b,n,m,k);

    return rsltIndicesTn;
}

TensorF* CudaImplementation::Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    PrintInfo("Gather","indices_axis",indices_axis,"",0,"",0,inputTn->getShape(),indices->getShape(),{});
    assert(inputTn->getRank()==3);
    assert(indices->getRank()==3);
    assert(inputTn->getShape()[0]==indices->getShape()[0]);

    unsigned int b,n,m,c,nsample;
    b = inputTn->getShape()[0];
    n = inputTn->getShape()[1];
    c = inputTn->getShape()[2];
    m = indices->getShape()[1];
    nsample = indices->getShape()[2];

    CudaTensorF* rsltTn = new CudaTensorF({b,m,nsample,c});

    //CHECK(cudaDeviceSynchronize());
    gather(inputTn->_buff,indices->_buff,rsltTn->_buff,b,n,c,m,nsample);
    //CHECK(cudaDeviceSynchronize());

    return rsltTn;
}

TensorF* CudaImplementation::Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    PrintInfo("Conv2D","overrideDim2",overrideDim2,"",0,"",0,inputTn->getShape(),weights->getShape(),{});
    CudaTensorF* buffTn = new CudaTensorF({
                inputTn->getShape()[0],
                inputTn->getShape()[1],
                inputTn->getShape()[2],
                weights->getShape()[3]});
    TensorF* rsltTn ;

    conv2d_mlp_try02(
            inputTn->_buff,
            weights->_buff,
            buffTn->_buff,
            inputTn->getShape()[0],
            inputTn->getShape()[1],
            inputTn->getShape()[2],
            inputTn->getShape()[3],
            weights->getShape()[3]);

    rsltTn = MatOps(scheduler,buffTn,biases,MAT_OPS::ADD);
    //delete(buffTn);
    return rsltTn;
}

TensorF* CudaImplementation::ReLU(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("ReLU","",0,"",0,"",0,inputTn->getShape(),{},{});
    assert(inputTn->getLength()!=0);
    CudaTensorF *rsltTn = new CudaTensorF(inputTn->getShape());
    activation_relu(inputTn->_buff, rsltTn->_buff, inputTn->getLength());

    return rsltTn;
}

TensorF* CudaImplementation::Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount) {
    //Makes new tensor with same rank as inputTn's with tileAxis, tileCount times multiplied
    //tileAxis is with respect to the input tensor's axes.
    //----------------------------------------------------------------------------------------
    // inputTn       rsltTn         tileAxis        inputTn's Rank
    // BxNx1xD ----> BxNxKxD        2               4
    // BxNx1   ----> BxNxK          2               3
    // Bx1xN   ----> BxKxN          1               3

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


    //CHECK(cudaDeviceSynchronize());

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

    //CHECK(cudaDeviceSynchronize());

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