//
// Created by saleh on 8/22/18.
//

#ifndef DEEPPOINTV1_CUDAIMPLEMENTATION_H
#define DEEPPOINTV1_CUDAIMPLEMENTATION_H

#include <PlatformImplementation.h>
#include "Kernels.h"
#include "../../inc/cuda_imp/CudaTensorF.h"
#include "../../inc/cuda_imp/CudaTensorI.h"

class CudaImplementation: public PlatformImplementation {
public:
    CudaImplementation(int aa); ///TODO: Constructor should handle platform initialization procedure!




    TensorF* Transpose(WorkScheduler scheduler, TensorF *batchedMat);
    TensorF* MatMul(WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2);
    //TensorF* MatMul(WorkScheduler scheduler, TensorF* batchedMat, float scalar);
    TensorF* Square(WorkScheduler scheduler, TensorF* batchedMat);
    TensorF* ReduceSum(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2);
    TensorF* ReduceSum4D(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3);
    TensorF* Mean(WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3);
    TensorF* Variance(WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3);
    //TensorF* MatAdd(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    //TensorF* MatSub(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    //TensorF* MatAddTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2);
    //TensorF* MatSubTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2);
    //TensorF* MatAddTiled(WorkScheduler scheduler, TensorF* inputTn1, float scalar);
    //TensorF* MatSubTiled(WorkScheduler scheduler, TensorF* inputTn1, float scalar);
    TensorF* MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode);
    TensorF* MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode);
    TensorF* Sqrt(WorkScheduler scheduler, TensorF* inputTn);
    //TensorF* Multiply(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    //TensorF* Divide(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    //TensorF* MultiplyTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    //TensorF* DivideTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* Concat2(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim);
    TensorF* ReduceMax(WorkScheduler scheduler, TensorF* inputTn, int reductionDim);

    TensorI* TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k);
    TensorF* Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis);
    TensorF* Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2=-1);
    TensorF* ReLU(WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount);

    void     DumpMatrix(WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir);
    void     DumpMatrix(WorkScheduler scheduler, string npy_fname, TensorI* inputTn, string npy_dir);
    bool     CompareTensors(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);





private:
    int a;
    void PrintInfo(string opName, const string &setting1, int val1, const string &setting2, int val2,
                   const string &setting3, float val3, vector<unsigned int> shape1, vector<unsigned int> shape2, vector<bool> comb={});
};


#endif //DEEPPOINTV1_CUDAIMPLEMENTATION_H
