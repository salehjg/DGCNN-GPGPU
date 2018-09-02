//
// Created by saleh on 8/22/18.
//

#ifndef DEEPPOINTV1_CPUIMPLEMENTATION_H
#define DEEPPOINTV1_CPUIMPLEMENTATION_H

#include <iostream>
#include <string>
#include "../PlatformImplementation.h"
#include "../../inc/TensorF.h"
#include "../../inc/TensorI.h"

using namespace std;

class CpuImplementation: public PlatformImplementation {
public:
    CpuImplementation();///TODO: Constructor should handle platform initialization procedure!
    /*
    float* Transpose(float* input, int batchsize, int matrix_rank, int matrixH, int matrixW);
    float* MatMul(float* mat1, float* mat2, int batchsize, int matrix_rank, int matrixH1, int matrixW1, int matrixH2, int matrixW2);
    float* MatMul(float* mat1, float scalar, int batchsize, int matrix_rank, int matrixH1, int matrixW1);
    float* MatAdd(float* mat1, float* mat2, int rank, int dim0, int dim1, int dim2);
    float* MatSub(float* mat1, float* mat2, int rank, int dim0, int dim1, int dim2, int dim3);
    float* Square(float* mat1, int batchsize, int matrix_rank, int matrixH1, int matrixW1);
    float* ReduceSum(float* mat1, bool over_axis0, bool over_axis1, bool over_axis2, int dim0, int dim1, int dim2);
    float* ReduceSum4D(float* mat1, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3, int dim0, int dim1, int dim2, int dim3);
    float* Mean(float* mat, int rank, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3, int dim0, int dim1, int dim2, int dim3);
    float* Variance(float* mat, int rank, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3, int dim0, int dim1, int dim2, int dim3);
    float* Concat2(float* matA, float* matB, int rank, int concat_dim, int dimA0, int dimA1, int dimA2, int dimA3, int dimB0, int dimB1, int dimB2, int dimB3);
    float* ReduceMax(float* input, int axis, int rank, int dim0, int dim1, int dim2, int dim3);
    template<typename T> int DumpMatrix(string npy_fname, int rank, T* mat, int dim0, int dim1, int dim2, int dim3, int dim4, string npy_dir);
    */


    TensorF* Transpose(WorkScheduler scheduler, TensorF *batchedMat);
    TensorF* MatMul(WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2);
    TensorF* MatMul(WorkScheduler scheduler, TensorF* batchedMat, float scalar);
    TensorF* Square(WorkScheduler scheduler, TensorF* batchedMat);
    TensorF* ReduceSum(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2);
    TensorF* ReduceSum4D(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3);
    TensorF* Mean(WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3);
    TensorF* Variance(WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3);
    TensorF* MatAdd(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MatSub(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MatAddTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2);
    TensorF* MatSubTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2);
    TensorF* MatAddTiled(WorkScheduler scheduler, TensorF* inputTn1, float scalar);
    TensorF* MatSubTiled(WorkScheduler scheduler, TensorF* inputTn1, float scalar);
    TensorF* Sqrt(WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Multiply(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* Divide(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MultiplyTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* DivideTiled(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* Concat2(WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim);
    TensorF* ReduceMax(WorkScheduler scheduler, TensorF* inputTn, int reductionDim);

    TensorI* TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k);
    TensorF* Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis);
    TensorF* Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2=-1);
    TensorF* ReLU(WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount);

    void     DumpMatrix(WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir);


private:


};


#endif //DEEPPOINTV1_CPUIMPLEMENTATION_H
