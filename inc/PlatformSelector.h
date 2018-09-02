//
// Created by saleh on 8/22/18.
//

#ifndef DEEPPOINTV1_PLATFORMSELECTOR_H
#define DEEPPOINTV1_PLATFORMSELECTOR_H

#include <vector>
#include "../inc/TensorF.h"
#include "../inc/TensorI.h"
#include "../inc/WeightsLoader.h"
using namespace std;


class PlatformSelector {
public:
    PlatformSelector(PLATFORMS defaultPlatform = PLATFORMS::CPU, vector<PLATFORMS> neededPlatforms);
    Tensor* CrossThePlatform(TensorF* srcTn, PLATFORMS platform);
    Tensor* CrossThePlatform(TensorI* srcTn, PLATFORMS platform);

    TensorF* Transpose(PLATFORM platform, WorkScheduler scheduler, TensorF *batchedMat);
    TensorF* MatMul(PLATFORM platform, WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2);
    TensorF* MatMul(PLATFORM platform, WorkScheduler scheduler, TensorF* batchedMat, float scalar);
    TensorF* Square(PLATFORM platform, WorkScheduler scheduler, TensorF* batchedMat);
    TensorF* ReduceSum(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2);
    TensorF* ReduceSum4D(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3);
    TensorF* Mean(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3);
    TensorF* Variance(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3);
    TensorF* MatAdd(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MatSub(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MatAddTiled(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2);
    TensorF* MatSubTiled(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2);
    TensorF* MatAddTiled(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, float scalar);
    TensorF* MatSubTiled(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, float scalar);
    TensorF* Sqrt(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Multiply(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* Divide(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MultiplyTiled(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* DivideTiled(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* Concat2(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim);
    TensorF* ReduceMax(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn, int reductionDim);

    TensorI* TopK(PLATFORM platform, WorkScheduler scheduler, TensorF* batchedMat, int axis, int k);
    TensorF* Gather(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis);
    TensorF* Conv2D(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2);
    TensorF* ReLU(PLATFORM platform, WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Tile(PLATFORM platform, WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount);

    void     DumpMatrix(PLATFORM platform, WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir);


    WeightsLoader* weightsLoader;
    PLATFORMS defaultPlatform;
private:
    PlatformImplementation *cpuPlatformClass;
    PlatformImplementation *cudaPlatformClass;
    PlatformImplementation *openclPlatformClass;
    PlatformImplementation *fpgaPlatformClass;

};


#endif //DEEPPOINTV1_PLATFORMSELECTOR_H
