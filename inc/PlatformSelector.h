//
// Created by saleh on 8/22/18.
//

#ifndef DEEPPOINTV1_PLATFORMSELECTOR_H
#define DEEPPOINTV1_PLATFORMSELECTOR_H

#include <vector>
#include "../inc/TensorF.h"
#include "../inc/TensorI.h"
#include "../inc/WeightsLoader.h"
#include "../inc/PlatformImplementation.h"
using namespace std;


class PlatformSelector {
public:
    PlatformSelector(PLATFORMS defaultPlatform , vector<PLATFORMS> neededPlatforms);
    TensorF* CrossThePlatform(TensorF* srcTn, PLATFORMS platform);
    TensorI* CrossThePlatform(TensorI* srcTn, PLATFORMS platform);
    TensorF* Transpose(PLATFORMS platform, WorkScheduler scheduler, TensorF *batchedMat);
    TensorF* MatMul(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2);
    TensorF* MatMul(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat, float scalar);
    TensorF* Square(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat);
    TensorF* ReduceSum(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2);
    TensorF* ReduceSum4D(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3);
    TensorF* Mean(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3);
    TensorF* Variance(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3);
    TensorF* MatAdd(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MatSub(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MatAddTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2);
    TensorF* MatSubTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2);
    TensorF* MatAddTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, float scalar);
    TensorF* MatSubTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, float scalar);
    TensorF* Sqrt(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Multiply(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* Divide(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* MultiplyTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* DivideTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2);
    TensorF* Concat2(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim);
    TensorF* ReduceMax(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, int reductionDim);

    TensorI* TopK(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat, int axis, int k);
    TensorF* Gather(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis);
    TensorF* Conv2D(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2=-1);
    TensorF* ReLU(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn);
    TensorF* Tile(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount);

    void     DumpMatrix(PLATFORMS platform, WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir="/home/saleh/00_repos/tensorflow_repo/00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/data/matrix_dumps/");
    void     DumpMatrix(PLATFORMS platform, WorkScheduler scheduler, string npy_fname, TensorI* inputTn, string npy_dir="/home/saleh/00_repos/tensorflow_repo/00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/data/matrix_dumps/");


    WeightsLoader* weightsLoader;
    PLATFORMS defaultPlatform;
private:
    PlatformImplementation *cpuPlatformClass;
    PlatformImplementation *cudaPlatformClass;
    PlatformImplementation *openclPlatformClass;
    PlatformImplementation *fpgaPlatformClass;

};


#endif //DEEPPOINTV1_PLATFORMSELECTOR_H
