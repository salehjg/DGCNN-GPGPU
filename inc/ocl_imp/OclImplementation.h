//
// Created by saleh on 8/22/18.
//

#ifndef DEEPPOINTV1_OCLIMPLEMENTATION_H
#define DEEPPOINTV1_OCLIMPLEMENTATION_H

#include <PlatformImplementation.h>
#include "Kernels.h"
#include "../../inc/ocl_imp/OclTensorF.h"
#include "../../inc/ocl_imp/OclTensorI.h"
#include <CL/cl.h>

struct OclKernelObject{
    const char *fileName;
    char* src;
    long  size;
    const char* kernelName;
    cl_kernel kernel;

    OclKernelObject(string dir, string fname, const string &kernelName){
        string *fPath = new string();
        fPath->append(dir);
        fPath->append(fname);
        this->fileName = fPath->c_str();
        this->kernelName = kernelName.c_str();
    }
};

class OclImplementation: public PlatformImplementation {
public:
    OclImplementation(int aa); ///TODO: Constructor should handle platform initialization procedure!

    TensorF* Transpose(WorkScheduler scheduler, TensorF *batchedMat);
    TensorF* MatMul(WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2);
    TensorF* Square(WorkScheduler scheduler, TensorF* batchedMat);
    TensorF* ReduceSum(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2);
    TensorF* ReduceSum4D(WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3);
    TensorF* Mean(WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3);
    TensorF* Variance(WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3);
    TensorF* MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode);
    TensorF* MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode);
    TensorF* Sqrt(WorkScheduler scheduler, TensorF* inputTn);
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

    cl_context          getContext();
    cl_command_queue    getQueue();
    void                ReadKernelSource(OclKernelObject *object);

    std::vector<OclKernelObject*> oclKernels;

private:
    int a;
    void PrintInfo(string opName, const string &setting1, int val1, const string &setting2, int val2,
                   const string &setting3, float val3, vector<unsigned int> shape1, vector<unsigned int> shape2, vector<bool> comb={});

    const std::string KERNEL_DIR = "/home/saleh/00_repos/tensorflow_repo/00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/kernels/ocl/";
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_int err;
};


#endif //DEEPPOINTV1_OCLIMPLEMENTATION_H
