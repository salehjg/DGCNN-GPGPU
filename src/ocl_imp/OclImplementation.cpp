//
// Created by saleh on 8/22/18.
//

#include "../../inc/ocl_imp/OclImplementation.h"
#include <iostream>
#include <CL/cl.h>
#include <assert.h>

using namespace std;

OclImplementation::OclImplementation(int aa) {
    a = aa;
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    assert(err == CL_SUCCESS);
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    assert(err == CL_SUCCESS);
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    assert(err == CL_SUCCESS);
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    assert(err == CL_SUCCESS);

    oclKernels = {
        new OclKernelObject(KERNEL_DIR , "kernel_test.cl.cc", "vectorAddition" ),
    };

    for(OclKernelObject *kernel : oclKernels){
        ReadKernelSource(kernel);

        //--------------------------------------------------------------------------------------------------------------
        program = clCreateProgramWithSource(context, 1, (const char **)&kernel->src,
                                            (const size_t *)&kernel->size, &err);
        if (err != CL_SUCCESS) {
            printf("Failed to create OpenCL program from source %d\n", (int) err);
            std::exit(1);
        }

        err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("Failed to build program %d\n", (int) err);
            char build_log[16348];
            clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, sizeof (build_log), build_log, NULL);
            printf ("Error in kernel: %s\n", build_log);
            std::exit(1);
        }

        kernel->kernel = clCreateKernel(program, "vectorAddition", &err);
        if (err != CL_SUCCESS) {
            printf("Failed to create kernel %d\n", (int) err);
            std::exit(1);
        }

        //--------------------------------------------------------------------------------------------------------------
        clReleaseProgram(program);
    }
}

cl_context OclImplementation::getContext(){
    return context;
}

cl_command_queue OclImplementation::getQueue() {
    return queue;
}


void OclImplementation::ReadKernelSource(OclKernelObject* object)
{
    const char *fileName = object->fileName;
    FILE *file=fopen(fileName,"rb");
    if(!file)
    {
        printf("ERROR:Failed to open file '%s'\n",fileName);
        return ;
    }

    if(fseek(file,0,SEEK_END))
    {
        printf("ERROR: Failed to open file '%s'\n",fileName);
        fclose(file);
        return ;
    }

    long size=ftell(file);
    if(size==0)
    {
        printf("ERROR: Failed to check position on file '%s'\n",fileName);
        fclose(file);
        return ;
    }

    rewind(file);

    char *src=(char *)malloc(sizeof(char)*size+1);
    if(!src)
    {
        printf("ERROR: Failed to allocate memory for file '%s'\n",fileName);
        fclose(file);
        return ;
    }

    size_t res=fread(src,1,sizeof(char) *size,file);
    if(res !=sizeof (char) * size)
    {
        printf("ERROR: Failed to read file '%s'\n",fileName);
        fclose(file);
        free(src);
        return ;
    }

    src[size]='\0';/*NULL terminated */
    fclose(file);

    object->src = src;
    object->size = size;
}


void OclImplementation::PrintInfo(
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

TensorF* OclImplementation::Transpose(WorkScheduler scheduler, TensorF *batchedMat){
    PrintInfo("Transpose","",0,"",0,"",0,batchedMat->getShape(),{});

    return nullptr;
}

TensorF* OclImplementation::MatMul(WorkScheduler scheduler,
                                   TensorF* batchedMat1, TensorF* batchedMat2){
    PrintInfo("MatMul","",0,"",0,"",0,batchedMat1->getShape(),batchedMat2->getShape());

    return nullptr;
}

TensorF* OclImplementation::Square(WorkScheduler scheduler, TensorF* batchedMat){
    PrintInfo("Square","",0,"",0,"",0,batchedMat->getShape(),{});

    return nullptr;
}

TensorF* OclImplementation::ReduceSum(WorkScheduler scheduler,
                                      TensorF* inputTn,
                                      bool over_axis0,
                                      bool over_axis1,
                                      bool over_axis2){
    PrintInfo("ReduceSum","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2});

    return nullptr;
}

///[axis0,axis1,axis2,axis3] //No batch op, uses data as is
TensorF* OclImplementation::ReduceSum4D(WorkScheduler scheduler,
                                        TensorF* inputTn,
                                        bool over_axis0,
                                        bool over_axis1,
                                        bool over_axis2,
                                        bool over_axis3){

    PrintInfo("ReduceSum4D","",0,"",0,"",0,inputTn->getShape(),{},{over_axis0,over_axis1,over_axis2,over_axis3});

    return nullptr;
}

TensorF* OclImplementation::Mean(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3){

    PrintInfo("Mean","",0,"",0,"",0,inputTn->getShape(),{},{mean_axis0,mean_axis1,mean_axis2,mean_axis3});

    return nullptr;
}

TensorF* OclImplementation::Variance(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3){
    PrintInfo("Variance","",0,"",0,"",0,inputTn->getShape(),{},{variance_axis0,variance_axis1,variance_axis2,variance_axis3});

    return nullptr;
}

TensorF* OclImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

    return nullptr;
}

TensorF* OclImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),{},{});

    return nullptr;
}

TensorF* OclImplementation::Sqrt(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("MatSubTiled","",0,"",0,"",0,inputTn->getShape(),{},{});

    return nullptr;
}

///concat 2 matrices
/// [matA, matB]
TensorF* OclImplementation::Concat2(
        WorkScheduler scheduler,
        TensorF* inputTn1,
        TensorF* inputTn2,
        int concatDim){
    PrintInfo("Concat2","concatDim",concatDim,"",0,"",0,inputTn1->getShape(),inputTn2->getShape(),{});

    return nullptr;
}

TensorF* OclImplementation::ReduceMax(
        WorkScheduler scheduler,
        TensorF* inputTn,
        int reductionDim){
    PrintInfo("ReduceMax","reductionDim",reductionDim,"",0,"",0,inputTn->getShape(),{},{});

    return nullptr;
}

TensorI* OclImplementation::TopK(WorkScheduler scheduler, TensorF* batchedMat, int axis, int k){
    PrintInfo("TopK","axis",axis,"k",k,"",0,batchedMat->getShape(),{},{});

    return nullptr;
}

TensorF* OclImplementation::Gather(WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    PrintInfo("Gather","indices_axis",indices_axis,"",0,"",0,inputTn->getShape(),indices->getShape(),{});

    return nullptr;
}

TensorF* OclImplementation::Conv2D(WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    PrintInfo("Conv2D","overrideDim2",overrideDim2,"",0,"",0,inputTn->getShape(),weights->getShape(),{});

    return nullptr;
}

TensorF* OclImplementation::ReLU(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("ReLU","",0,"",0,"",0,inputTn->getShape(),{},{});

    return nullptr;
}

TensorF* OclImplementation::Tile(WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount) {
    //Makes new tensor with same rank as inputTn's with tileAxis, tileCount times multiplied
    //tileAxis is in respect to the input tensor's axes.
    //----------------------------------------------------------------------------------------
    // inputTn       rsltTn         tileAxis        inputTn's Rank
    // BxNx1xD ----> BxNxKxD        2               4
    // BxNx1   ----> BxNxK          2               3
    // Bx1xN   ----> BxKxN          1               3
    // 1xD     ----> KxD            0               2

    PrintInfo("Tile","tileAxis",tileAxis,"tileCount",tileCount,"",0,inputTn->getShape(),{},{});

    return nullptr;
}

void OclImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorF* inputTn,
        string npy_dir){

}

void OclImplementation::DumpMatrix(
        WorkScheduler scheduler,
        string npy_fname,
        TensorI* inputTn,
        string npy_dir){

}

bool OclImplementation::CompareTensors(WorkScheduler scheduler,TensorF *inputTn1, TensorF *inputTn2) {
    return false;
}

