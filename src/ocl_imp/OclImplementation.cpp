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
        new OclKernelObject(KERNEL_DIR , "/kernels/ocl/transpose.cl.cc", "transposeBatch_try01" ),
        new OclKernelObject(KERNEL_DIR , "/kernels/ocl/mamul.cl.cc", "kernel_batch_matmul" ),
    };

    for(OclKernelObject *kernel : oclKernels){
        ReadKernelSource(kernel);

        //--------------------------------------------------------------------------------------------------------------
        program = clCreateProgramWithSource(context, 1, (const char **)&kernel->src,
                                            (const size_t *)&kernel->size, &err);
        if (err != CL_SUCCESS) {
            printf("Failed to create OpenCL program from source %d\n\t%s\n", (int) err, getErrorString(err));
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

        kernel->kernel = clCreateKernel(program, kernel->kernelName, &err);
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

void OclImplementation::GetPaddedWorkSize(
        int dims,
        size_t * inBlockSize,
        size_t * inWorkSize,
        size_t * outPaddedWorkSize){
    for(int i = 0; i < dims; i++){
        outPaddedWorkSize[i] = (inWorkSize[i] + inBlockSize[i] - 1 ) / (inBlockSize[i]);
        outPaddedWorkSize[i] *= inBlockSize[i];
    }
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
    cl_int error;

    assert(batchedMat->getRank()==2 || batchedMat->getRank()==3);
    int rankDiff = 3 - batchedMat->getRank();
    if(rankDiff) batchedMat->ExpandDimZero();

    cl_uint dim0,dim1,dim2;
    dim0 = batchedMat->getShape()[0];
    dim1 = batchedMat->getShape()[1];
    dim2 = batchedMat->getShape()[2];

    OclTensorF *rsltTn = new OclTensorF(context,{dim0,dim2,dim1});

    error = clSetKernelArg(oclKernels[0]->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat)->ocl_buff);
    error |= clSetKernelArg(oclKernels[0]->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(oclKernels[0]->kernel, 2, sizeof(cl_uint), (void*)&dim0);
    error |= clSetKernelArg(oclKernels[0]->kernel, 3, sizeof(cl_uint), (void*)&dim1);
    error |= clSetKernelArg(oclKernels[0]->kernel, 4, sizeof(cl_uint), (void*)&dim2);
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {16, 16};
    unsigned long globalThreads[] = {dim2, dim1};

    error = clEnqueueNDRangeKernel(queue,
                                   oclKernels[0]->kernel,
                                   2, //two-dim
                                   NULL,
                                   globalThreads,
                                   NULL, //localThreads,
                                   0,
                                   NULL,
                                   &exeEvt);
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    clWaitForEvents(1, &exeEvt);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }

    if(rankDiff) batchedMat->SqueezeDimZero();
    return rsltTn;
}

TensorF* OclImplementation::MatMul(WorkScheduler scheduler,
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

    OclTensorF*rsltTn = new OclTensorF(context,{dim0A,dim1A, dim2B});


    /*
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
    */

    const cl_uint BLOCK_DIM_X=8, BLOCK_DIM_Y=4;

    size_t global_work_size[] = {dim2B, dim1A, dim0A};
    size_t global_padded_work_size[3];
    size_t local_block_size[] = {BLOCK_DIM_X, BLOCK_DIM_Y, 1};
    unsigned long shared_mem_size = (BLOCK_DIM_Y*dim2A + BLOCK_DIM_X* dim1B)*sizeof(float);

    GetPaddedWorkSize(3, local_block_size, global_work_size, global_padded_work_size);

    cout<< "LOCAL:      " << local_block_size[0] << ", " <<local_block_size[1] << ", " <<local_block_size[2] << "\n";
    cout<< "GLOBAL:     " << global_work_size[0] << ", " <<global_work_size[1] << ", " <<global_work_size[2] << "\n";
    cout<< "GLOBAL_PAD: " << global_padded_work_size[0] << ", " <<global_padded_work_size[1] << ", " <<global_padded_work_size[2] << "\n";




    err  = clSetKernelArg(oclKernels[1]->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat1)->ocl_buff);
    err |= clSetKernelArg(oclKernels[1]->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat2)->ocl_buff);
    err |= clSetKernelArg(oclKernels[1]->kernel, 2, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);

    err |= clSetKernelArg(oclKernels[1]->kernel, 3, shared_mem_size, NULL);

    err |= clSetKernelArg(oclKernels[1]->kernel, 4, sizeof(cl_uint), (void*)& batchedMat1->getShape()[0]);

    err |= clSetKernelArg(oclKernels[1]->kernel, 5, sizeof(cl_uint), (void*)& batchedMat1->getShape()[1]);
    err |= clSetKernelArg(oclKernels[1]->kernel, 6, sizeof(cl_uint), (void*)& batchedMat1->getShape()[2]);

    err |= clSetKernelArg(oclKernels[1]->kernel, 7, sizeof(cl_uint), (void*)& batchedMat2->getShape()[1]);
    err |= clSetKernelArg(oclKernels[1]->kernel, 8, sizeof(cl_uint), (void*)& batchedMat2->getShape()[2]);

    err |= clSetKernelArg(oclKernels[1]->kernel, 9, sizeof(cl_uint), (void*)& rsltTn->getShape()[1]);
    err |= clSetKernelArg(oclKernels[1]->kernel, 10, sizeof(cl_uint), (void*)& rsltTn->getShape()[2]);

    err |= clSetKernelArg(oclKernels[1]->kernel, 11, sizeof(cl_uint), (void*)& global_work_size[0]);
    err |= clSetKernelArg(oclKernels[1]->kernel, 12, sizeof(cl_uint), (void*)& global_work_size[1]);
    err |= clSetKernelArg(oclKernels[1]->kernel, 13, sizeof(cl_uint), (void*)& global_work_size[2]);



    /*
        kernel_batch_matmul<8,4> <<<gridsize, blocksize, sharedmemsize>>>(
                matA,
                        matB,
                        matC,
                        dim0A,

                        dim1A, //hA
                        dim2A, //wA

                        dim1B, //hA
                        dim2B, //wA

                        dim1A,
                        dim2B);
    */


    cl_event exeEvt;
    err = clEnqueueNDRangeKernel(queue,
                                   oclKernels[1]->kernel,
                                   3, //two-dim
                                   NULL,
                                   global_padded_work_size,
                                   local_block_size,
                                   0,
                                   NULL,
                                   &exeEvt);
    if(err != CL_SUCCESS) cout<<getErrorString(err)<<endl;
    clWaitForEvents(1, &exeEvt);

    if(err != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }





    for(int i=0;i<rankDiff;i++){
        batchedMat1->SqueezeDimZero();
        batchedMat2->SqueezeDimZero();
        rsltTn->SqueezeDimZero();
    }

    return rsltTn;
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

const char * OclImplementation::getErrorString(cl_int error)
{
    switch(error){
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

            // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}
