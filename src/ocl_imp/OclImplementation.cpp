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
        /*00*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/transpose.cl.cc", "transposeBatch_try01" ),
        /*01*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/mamul.cl.cc", "kernel_batch_matmul" ),
        /*02*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/square.cl.cc", "kernel_square" ),
        /*03*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/sqrt.cl.cc", "kernel_sqrt_float" ),
        /*04*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/relu.cl.cc", "kernel_relu" ),
        /*05*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/mat_add_sub_elementwisemul.cl.cc", "kernel_mat_ops_try01" ),
        /*06*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/concat.cl.cc", "kernel_concat_try01" ),
        /*07*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/tile.cl.cc", "kernel_tile_try03" ),
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
            printf ("Error in %s: %s\n",kernel->kernelName ,build_log);
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

    OclKernelObject *kernelObject = oclKernels[0];

    error =  clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2, sizeof(cl_uint), (void*)&dim0);
    error |= clSetKernelArg(kernelObject->kernel, 3, sizeof(cl_uint), (void*)&dim1);
    error |= clSetKernelArg(kernelObject->kernel, 4, sizeof(cl_uint), (void*)&dim2);
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {16, 16};
    unsigned long globalThreads[] = {dim2, dim1};

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject->kernel,
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

    OclKernelObject *kernelObject = oclKernels[1];

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




    err  = clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat1)->ocl_buff);
    err |= clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat2)->ocl_buff);
    err |= clSetKernelArg(kernelObject->kernel, 2, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);

    err |= clSetKernelArg(kernelObject->kernel, 3, shared_mem_size, NULL);

    err |= clSetKernelArg(kernelObject->kernel, 4, sizeof(cl_uint), (void*)& batchedMat1->getShape()[0]);

    err |= clSetKernelArg(kernelObject->kernel, 5, sizeof(cl_uint), (void*)& batchedMat1->getShape()[1]);
    err |= clSetKernelArg(kernelObject->kernel, 6, sizeof(cl_uint), (void*)& batchedMat1->getShape()[2]);

    err |= clSetKernelArg(kernelObject->kernel, 7, sizeof(cl_uint), (void*)& batchedMat2->getShape()[1]);
    err |= clSetKernelArg(kernelObject->kernel, 8, sizeof(cl_uint), (void*)& batchedMat2->getShape()[2]);

    err |= clSetKernelArg(kernelObject->kernel, 9, sizeof(cl_uint), (void*)& rsltTn->getShape()[1]);
    err |= clSetKernelArg(kernelObject->kernel, 10, sizeof(cl_uint), (void*)& rsltTn->getShape()[2]);

    err |= clSetKernelArg(kernelObject->kernel, 11, sizeof(cl_uint), (void*)& global_work_size[0]);
    err |= clSetKernelArg(kernelObject->kernel, 12, sizeof(cl_uint), (void*)& global_work_size[1]);
    err |= clSetKernelArg(kernelObject->kernel, 13, sizeof(cl_uint), (void*)& global_work_size[2]);



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
                                 kernelObject->kernel,
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
    assert(batchedMat->getLength()!=0);
    OclTensorF*rsltTn = new OclTensorF(context,batchedMat->getShape());

    OclKernelObject *kernelObject = oclKernels[2];

    cl_int error;
    cl_ulong len = batchedMat->getLength();
    error =  clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)batchedMat)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2, sizeof(cl_ulong), (void*)&len);
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {16, 16};
    size_t globalThreads[] = {len};

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject->kernel,
                                   1, //two-dim
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

    return rsltTn;
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

    TensorF* rsltTn = new OclTensorF(context ,inputTn1->getShape() );

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

/*
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
*/
    {

/*
    unsigned long block,grid;
    unsigned int dimB0_IsNotZero=1, dimB1_IsNotZero=1, dimB2_IsNotZero=1, dimB3_IsNotZero=1;
    if(rankA>4 || rankB>4){printf("Error@mat_ops_try01: BAD_RANK");return;}

    block = BLOCK_SIZE;
    grid = (dim0A*dim1A*dim2A*dim3A + block -1 )/(block);

    int tmp =15>>(4-rankB);
    dimB0_IsNotZero = (tmp >> 3) & 1;
    dimB1_IsNotZero = (tmp >> 2) & 1;
    dimB2_IsNotZero = (tmp >> 1) & 1;
    dimB3_IsNotZero = (tmp >> 0) & 1;

    if(rankB==1 && dim0B==0&&dim1B==0&&dim2B==0&&dim3B==1){//scalar value
        dimB3_IsNotZero=0; //force it to be zero, so in the kernel, indxS2 would be zero;
    }
    //printf("TnA: dim0: %u, dim1: %u, dim2: %u, dim3: %u\n",dim0A,dim1A,dim2A,dim3A);
    //printf("TnB: dim0: %u, dim1: %u, dim2: %u, dim3: %u\n",dim0B,dim1B,dim2B,dim3B);

    //printf("block: %ld, grid: %ld\n",block,grid);
    //printf("mode: %d, isNotZero0: %d, isNotZero1: %d, isNotZero2: %d, isNotZero3: %d\n",
    //       operationMode,dimB0_IsNotZero,dimB1_IsNotZero,dimB2_IsNotZero,dimB3_IsNotZero);

    kernel_mat_ops_try01<<<grid,block>>>(
            g_iA,
                    g_iB,
                    g_o,
                    dim0A, dim1A, dim2A, dim3A,
                    dim0B, dim1B, dim2B, dim3B,
                    dimB0_IsNotZero,
                    dimB1_IsNotZero,
                    dimB2_IsNotZero,
                    dimB3_IsNotZero,
                    operationMode,
                    1, // <----- EPT
                    dim0A*dim1A*dim2A*dim3A);
    */
        unsigned int dimB0_IsNotZero=1, dimB1_IsNotZero=1, dimB2_IsNotZero=1, dimB3_IsNotZero=1;
        int rankB = inputTn2->getRank();
        int operationMode = mode==MAT_OPS::ADD ? 0 :
                            mode==MAT_OPS::SUB ? 1 :
                            mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                            3;
        int EPT = 1;
        unsigned long lenA = inputTn1->getLength();

        int tmp =15>>(4-rankB);
        dimB0_IsNotZero = (tmp >> 3) & 1;
        dimB1_IsNotZero = (tmp >> 2) & 1;
        dimB2_IsNotZero = (tmp >> 1) & 1;
        dimB3_IsNotZero = (tmp >> 0) & 1;
        if(rankB==1 && dim0B==0&&dim1B==0&&dim2B==0&&dim3B==1){//scalar value
            dimB3_IsNotZero=0; //force it to be zero, so in the kernel, indxS2 would be zero;
        }

        OclKernelObject *kernelObject = oclKernels[5];

        cl_int error;

        error =  clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn1)->ocl_buff);
        error =  clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn2)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel, 2, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
        error |= clSetKernelArg(kernelObject->kernel, 3, sizeof(cl_uint), (void*)&dim0);
        error |= clSetKernelArg(kernelObject->kernel, 4, sizeof(cl_uint), (void*)&dim1);
        error |= clSetKernelArg(kernelObject->kernel, 5, sizeof(cl_uint), (void*)&dim2);
        error |= clSetKernelArg(kernelObject->kernel, 6, sizeof(cl_uint), (void*)&dim3);
        error |= clSetKernelArg(kernelObject->kernel, 7, sizeof(cl_uint), (void*)&dim0B);
        error |= clSetKernelArg(kernelObject->kernel, 8, sizeof(cl_uint), (void*)&dim1B);
        error |= clSetKernelArg(kernelObject->kernel, 9, sizeof(cl_uint), (void*)&dim2B);
        error |= clSetKernelArg(kernelObject->kernel, 10, sizeof(cl_uint), (void*)&dim3B);
        error |= clSetKernelArg(kernelObject->kernel, 11, sizeof(cl_uint), (void*)&dimB0_IsNotZero);
        error |= clSetKernelArg(kernelObject->kernel, 12, sizeof(cl_uint), (void*)&dimB0_IsNotZero);
        error |= clSetKernelArg(kernelObject->kernel, 13, sizeof(cl_uint), (void*)&dimB0_IsNotZero);
        error |= clSetKernelArg(kernelObject->kernel, 14, sizeof(cl_uint), (void*)&dimB0_IsNotZero);
        error |= clSetKernelArg(kernelObject->kernel, 15, sizeof(cl_int), (void*)&operationMode);
        error |= clSetKernelArg(kernelObject->kernel, 16, sizeof(cl_int), (void*)&EPT);
        error |= clSetKernelArg(kernelObject->kernel, 17, sizeof(cl_ulong), (void*)&lenA);
        if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
        assert(error==0);

        cl_event exeEvt;
        //unsigned long localThreads[]  = {16};
        size_t globalThreads[] = {lenA};

        error = clEnqueueNDRangeKernel(queue,
                                       kernelObject->kernel,
                                       1,
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
    }

    for(int i =0;i<rankDiff;i++){
        inputTn1->SqueezeDimZero();
        rsltTn->SqueezeDimZero();
    }

    return rsltTn;

/*
    unsigned long block,grid;
    unsigned int dimB0_IsNotZero=1, dimB1_IsNotZero=1, dimB2_IsNotZero=1, dimB3_IsNotZero=1;
    if(rankA>4 || rankB>4){printf("Error@mat_ops_try01: BAD_RANK");return;}

    block = BLOCK_SIZE;
    grid = (dim0A*dim1A*dim2A*dim3A + block -1 )/(block);

    int tmp =15>>(4-rankB);
    dimB0_IsNotZero = (tmp >> 3) & 1;
    dimB1_IsNotZero = (tmp >> 2) & 1;
    dimB2_IsNotZero = (tmp >> 1) & 1;
    dimB3_IsNotZero = (tmp >> 0) & 1;

    if(rankB==1 && dim0B==0&&dim1B==0&&dim2B==0&&dim3B==1){//scalar value
        dimB3_IsNotZero=0; //force it to be zero, so in the kernel, indxS2 would be zero;
    }
    //printf("TnA: dim0: %u, dim1: %u, dim2: %u, dim3: %u\n",dim0A,dim1A,dim2A,dim3A);
    //printf("TnB: dim0: %u, dim1: %u, dim2: %u, dim3: %u\n",dim0B,dim1B,dim2B,dim3B);

    //printf("block: %ld, grid: %ld\n",block,grid);
    //printf("mode: %d, isNotZero0: %d, isNotZero1: %d, isNotZero2: %d, isNotZero3: %d\n",
    //       operationMode,dimB0_IsNotZero,dimB1_IsNotZero,dimB2_IsNotZero,dimB3_IsNotZero);

    kernel_mat_ops_try01<<<grid,block>>>(
            g_iA,
                    g_iB,
                    g_o,
                    dim0A, dim1A, dim2A, dim3A,
                    dim0B, dim1B, dim2B, dim3B,
                    dimB0_IsNotZero,
                    dimB1_IsNotZero,
                    dimB2_IsNotZero,
                    dimB3_IsNotZero,
                    operationMode,
                    1, // <----- EPT
                    dim0A*dim1A*dim2A*dim3A);
    */
}

TensorF* OclImplementation::MatOps(WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode){
    PrintInfo("MatOps",
              "mode",(mode==MAT_OPS::ADD ? 0 :
                      mode==MAT_OPS::SUB ? 1 :
                      mode==MAT_OPS::MUL_ELEMENTWISE ? 2 :
                      3),
              "",0,"",0,inputTn1->getShape(),{},{});
    float* val = new float[1]; val[0] = scalar;
    OclTensorF* tmpTn = new OclTensorF();
    tmpTn->InitWithHostData(context, queue, {1}, val);
    MatOps(scheduler,inputTn1,tmpTn,mode);
}

TensorF* OclImplementation::Sqrt(WorkScheduler scheduler, TensorF* inputTn){
    PrintInfo("Sqrt","",0,"",0,"",0,inputTn->getShape(),{});
    assert(inputTn->getLength()!=0);
    OclTensorF*rsltTn = new OclTensorF(context,inputTn->getShape());

    OclKernelObject *kernelObject = oclKernels[3];

    cl_int error;
    cl_ulong len = inputTn->getLength();
    error =  clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2, sizeof(cl_ulong), (void*)&len);
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {16, 16};
    size_t globalThreads[] = {len};

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject->kernel,
                                   1, //two-dim
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

    return rsltTn;
}

///concat 2 matrices
/// [matA, matB]
TensorF* OclImplementation::Concat2(
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

    unsigned int EPT=4;

    OclTensorF* rsltTn = new OclTensorF(context, {dimA0,dimA1,dimA2,dimA3+dimB3});

    /*
    concat_try01(
            inputTn1->_buff,
            inputTn2->_buff,
            rsltTn->_buff,
            dimA0,dimA1,dimA2,dimA3,
            dimB0,dimB1,dimB2,dimB3,
            (unsigned int)concatDim);
    */

    OclKernelObject *kernelObject = oclKernels[6];

    cl_int error;
    error =  clSetKernelArg(kernelObject->kernel, 0 , sizeof(cl_mem) , (void*)&((OclTensorF*)inputTn1)->ocl_buff);
    error =  clSetKernelArg(kernelObject->kernel, 1 , sizeof(cl_mem) , (void*)&((OclTensorF*)inputTn2)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2 , sizeof(cl_mem) , (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 3 , sizeof(cl_uint), (void*)&dimA0);
    error |= clSetKernelArg(kernelObject->kernel, 4 , sizeof(cl_uint), (void*)&dimA1);
    error |= clSetKernelArg(kernelObject->kernel, 5 , sizeof(cl_uint), (void*)&dimA2);
    error |= clSetKernelArg(kernelObject->kernel, 6 , sizeof(cl_uint), (void*)&dimA3);
    error |= clSetKernelArg(kernelObject->kernel, 7, sizeof(cl_uint), (void*)&dimB3);
    error |= clSetKernelArg(kernelObject->kernel, 8, sizeof(cl_int), (void*)&concatDim);
    error |= clSetKernelArg(kernelObject->kernel, 9, sizeof(cl_int), (void*)&EPT);
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {16, 16};
    size_t globalThreads[] = {rsltTn->getLength()};

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject->kernel,
                                   1,
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

    return rsltTn;
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
    assert(inputTn->getLength()!=0);
    OclTensorF*rsltTn = new OclTensorF(context,inputTn->getShape());

    OclKernelObject *kernelObject = oclKernels[4];

    cl_int error;
    cl_ulong len = inputTn->getLength();
    error =  clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2, sizeof(cl_ulong), (void*)&len);
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {16, 16};
    size_t globalThreads[] = {len};

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject->kernel,
                                   1, //two-dim
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

    return rsltTn;
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

    int EPT=2;
    int rank = inputTn->getRank();
    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    OclTensorF* rsltTn = nullptr;
    if(inputTn->getRank()==4 &&  tileAxis==2){
        rsltTn= new OclTensorF(context, {_dim0,_dim1,(unsigned int)tileCount,_dim3});
    }
    if(inputTn->getRank()==3 &&  tileAxis==1){
        rsltTn= new OclTensorF(context, {_dim0,(unsigned int)tileCount,_dim2});
    }
    if(inputTn->getRank()==3 &&  tileAxis==2){
        rsltTn= new OclTensorF(context, {_dim0,_dim1,(unsigned int)tileCount});
    }

    /*
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
    );*/

    OclKernelObject *kernelObject = oclKernels[7];

    cl_int error;
    cl_ulong len = inputTn->getLength();
    error =  clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem),  (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem),  (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2, sizeof(cl_uint), (void*)&_dim0);
    error |= clSetKernelArg(kernelObject->kernel, 3, sizeof(cl_uint), (void*)&_dim1);
    error |= clSetKernelArg(kernelObject->kernel, 4, sizeof(cl_uint), (void*)&_dim2);
    error |= clSetKernelArg(kernelObject->kernel, 5, sizeof(cl_uint), (void*)&_dim3);
    error |= clSetKernelArg(kernelObject->kernel, 6, sizeof(cl_int),  (void*)&rank);
    error |= clSetKernelArg(kernelObject->kernel, 7, sizeof(cl_int),  (void*)&tileAxis);
    error |= clSetKernelArg(kernelObject->kernel, 8, sizeof(cl_int),  (void*)&tileCount);
    error |= clSetKernelArg(kernelObject->kernel, 9, sizeof(cl_int),  (void*)&EPT);
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {16, 16};
    size_t globalThreads[] = {len};

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject->kernel,
                                   1, //two-dim
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

    return rsltTn;
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
