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
        /*00*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/transpose.cl.cc",                     "transposeBatch_try01"          ),
        /*01*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/mamul.cl.cc",                         "kernel_batch_matmul"           ),
        /*02*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/square.cl.cc",                        "kernel_square"                 ),
        /*03*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/sqrt.cl.cc",                          "kernel_sqrt_float"             ),
        /*04*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/relu.cl.cc",                          "kernel_relu"                   ),
        /*05*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/mat_add_sub_elementwisemul.cl.cc",    "kernel_mat_ops_try01"          ),
        /*06*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/concat.cl.cc",                        "kernel_concat_try01"           ),
        /*07*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/tile.cl.cc",                          "kernel_tile_try03"             ),
        /*08*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/reduce_max.cl.cc",                    "kernel_reduce_max_try01"       ),
        /*09*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/reduce_sum_3d.cl.cc",                 "kernel_reduce_sum_3d_try03"    ),
        /*10*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/reduce_sum_4d.cl.cc",                 "kernel_reduce_sum_4d_try04"    ),
        /*11*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/reduce_mean.cl.cc",                   "kernel_divide_by_const_try01"  ),
        /*12*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/reduce_variance.cl.cc",               "kernel_multiply_const_sub_try01"  ),
        /*13*/ new OclKernelObject(KERNEL_DIR , "/kernels/ocl/conv2d_mlp.cl.cc",                    "kernel_conv2d_mlp_try01"  ),
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
    unsigned int _dim0,_dim1,_dim2;
    int _overAxis0, _overAxis1, _overAxis2;

    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];

    _overAxis0 = over_axis0;
    _overAxis1 = over_axis1;
    _overAxis2 = over_axis2;

    OclTensorF* rsltTn ;
    if(inputTn->getRank()==3 &&  !over_axis0 && !over_axis1 && over_axis2)rsltTn= new OclTensorF(context, {_dim0,_dim1});
    if(inputTn->getRank()==3 &&  !over_axis0 && over_axis1 && !over_axis2)rsltTn= new OclTensorF(context, {_dim0,_dim2});
    if(inputTn->getRank()==3 &&  over_axis0 && !over_axis1 && !over_axis2)rsltTn= new OclTensorF(context, {_dim1,_dim2});


    if(inputTn->getRank()==2 &&  !over_axis0 && over_axis1 )rsltTn= new OclTensorF(context, {_dim0});

    /*
    reduce_sum_3d_try03(
            inputTn->_buff,
            rsltTn->_buff,
            _dim0,
            _dim1,
            _dim2,
            over_axis0,
            over_axis1,
            over_axis2);
    */

    OclKernelObject *kernelObject = oclKernels[9];

    size_t global_work_size[] = {rsltTn->getLength()};
    size_t global_padded_work_size[1];
    size_t local_block_size[] = {256};
    unsigned long shared_mem_size = (local_block_size[0])*sizeof(cl_float);
    GetPaddedWorkSize(1, local_block_size, global_work_size, global_padded_work_size);

    if(over_axis2){
        //each thread block(NOT each thread) is mapped into only a single dim2 slice
        global_padded_work_size[0] = local_block_size[0] * rsltTn->getLength();
        global_work_size[0] = local_block_size[0] * rsltTn->getLength();
    }

    cout<< "LOCAL:      " << local_block_size[0] << "\n";
    cout<< "GLOBAL:     " << global_work_size[0] << "\n";
    cout<< "GLOBAL_PAD: " << global_padded_work_size[0] << "\n";

    cl_int error;
    error =  clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2, shared_mem_size, NULL);
    error |= clSetKernelArg(kernelObject->kernel, 3, sizeof(cl_uint), (void*)&_dim2);
    error |= clSetKernelArg(kernelObject->kernel, 4, sizeof(cl_uint), (void*)&_dim1);
    error |= clSetKernelArg(kernelObject->kernel, 5, sizeof(cl_uint), (void*)&_dim0);
    error |= clSetKernelArg(kernelObject->kernel, 6, sizeof(cl_int), (void*)&_overAxis0);
    error |= clSetKernelArg(kernelObject->kernel, 7, sizeof(cl_int), (void*)&_overAxis1);
    error |= clSetKernelArg(kernelObject->kernel, 8, sizeof(cl_int), (void*)&_overAxis2);
    error |= clSetKernelArg(kernelObject->kernel, 9, sizeof(cl_ulong), (void*)&(global_work_size[0]));
    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject->kernel,
                                   1,
                                   NULL,
                                   global_padded_work_size,
                                   local_block_size,
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

///[axis0,axis1,axis2,axis3] //No batch op, uses data as is
TensorF* OclImplementation::ReduceSum4D(WorkScheduler scheduler,
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

    OclTensorF* rsltTn = new OclTensorF(context, {_dim3}); // TTTF
    OclKernelObject *kernelObject = oclKernels[10];

    /*
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
            over_axis3); */

    unsigned long SPT,TGC,TGO,TGPB,TPG;
    unsigned int BLOCK_SIZE = 512;

    //Dim3 slice per thread
    SPT = 2048; //cte

    //thread group offset
    TGO = _dim3 * SPT;

    //thread group count
    TGC = (unsigned long)((_dim0*_dim1*_dim2+(SPT-1))/SPT);

    //thread group per block
    TGPB = (unsigned long)((BLOCK_SIZE)/ _dim3);
    if(TGPB%2 && TGPB > 1) TGPB--;

    TPG = (unsigned long)_dim3; //threads per group

    // Fill the buffer with the initial value
    cl_float initlValue = 0;
    err = clEnqueueFillBuffer(
            queue,
            rsltTn->ocl_buff,
            &initlValue,
            sizeof(cl_float),
            0,
            (size_t)(rsltTn->getLength()*sizeof(cl_float)),
            0,
            NULL,
            NULL);

    if(err != CL_SUCCESS) {
        perror("Couldn't fill a buffer object");
        exit(1);
    }
    clFinish(queue);

    /*kernel_reduce_sum_4d_try04 <<<grid, block, TGPB*TPG*sizeof(float)>>> (
            g_idata, g_buffer, g_odata, 1,
                    dim0, dim1, dim2, dim3,
                    overaxis0, overaxis1, overaxis2, overaxis3,
                    TGC,
                    TGPB,
                    SPT,
                    TGO
    );*/

    size_t global_work_size[] = {(( TGC+(TGPB-1) ) / TGPB)*(BLOCK_SIZE)};
    size_t global_padded_work_size[1];
    size_t local_block_size[] = {BLOCK_SIZE};
    unsigned long shared_mem_size = (TGPB*TPG)*sizeof(cl_float);
    GetPaddedWorkSize(1, local_block_size, global_work_size, global_padded_work_size);


    cout<< "LOCAL:      " << local_block_size[0] << "\n";
    cout<< "GLOBAL:     " << global_work_size[0] << "\n";
    cout<< "GLOBAL_PAD: " << global_padded_work_size[0] << "\n";

    cl_int error;
    cl_int pow_y = 1;
    cl_int _overAxis0,_overAxis1,_overAxis2,_overAxis3;
    _overAxis0 = over_axis0;
    _overAxis1 = over_axis1;
    _overAxis2 = over_axis2;
    _overAxis3 = over_axis3;
    error =  clSetKernelArg(kernelObject->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2, shared_mem_size, NULL);
    error |= clSetKernelArg(kernelObject->kernel, 3, sizeof(cl_int), (void*)&pow_y);

    error |= clSetKernelArg(kernelObject->kernel, 4, sizeof(cl_uint), (void*)&_dim0);
    error |= clSetKernelArg(kernelObject->kernel, 5, sizeof(cl_uint), (void*)&_dim1);
    error |= clSetKernelArg(kernelObject->kernel, 6, sizeof(cl_uint), (void*)&_dim2);
    error |= clSetKernelArg(kernelObject->kernel, 7, sizeof(cl_uint), (void*)&_dim3);

    error |= clSetKernelArg(kernelObject->kernel, 8, sizeof(cl_int), (void*)&_overAxis0);
    error |= clSetKernelArg(kernelObject->kernel, 9, sizeof(cl_int), (void*)&_overAxis1);
    error |= clSetKernelArg(kernelObject->kernel, 10, sizeof(cl_int), (void*)&_overAxis2);
    error |= clSetKernelArg(kernelObject->kernel, 11, sizeof(cl_int), (void*)&_overAxis3);

    error |= clSetKernelArg(kernelObject->kernel, 12, sizeof(cl_ulong), (void*)&TGC);
    error |= clSetKernelArg(kernelObject->kernel, 13, sizeof(cl_ulong), (void*)&TGPB);
    error |= clSetKernelArg(kernelObject->kernel, 14, sizeof(cl_ulong), (void*)&SPT);
    error |= clSetKernelArg(kernelObject->kernel, 15, sizeof(cl_ulong), (void*)&TGO);
    error |= clSetKernelArg(kernelObject->kernel, 16, sizeof(cl_ulong), (void*)&(global_work_size[0]) );


    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject->kernel,
                                   1,
                                   NULL,
                                   global_padded_work_size,
                                   local_block_size,
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

TensorF* OclImplementation::Mean(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3){

    PrintInfo("Mean","",0,"",0,"",0,inputTn->getShape(),{},{mean_axis0,mean_axis1,mean_axis2,mean_axis3});
    assert(mean_axis0&&mean_axis1&&mean_axis2&&!mean_axis3); // TTTF ONLY

    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    OclTensorF* tmpTn = new OclTensorF(context, {_dim3}); // TTTF
    OclTensorF* rsltTn = new OclTensorF(context, {_dim3}); // TTTF
    OclKernelObject *kernelObject_reduction = oclKernels[10];
    OclKernelObject *kernelObject_coef = oclKernels[11];

    unsigned long SPT,TGC,TGO,TGPB,TPG;
    unsigned int BLOCK_SIZE = 512;

    //Dim3 slice per thread
    SPT = 2048; //cte

    //thread group offset
    TGO = _dim3 * SPT;

    //thread group count
    TGC = (unsigned long)((_dim0*_dim1*_dim2+(SPT-1))/SPT);

    //thread group per block
    TGPB = (unsigned long)((BLOCK_SIZE)/ _dim3);
    if(TGPB%2 && TGPB > 1) TGPB--;

    TPG = (unsigned long)_dim3; //threads per group

    // Fill the buffer with the initial value
    cl_float initlValue = 0;
    err = clEnqueueFillBuffer(
            queue,
            tmpTn->ocl_buff,
            &initlValue,
            sizeof(cl_float),
            0,
            (size_t)(tmpTn->getLength()*sizeof(cl_float)),
            0,
            NULL,
            NULL);

    if(err != CL_SUCCESS) {
        perror("Couldn't fill a buffer object");
        exit(1);
    }
    clFinish(queue);

    size_t global_work_size[] = {(( TGC+(TGPB-1) ) / TGPB)*(BLOCK_SIZE)};
    size_t global_padded_work_size[1];
    size_t local_block_size[] = {BLOCK_SIZE};
    unsigned long shared_mem_size = (TGPB*TPG)*sizeof(cl_float);
    GetPaddedWorkSize(1, local_block_size, global_work_size, global_padded_work_size);


    cout<< "LOCAL:      " << local_block_size[0] << "\n";
    cout<< "GLOBAL:     " << global_work_size[0] << "\n";
    cout<< "GLOBAL_PAD: " << global_padded_work_size[0] << "\n";

    cl_int error;
    cl_int pow_y = 1;
    cl_int _overAxis0,_overAxis1,_overAxis2,_overAxis3;
    _overAxis0 = mean_axis0;
    _overAxis1 = mean_axis1;
    _overAxis2 = mean_axis2;
    _overAxis3 = mean_axis3;
    error =  clSetKernelArg(kernelObject_reduction->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)tmpTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 2, shared_mem_size, NULL);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 3, sizeof(cl_int), (void*)&pow_y);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 4, sizeof(cl_uint), (void*)&_dim0);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 5, sizeof(cl_uint), (void*)&_dim1);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 6, sizeof(cl_uint), (void*)&_dim2);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 7, sizeof(cl_uint), (void*)&_dim3);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 8, sizeof(cl_int), (void*)&_overAxis0);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 9, sizeof(cl_int), (void*)&_overAxis1);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 10, sizeof(cl_int), (void*)&_overAxis2);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 11, sizeof(cl_int), (void*)&_overAxis3);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 12, sizeof(cl_ulong), (void*)&TGC);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 13, sizeof(cl_ulong), (void*)&TGPB);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 14, sizeof(cl_ulong), (void*)&SPT);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 15, sizeof(cl_ulong), (void*)&TGO);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 16, sizeof(cl_ulong), (void*)&(global_work_size[0]) );


    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject_reduction->kernel,
                                   1,
                                   NULL,
                                   global_padded_work_size,
                                   local_block_size,
                                   0,
                                   NULL,
                                   &exeEvt);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    clWaitForEvents(1, &exeEvt);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }





    // Launching coef kernel

    unsigned long len = _dim3; //Axes combination is TTTF
    float coef = (_dim0*_dim1*_dim2);
    global_work_size[0] = len;

    error =  clSetKernelArg(kernelObject_coef->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)tmpTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_coef->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_coef->kernel, 2, sizeof(cl_ulong), (void*)&len);
    error |= clSetKernelArg(kernelObject_coef->kernel, 3, sizeof(cl_float), (void*)&coef);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt2;
    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject_coef->kernel,
                                   1,
                                   NULL,
                                   global_work_size,
                                   NULL,
                                   0,
                                   NULL,
                                   &exeEvt2);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    clWaitForEvents(1, &exeEvt2);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }

    delete(tmpTn);
    return rsltTn;
}

TensorF* OclImplementation::Variance(
        WorkScheduler scheduler,
        TensorF* inputTn,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3){
    PrintInfo("Variance","",0,"",0,"",0,inputTn->getShape(),{},{variance_axis0,variance_axis1,variance_axis2,variance_axis3});

    assert(variance_axis0&&variance_axis1&&variance_axis2&&!variance_axis3); // TTTF ONLY

    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    OclTensorF* tmpTn = new OclTensorF(context, {_dim3}); // TTTF
    OclTensorF* medianTn = new OclTensorF(context, {_dim3}); // TTTF
    OclTensorF* varianceXi2Tn = new OclTensorF(context, {_dim3}); // TTTF
    OclTensorF* rsltTn = new OclTensorF(context, {_dim3}); // TTTF
    OclKernelObject *kernelObject_reduction = oclKernels[10];
    OclKernelObject *kernelObject_coef = oclKernels[11];
    OclKernelObject *kernelObject_coef2 = oclKernels[12];

    unsigned long SPT,TGC,TGO,TGPB,TPG;
    unsigned int BLOCK_SIZE = 512;

    //Dim3 slice per thread
    SPT = 512; //cte

    //thread group offset
    TGO = _dim3 * SPT;

    //thread group count
    TGC = (unsigned long)((_dim0*_dim1*_dim2+(SPT-1))/SPT);

    //thread group per block
    TGPB = (unsigned long)((BLOCK_SIZE)/ _dim3);
    if(TGPB%2 && TGPB > 1) TGPB--;

    TPG = (unsigned long)_dim3; //threads per group

    // Fill the buffer with the initial value
    cl_float initlValue = 0;
    err = clEnqueueFillBuffer(
            queue,
            tmpTn->ocl_buff,
            &initlValue,
            sizeof(cl_float),
            0,
            (size_t)(tmpTn->getLength()*sizeof(cl_float)),
            0,
            NULL,
            NULL);




    ///TODO: FILL OTHER TENSORS TOO




    if(err != CL_SUCCESS) {
        perror("Couldn't fill a buffer object");
        exit(1);
    }
    clFinish(queue);

    size_t global_work_size[] = {(( TGC+(TGPB-1) ) / TGPB)*(BLOCK_SIZE)};
    size_t global_padded_work_size[1];
    size_t local_block_size[] = {BLOCK_SIZE};
    unsigned long shared_mem_size = (TGPB*TPG)*sizeof(cl_float);
    GetPaddedWorkSize(1, local_block_size, global_work_size, global_padded_work_size);


    cout<< "LOCAL:      " << local_block_size[0] << "\n";
    cout<< "GLOBAL:     " << global_work_size[0] << "\n";
    cout<< "GLOBAL_PAD: " << global_padded_work_size[0] << "\n";

    cl_int error;
    cl_int pow_y = 1;
    cl_int _overAxis0,_overAxis1,_overAxis2,_overAxis3;
    _overAxis0 = variance_axis0;
    _overAxis1 = variance_axis1;
    _overAxis2 = variance_axis2;
    _overAxis3 = variance_axis3;
    error =  clSetKernelArg(kernelObject_reduction->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)tmpTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 2, shared_mem_size, NULL);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 3, sizeof(cl_int), (void*)&pow_y);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 4, sizeof(cl_uint), (void*)&_dim0);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 5, sizeof(cl_uint), (void*)&_dim1);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 6, sizeof(cl_uint), (void*)&_dim2);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 7, sizeof(cl_uint), (void*)&_dim3);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 8, sizeof(cl_int), (void*)&_overAxis0);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 9, sizeof(cl_int), (void*)&_overAxis1);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 10, sizeof(cl_int), (void*)&_overAxis2);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 11, sizeof(cl_int), (void*)&_overAxis3);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 12, sizeof(cl_ulong), (void*)&TGC);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 13, sizeof(cl_ulong), (void*)&TGPB);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 14, sizeof(cl_ulong), (void*)&SPT);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 15, sizeof(cl_ulong), (void*)&TGO);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 16, sizeof(cl_ulong), (void*)&(global_work_size[0]) );


    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt,exeEvt2,exeEvt3,exeEvt4;

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject_reduction->kernel,
                                   1,
                                   NULL,
                                   global_padded_work_size,
                                   local_block_size,
                                   0,
                                   NULL,
                                   &exeEvt);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    clWaitForEvents(1, &exeEvt);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }






    ///TODO: Im not sure if I should repeat setKernelArg for the arguments with same params.
    pow_y = 2;
    error =  clSetKernelArg(kernelObject_reduction->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)varianceXi2Tn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 2, shared_mem_size, NULL);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 3, sizeof(cl_int), (void*)&pow_y);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 4, sizeof(cl_uint), (void*)&_dim0);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 5, sizeof(cl_uint), (void*)&_dim1);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 6, sizeof(cl_uint), (void*)&_dim2);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 7, sizeof(cl_uint), (void*)&_dim3);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 8, sizeof(cl_int), (void*)&_overAxis0);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 9, sizeof(cl_int), (void*)&_overAxis1);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 10, sizeof(cl_int), (void*)&_overAxis2);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 11, sizeof(cl_int), (void*)&_overAxis3);

    error |= clSetKernelArg(kernelObject_reduction->kernel, 12, sizeof(cl_ulong), (void*)&TGC);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 13, sizeof(cl_ulong), (void*)&TGPB);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 14, sizeof(cl_ulong), (void*)&SPT);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 15, sizeof(cl_ulong), (void*)&TGO);
    error |= clSetKernelArg(kernelObject_reduction->kernel, 16, sizeof(cl_ulong), (void*)&(global_work_size[0]) );


    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject_reduction->kernel,
                                   1,
                                   NULL,
                                   global_padded_work_size,
                                   local_block_size,
                                   0,
                                   NULL,
                                   &exeEvt2);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    clWaitForEvents(1, &exeEvt2);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }






    // Launching coef1 kernel
    unsigned long len = _dim3; //Axes combination is TTTF
    float coef = (_dim0*_dim1*_dim2);
    global_work_size[0] = len;

    error =  clSetKernelArg(kernelObject_coef->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)tmpTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_coef->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)medianTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_coef->kernel, 2, sizeof(cl_ulong), (void*)&len);
    error |= clSetKernelArg(kernelObject_coef->kernel, 3, sizeof(cl_float), (void*)&coef);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject_coef->kernel,
                                   1,
                                   NULL,
                                   global_work_size,
                                   NULL,
                                   0,
                                   NULL,
                                   &exeEvt3);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    clWaitForEvents(1, &exeEvt3);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }







    // Launching coef2 kernel
    error =  clSetKernelArg(kernelObject_coef2->kernel, 0, sizeof(cl_mem), (void*)&((OclTensorF*)varianceXi2Tn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_coef2->kernel, 1, sizeof(cl_mem), (void*)&((OclTensorF*)medianTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_coef2->kernel, 2, sizeof(cl_float), (void*)&coef);
    error |= clSetKernelArg(kernelObject_coef2->kernel, 3, sizeof(cl_mem), (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject_coef2->kernel, 4, sizeof(cl_ulong), (void*)&len);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    error = clEnqueueNDRangeKernel(queue,
                                   kernelObject_coef2->kernel,
                                   1,
                                   NULL,
                                   global_work_size,
                                   NULL,
                                   0,
                                   NULL,
                                   &exeEvt4);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    clWaitForEvents(1, &exeEvt4);

    if(error != CL_SUCCESS) {
        printf("Kernel execution failure!\n");
        exit(-22);
    }


    delete(tmpTn);
    delete(varianceXi2Tn);
    delete(medianTn);
    return rsltTn;
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
    assert(inputTn->getRank()==4);

    size_t kGrid;
    int kDim0,kDim1,kDim2;
    int overAxis0, overAxis1, overAxis2;
    unsigned int _dim0,_dim1,_dim2,_dim3;
    _dim0 = inputTn->getShape()[0];
    _dim1 = inputTn->getShape()[1];
    _dim2 = inputTn->getShape()[2];
    _dim3 = inputTn->getShape()[3];

    OclTensorF* rsltTn = nullptr;
    if(inputTn->getRank()==4 &&  reductionDim==1)rsltTn= new OclTensorF(context, {_dim0,_dim2,_dim3});
    if(inputTn->getRank()==4 &&  reductionDim==2)rsltTn= new OclTensorF(context, {_dim0,_dim1,_dim3});

    if(reductionDim==2){
        /*
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
                reductionDim==3); */


        kDim0 = _dim0*_dim1;
        kDim1 = _dim2;
        kDim2 = _dim3;
        kGrid = kDim0*kDim2;

        overAxis0 = 0;
        overAxis1 = 1;
        overAxis2 = 0;


        /*kernel_reduce_max_try01 <<<kGrid, BLOCK_SIZE>>> (
                g_idata, g_odata,
                        kDim2, kDim1, kDim0,
                        overaxis0 && overaxis1, overaxis2,overaxis3);*/

    }

    if(reductionDim==1){
        // for all reductionDim=1 : dim2 is always 1(i.e. tensor is 3 dimensional)
        /* reduce_max_3d_try01(
                inputTn->_buff,
                rsltTn->_buff,
                _dim0,
                _dim1,
                _dim3,
                reductionDim==0,
                reductionDim==1,
                reductionDim==2);*/


        kDim0 = _dim0;
        kDim1 = _dim1;
        kDim2 = _dim3;
        kGrid = kDim0*kDim2;

        overAxis0 = 0;
        overAxis1 = 1;
        overAxis2 = 0;

        /*kernel_reduce_max_try01 <<<kGrid, BLOCK_SIZE>>> (
                g_idata, g_odata,
                        kDim2, kDim1, kDim0,
                        overaxis0 ,overaxis1, overaxis2);*/
    }

    OclKernelObject *kernelObject = oclKernels[8];
    cl_int error;

    error =  clSetKernelArg(kernelObject->kernel, 0 , sizeof(cl_mem) , (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1 , sizeof(cl_mem) , (void*)&((OclTensorF*)rsltTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2 , sizeof(cl_uint) , (void*)&kDim2);
    error |= clSetKernelArg(kernelObject->kernel, 3 , sizeof(cl_uint) , (void*)&kDim1);
    error |= clSetKernelArg(kernelObject->kernel, 4 , sizeof(cl_uint) , (void*)&kDim0);
    error |= clSetKernelArg(kernelObject->kernel, 5 , sizeof(cl_int), (void*)&overAxis0);
    error |= clSetKernelArg(kernelObject->kernel, 6 , sizeof(cl_int), (void*)&overAxis1);
    error |= clSetKernelArg(kernelObject->kernel, 7 , sizeof(cl_int), (void*)&overAxis2);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {16, 16};
    size_t globalThreads[] = {kGrid};

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

    OclTensorF* buffTn = new OclTensorF(context,
                                        {inputTn->getShape()[0],
                                         inputTn->getShape()[1],
                                         inputTn->getShape()[2],
                                         weights->getShape()[3]});



    // Fill the buffer with the initial value
    cl_float initlValue = 0;
    err = clEnqueueFillBuffer(
            queue,
            buffTn->ocl_buff,
            &initlValue,
            sizeof(cl_float),
            0,
            (size_t)(buffTn->getLength()*sizeof(cl_float)),
            0,
            NULL,
            NULL);

    if(err != CL_SUCCESS) {
        perror("Couldn't fill a buffer object");
        exit(1);
    }
    clFinish(queue);




    TensorF* rsltTn ;

    unsigned int B      = inputTn->getShape()[0];
    unsigned int N      = inputTn->getShape()[1];
    unsigned int K      = inputTn->getShape()[2];
    unsigned int D      = inputTn->getShape()[3];
    unsigned int chOut  = weights->getShape()[3];

    assert(D<=1024); // this kernel cannot accept dim3>1024

    OclKernelObject *kernelObject = oclKernels[13];
    cl_int error;

    error =  clSetKernelArg(kernelObject->kernel, 0 , sizeof(cl_mem) , (void*)&((OclTensorF*)inputTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 1 , sizeof(cl_mem) , (void*)&((OclTensorF*)weights)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 2 , sizeof(cl_mem) , (void*)&((OclTensorF*)buffTn)->ocl_buff);
    error |= clSetKernelArg(kernelObject->kernel, 3 , sizeof(cl_uint) , (void*)&B);
    error |= clSetKernelArg(kernelObject->kernel, 4 , sizeof(cl_uint) , (void*)&N);
    error |= clSetKernelArg(kernelObject->kernel, 5 , sizeof(cl_uint) , (void*)&K);
    error |= clSetKernelArg(kernelObject->kernel, 6 , sizeof(cl_uint) , (void*)&D);
    error |= clSetKernelArg(kernelObject->kernel, 7 , sizeof(cl_uint) , (void*)&chOut);

    if(error != CL_SUCCESS) cout<<getErrorString(error)<<endl;
    assert(error==0);

    cl_event exeEvt;
    //unsigned long localThreads[]  = {D};
    size_t globalThreads[] = {B*N*K*D};

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

    rsltTn = MatOps(scheduler,buffTn,biases,MAT_OPS::ADD);
    delete(buffTn);
    return rsltTn;
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
