//
// Created by saleh on 8/22/18.
//

#include "../inc/PlatformSelector.h"
#include "../inc/cpu_imp/CpuImplementation.h"

#ifdef USE_CUDA
#include <cuda_imp/CudaMemHelper.h>
#include <cuda_imp/common.h>
#include "../inc/cuda_imp/CudaImplementation.h"
#include "../inc/cuda_imp/CudaTensorF.h"
#include "../inc/cuda_imp/CudaTensorI.h"
#endif

#ifdef USE_OCL
#include <CL/cl.h>
#include "../inc/ocl_imp/OclImplementation.h"
#include "../inc/ocl_imp/OclTensorF.h"
#include "../inc/ocl_imp/OclTensorI.h"
#endif

PlatformSelector::PlatformSelector(PLATFORMS defaultPlatform, vector<PLATFORMS> neededPlatforms) {
    this->defaultPlatform = defaultPlatform;

    for(std::vector<PLATFORMS>::iterator it = neededPlatforms.begin(); it != neededPlatforms.end(); ++it) {
        switch(*it){
            case PLATFORMS::CPU : {
                cpuPlatformClass = new CpuImplementation();
                break;
            }
#ifdef USE_CUDA
            case PLATFORMS::GPU_CUDA : {
                cudaPlatformClass = new CudaImplementation(11);
                break;
            }
#endif
#ifdef USE_OCL
            case PLATFORMS::GPU_OCL : {
                openclPlatformClass = new OclImplementation(11);
                break;
            }
#endif
        }
    }

    weightsLoader = new WeightsLoader(neededPlatforms);
#ifdef USE_OCL
    weightsLoader->LoadFromDisk(REPO_DIR "/data/weights/",
                                REPO_DIR "/data/weights/filelist.txt",
                                openclPlatformClass->getContext(),
                                openclPlatformClass->getQueue());
#else
    weightsLoader->LoadFromDisk(REPO_DIR "/data/weights/",
                                REPO_DIR "/data/weights/filelist.txt" );
#endif
}

TensorF* PlatformSelector::CrossThePlatform(TensorF *srcTn, PLATFORMS platform) {
    switch(srcTn->getPlatform()){
        case PLATFORMS::CPU :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    return srcTn;
                    break;
                }
#ifdef USE_CUDA
                case PLATFORMS::GPU_CUDA :{
                    CudaTensorF *rsltTn = new CudaTensorF();
                    rsltTn->InitWithHostData(srcTn->getShape(),srcTn->_buff);
                    return rsltTn;
                    break;
                }
#endif
#ifdef USE_OCL
                case PLATFORMS::GPU_OCL :{
                    OclTensorF *rsltTn = new OclTensorF();
                    rsltTn->InitWithHostData(openclPlatformClass->getContext(), openclPlatformClass->getQueue(), srcTn->getShape(),srcTn->_buff);
                    return rsltTn;
                    break;
                }
#endif
            }
            break;

        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    CudaTensorF * srcTensor = (CudaTensorF*)srcTn;
                    TensorF* rsltTn = srcTensor->TransferToHost();
                    return rsltTn;
                    break;
                }
                case PLATFORMS::GPU_CUDA :{
                    return srcTn;
                    break;
                }
                case PLATFORMS::GPU_OCL :{
                    throw "Not Implemented.";
                    break;
                }
            }
            break;

        }
#endif

#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    OclTensorF* srcTensor = (OclTensorF*)srcTn;
                    TensorF* rsltTn = srcTensor->TransferToHost(openclPlatformClass->getQueue());
                    return rsltTn;
                    break;
                }
                case PLATFORMS::GPU_CUDA :{
                    throw "Not Implemented.";
                    break;
                }
                case PLATFORMS::GPU_OCL :{
                    return srcTn;
                    break;
                }
            }
            break;
        }
#endif
    }
}


TensorI* PlatformSelector::CrossThePlatform(TensorI *srcTn, PLATFORMS platform) {
    switch(srcTn->getPlatform()){
        case PLATFORMS::CPU :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    return srcTn;
                    break;
                }
#ifdef USE_CUDA
                case PLATFORMS::GPU_CUDA :{
                    CudaTensorI *rsltTn = new CudaTensorI();
                    rsltTn->InitWithHostData(srcTn->getShape(),srcTn->_buff);
                    return rsltTn;
                    break;
                }
#endif
#ifdef USE_OCL
                case PLATFORMS::GPU_OCL :{
                    OclTensorI *rsltTn = new OclTensorI();
                    rsltTn->InitWithHostData(openclPlatformClass->getContext(), openclPlatformClass->getQueue(), srcTn->getShape(),srcTn->_buff);
                    return rsltTn;
                }
#endif
            }
            break;

        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    CudaTensorI * srcTensor = (CudaTensorI*)srcTn;
                    return srcTensor->TransferToHost();
                    break;
                }
                case PLATFORMS::GPU_CUDA :{
                    return srcTn;
                    break;
                }
                case PLATFORMS::GPU_OCL :{
                    throw "Not Implemented.";
                    break;
                }
            }
            break;

        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            //--------------------------------------------------------------
            switch(platform){
                case PLATFORMS::CPU :{
                    OclTensorI* srcTensor = (OclTensorI*)srcTn;
                    TensorI* rsltTn = srcTensor->TransferToHost(openclPlatformClass->getQueue());
                    return rsltTn;
                    break;
                }
                case PLATFORMS::GPU_CUDA :{
                    throw "Not Implemented.";
                    break;
                }
                case PLATFORMS::GPU_OCL :{
                    return srcTn;
                    break;
                }
            }
            break;
        }
#endif
    }
}

TensorF* PlatformSelector::Transpose(PLATFORMS platform, WorkScheduler scheduler, TensorF *batchedMat){
    TensorF *__batchedMat = CrossThePlatform(batchedMat, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Transpose(scheduler, __batchedMat);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Transpose(scheduler,__batchedMat);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Transpose(scheduler,__batchedMat);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::MatMul(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat1, TensorF* batchedMat2){
    TensorF* __batchedMat1 = CrossThePlatform(batchedMat1, platform);
    TensorF* __batchedMat2 = CrossThePlatform(batchedMat2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatMul(scheduler, __batchedMat1, __batchedMat2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->MatMul(scheduler, __batchedMat1, __batchedMat2);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->MatMul(scheduler, __batchedMat1, __batchedMat2);
            break;
        }
#endif
    }
    return nullptr;
}
/*
TensorF* PlatformSelector::MatMul(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat, float scalar){
    TensorF* __batchedMat = CrossThePlatform(batchedMat, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatMul(scheduler, __batchedMat, scalar);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}
*/

TensorF* PlatformSelector::Square(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat){
    TensorF* __batchedMat = CrossThePlatform(batchedMat, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Square(scheduler, __batchedMat);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Square(scheduler, __batchedMat);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Square(scheduler, __batchedMat);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::ReduceSum(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->ReduceSum(scheduler, __inputTn, over_axis0, over_axis1, over_axis2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            TensorF* rlstTn = cudaPlatformClass->ReduceSum(scheduler, __inputTn,over_axis0, over_axis1, over_axis2);
            return rlstTn;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->ReduceSum(scheduler, __inputTn, over_axis0, over_axis1, over_axis2);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::ReduceSum4D(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool over_axis0, bool over_axis1, bool over_axis2, bool over_axis3){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->ReduceSum4D(scheduler, __inputTn, over_axis0, over_axis1, over_axis2, over_axis3);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->ReduceSum4D(scheduler, __inputTn,over_axis0, over_axis1, over_axis2, over_axis3);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->ReduceSum4D(scheduler, __inputTn, over_axis0, over_axis1, over_axis2, over_axis3);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Mean(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool mean_axis0, bool mean_axis1, bool mean_axis2, bool mean_axis3){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Mean(scheduler, __inputTn, mean_axis0, mean_axis1, mean_axis2, mean_axis3);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Mean(scheduler, __inputTn, mean_axis0, mean_axis1, mean_axis2, mean_axis3);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Mean(scheduler, __inputTn, mean_axis0, mean_axis1, mean_axis2, mean_axis3);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Variance(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, bool variance_axis0, bool variance_axis1, bool variance_axis2, bool variance_axis3){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Variance(scheduler, __inputTn, variance_axis0, variance_axis1, variance_axis2, variance_axis3);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Variance(scheduler, __inputTn, variance_axis0, variance_axis1, variance_axis2, variance_axis3);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Variance(scheduler, __inputTn, variance_axis0, variance_axis1, variance_axis2, variance_axis3);
            break;
        }
#endif
    }
    return nullptr;
}

/*
TensorF* PlatformSelector::MatAdd(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatAdd(scheduler, __inputTn1, __inputTn2);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}

TensorF* PlatformSelector::MatSub(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatSub(scheduler, __inputTn1, __inputTn2);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}

TensorF* PlatformSelector::MatAddTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputSmallTn2 = CrossThePlatform(inputSmallTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatAddTiled(scheduler, __inputTn1, __inputSmallTn2);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}

TensorF* PlatformSelector::MatSubTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputSmallTn2){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputSmallTn2 = CrossThePlatform(inputSmallTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatSubTiled(scheduler, __inputTn1, __inputSmallTn2);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}
 */
/*
TensorF* PlatformSelector::MatAddTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, float scalar){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatAddTiled(scheduler, __inputTn1, scalar);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}
*/
/*
TensorF* PlatformSelector::MatSubTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, float scalar){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatSubTiled(scheduler, __inputTn1, scalar);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}
*/

TensorF* PlatformSelector::MatOps(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2, MAT_OPS mode) {
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatOps(scheduler,__inputTn1,__inputTn2,mode);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->MatOps(scheduler,__inputTn1,__inputTn2,mode);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->MatOps(scheduler,__inputTn1,__inputTn2,mode);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::MatOps(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, float scalar, MAT_OPS mode) {
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MatOps(scheduler,__inputTn1,scalar,mode);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->MatOps(scheduler,__inputTn1,scalar,mode);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->MatOps(scheduler,__inputTn1,scalar,mode);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Sqrt(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Sqrt(scheduler, __inputTn);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Sqrt(scheduler, __inputTn);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Sqrt(scheduler, __inputTn);
            break;
        }
#endif
    }
    return nullptr;
}

/*
TensorF* PlatformSelector::Multiply(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Multiply(scheduler, __inputTn1, __inputTn2);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}

TensorF* PlatformSelector::Divide(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Divide(scheduler, __inputTn1, __inputTn2);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}


TensorF* PlatformSelector::MultiplyTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->MultiplyTiled(scheduler, __inputTn1, __inputTn2);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}

TensorF* PlatformSelector::DivideTiled(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->DivideTiled(scheduler, __inputTn1, __inputTn2);
            break;
        }
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
    }
    return nullptr;
}
*/

TensorF* PlatformSelector::Concat2(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn1, TensorF* inputTn2, int concatDim){
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Concat2(scheduler, __inputTn1, __inputTn2, concatDim);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            TensorF* rlstTn = cudaPlatformClass->Concat2(scheduler, __inputTn1, __inputTn2, concatDim);
            return rlstTn;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Concat2(scheduler, __inputTn1, __inputTn2, concatDim);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::ReduceMax(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, int reductionDim){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->ReduceMax(scheduler, __inputTn, reductionDim);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            TensorF* rlstTn = cudaPlatformClass->ReduceMax(scheduler, __inputTn,reductionDim);
            return rlstTn;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->ReduceMax(scheduler, __inputTn, reductionDim);
            break;
        }
#endif
    }
    return nullptr;
}


TensorI* PlatformSelector::TopK(PLATFORMS platform, WorkScheduler scheduler, TensorF* batchedMat, int axis, int k){
    TensorF* __batchedMat = CrossThePlatform(batchedMat, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->TopK(scheduler, __batchedMat, axis, k);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->TopK(scheduler, __batchedMat, axis, k);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->TopK(scheduler, __batchedMat, axis, k);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Gather(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, TensorI* indices, int indices_axis){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    TensorI* __indices = CrossThePlatform(indices, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Gather(scheduler, __inputTn, __indices, indices_axis);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Gather(scheduler, __inputTn, __indices, indices_axis);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Gather(scheduler, __inputTn, __indices, indices_axis);
            break;
        }
#endif
    }
    return nullptr;
}


TensorF* PlatformSelector::Conv2D(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn, TensorF* weights, TensorF* biases, int overrideDim2){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    TensorF* __weights = CrossThePlatform(weights, platform);
    TensorF* __biases = CrossThePlatform(biases, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Conv2D(scheduler, __inputTn, __weights, __biases, overrideDim2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->Conv2D(scheduler, __inputTn, __weights, __biases, overrideDim2);;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Conv2D(scheduler, __inputTn, __weights, __biases, overrideDim2);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::ReLU(PLATFORMS platform, WorkScheduler scheduler, TensorF* inputTn){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->ReLU(scheduler, __inputTn);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            return cudaPlatformClass->ReLU(scheduler, __inputTn);
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->ReLU(scheduler, __inputTn);
            break;
        }
#endif
    }
    return nullptr;
}

TensorF* PlatformSelector::Tile(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn, int tileAxis, int tileCount){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->Tile(scheduler, __inputTn, tileAxis, tileCount);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            TensorF* rlstTn = cudaPlatformClass->Tile(scheduler, __inputTn, tileAxis, tileCount);
            return rlstTn;
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            return openclPlatformClass->Tile(scheduler, __inputTn, tileAxis, tileCount);
            break;
        }
#endif
    }
    return nullptr;
}

void PlatformSelector::DumpMatrix(PLATFORMS platform, WorkScheduler scheduler, string npy_fname, TensorF* inputTn, string npy_dir){
    TensorF* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->DumpMatrix(scheduler, npy_fname,__inputTn,npy_dir);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            TensorF* __inputTn2 = CrossThePlatform(inputTn, PLATFORMS::CPU);
            return cpuPlatformClass->DumpMatrix(scheduler, npy_fname,__inputTn2,npy_dir);
            break;
        }
#endif
    }
    return;
}

void PlatformSelector::DumpMatrix(PLATFORMS platform, WorkScheduler scheduler, string npy_fname, TensorI* inputTn, string npy_dir){
    TensorI* __inputTn = CrossThePlatform(inputTn, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->DumpMatrix(scheduler, npy_fname,__inputTn,npy_dir);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            TensorI* __inputTn2 = CrossThePlatform(inputTn, PLATFORMS::CPU);
            return cpuPlatformClass->DumpMatrix(scheduler, npy_fname,__inputTn2,npy_dir);
            break;
        }
#endif
    }
    return;
}

bool PlatformSelector::CompareTensors(PLATFORMS platform, WorkScheduler scheduler, TensorF *inputTn1, TensorF *inputTn2) {
    TensorF* __inputTn1 = CrossThePlatform(inputTn1, platform);
    TensorF* __inputTn2 = CrossThePlatform(inputTn2, platform);
    switch(platform){
        case PLATFORMS::CPU :{
            return cpuPlatformClass->CompareTensors(scheduler, __inputTn1,__inputTn2);
            break;
        }
#ifdef USE_CUDA
        case PLATFORMS::GPU_CUDA :{
            throw "Not Implement.";
            break;
        }
#endif
#ifdef USE_OCL
        case PLATFORMS::GPU_OCL :{
            throw "Not Implement.";
            break;
        }
#endif
    }
    return false;
}
