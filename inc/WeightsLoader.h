//
// Created by saleh on 8/30/18.
//

#ifndef DEEPPOINTV1_WEIGHTSLOADER_H
#define DEEPPOINTV1_WEIGHTSLOADER_H

#include <build_config.h>
#include "../inc/TensorF.h"

#ifdef USE_CUDA
#include "../inc/cuda_imp/CudaTensorF.h"
#endif

#ifdef USE_OCL
#include "../inc/ocl_imp/OclTensorF.h"
#endif

#include "../../submodules/cnpy/cnpy.h"
#include <vector>
using namespace std;

#ifndef USE_OCL
    struct cl_context{

    };
    struct cl_command_queue{

    };
#endif

class WeightsLoader {
public:
    WeightsLoader(vector<PLATFORMS> neededPlatforms);
    void LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList, cl_context oclContex, cl_command_queue oclQueue);
    TensorF* AccessWeights(PLATFORMS platform, string name);

private:
    map<string,int> strToIndexMap;
    vector<cnpy::NpyArray> _cnpyBuff;
    TensorF** weightsCPU;
    TensorF** weightsCUDA;
    TensorF** weightsOCL;

    bool _isUsedCPU  = false;
    bool _isUsedCUDA = false;
    bool _isUsedOCL  = false;
    ///TODO: Implement weightMapOCL

};


#endif //DEEPPOINTV1_WEIGHTSLOADER_H
