//
// Created by saleh on 8/30/18.
//

#ifndef DEEPPOINTV1_WEIGHTSLOADER_H
#define DEEPPOINTV1_WEIGHTSLOADER_H

#include "../inc/TensorF.h"
#include "../inc/cuda_imp/CudaTensorF.h"
#include "../../submodules/cnpy/cnpy.h"
#include <vector>
using namespace std;

class WeightsLoader {
public:
    WeightsLoader(vector<PLATFORMS> neededPlatforms);
    void LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList);
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
