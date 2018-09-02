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
    void LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList)
    TensorF* AccessWeights(PLATFORMS platform, string name);
private:
    map<string,TensorF*> weightsMapCPU;
    map<string,TensorF*> weightsMapCUDA;
    bool _isUsedCPU  = false;
    bool _isUsedCUDA = false;
    bool _isUsedOCL  = false;
    ///TODO: Implement weightMapOCL

};


#endif //DEEPPOINTV1_WEIGHTSLOADER_H
