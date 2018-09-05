//
// Created by saleh on 8/30/18.
//

#include <fstream>
#include "../inc/WeightsLoader.h"

WeightsLoader::WeightsLoader(vector<PLATFORMS> neededPlatforms) {
    for(std::vector<PLATFORMS>::iterator it = neededPlatforms.begin(); it != neededPlatforms.end(); ++it) {
        switch(*it){
            case PLATFORMS::CPU : {
                _isUsedCPU = true;
                break;
            }
            case PLATFORMS::GPU_CUDA : {
                _isUsedCUDA = true;
                break;
            }
            case PLATFORMS::GPU_OCL : {
                _isUsedOCL = true;
                break;
            }
        }
    }
}

void WeightsLoader::LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList) {
    ifstream txtfile (pathToTxtFnameList);

    if (!txtfile.is_open())
    {
        cout<<"Failed to open text file!";
        return;
    }

    string line; int i=0;
    vector<cnpy::NpyArray> _weights_vector;
    while (std::getline(txtfile, line)) {
        //cout<<"Line:"<<i++<<":"<<line<<endl;
        string weight_npy_path = weightsBaseDir + line;
        _weights_vector.push_back(cnpy::npy_load(weight_npy_path));
        //weights_map.insert(std::make_pair(line,_weights_vector.back().data<float>()));
        //weightsshape_map.insert(std::make_pair(line,_weights_vector.back().shape));
        if(_isUsedCPU){
            vector<unsigned int> __shape(_weights_vector.back().shape.begin(),_weights_vector.back().shape.end());
            weightsMapCPU.insert(
                    std::make_pair(
                            line,
                            new TensorF(
                                    __shape,
                                    _weights_vector.back().data<float>()
                            )
                    )
            );
        }
        if(_isUsedCUDA){
            vector<unsigned int> __shape(_weights_vector.back().shape.begin(),_weights_vector.back().shape.end());
            CudaTensorF* cudaWeight = new CudaTensorF();
            cudaWeight->InitWithHostData(__shape,
                                         _weights_vector.back().data<float>());
            weightsMapCUDA.insert(std::make_pair(line, cudaWeight));
        }
        if(_isUsedOCL){
            throw "Not Implemented.";
        }
    }

    txtfile.close();
}

TensorF* WeightsLoader::AccessWeights(PLATFORMS platform, string name) {
    switch(platform){
        case PLATFORMS::CPU:{
            return weightsMapCPU[name];
            break;
        }
        case PLATFORMS::GPU_CUDA:{
            return weightsMapCPU[name];
            break;
        }
        case PLATFORMS::GPU_OCL:{
            throw "Not Implemented.";
            break;
        }
    }
}