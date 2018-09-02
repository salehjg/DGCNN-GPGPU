//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_CUDATENSORF_H
#define DEEPPOINTV1_CUDATENSORF_H

#include "../../inc/TensorF.h"

class CudaTensorF: public TensorF {
public:
    CudaTensorF();

    CudaTensorF(std::vector<unsigned int> shape);
    CudaTensorF(std::vector<unsigned int> shape,float *buff);
    void Init(std::vector<unsigned int> shape);
    void Init(std::vector<unsigned int> shape, float* buff);
    void InitWithHostData(std::vector<unsigned int> shape,float* hostBuff);
    TensorF* TransferToHost();
    ~CudaTensorF();

    float** ConvertDevice1D_to_Device2D(float* input_batched_1d, int batchsize, int perBatchLen);
    float** ConvertDevice1D_to_Device2D(float* input_batched_1d, std::vector<int> shape);
    float** ConvertHost1D_to_Device2D(float* input_batched_1d, float** outDevice1D, int batchsize, int perBatchLen);
    float** ConvertHost1D_to_Device2D(float* input_batched_1d, float** outDevice1D, std::vector<int> shape);

private:


};


#endif //DEEPPOINTV1_CUDATENSORF_H
