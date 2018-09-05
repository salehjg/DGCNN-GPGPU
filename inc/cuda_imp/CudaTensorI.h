//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_CUDATENSORI_H
#define DEEPPOINTV1_CUDATENSORI_H

#include "../../inc/TensorF.h" //for platforms enum!
#include "../../inc/TensorI.h"

class CudaTensorI: public TensorI {
public:
    CudaTensorI();

    CudaTensorI(std::vector<unsigned int> shape);
    CudaTensorI(std::vector<unsigned int> shape,int *buff);
    void Init(std::vector<unsigned int> shape);
    void Init(std::vector<unsigned int> shape, int* buff);
    void InitWithHostData(std::vector<unsigned int> shape,int* hostBuff);
    TensorI* TransferToHost();
    ~CudaTensorI();

    int** ConvertDevice1D_to_Device2D(int* input_batched_1d, int batchsize, int perBatchLen);
    int** ConvertDevice1D_to_Device2D(int* input_batched_1d, std::vector<int> shape);
    int** ConvertHost1D_to_Device2D(int* input_batched_1d, int** outDevice1D, int batchsize, int perBatchLen);
    int** ConvertHost1D_to_Device2D(int* input_batched_1d, int** outDevice1D, std::vector<int> shape);

private:


};


#endif //DEEPPOINTV1_CUDATENSORI_H
