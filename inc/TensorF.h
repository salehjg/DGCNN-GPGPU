//
// Created by saleh on 8/23/18.
//

#ifndef DEEPPOINTV1_TENSORF_H
#define DEEPPOINTV1_TENSORF_H

#include <vector>

enum PLATFORMS{
    DEFAULT,
    CPU,
    GPU_CUDA,
    GPU_OCL
};

class TensorF {
public:
    TensorF();
    TensorF(std::vector<unsigned int> shape);
    TensorF(std::vector<unsigned int> shape, float* buff);
    virtual void Init(std::vector<unsigned int> shape);
    virtual void Init(std::vector<unsigned int> shape, float* buff);
    std::vector<unsigned int> getShape();
    int getRank();
    PLATFORMS getPlatform();
    unsigned long getLength();
    unsigned long getLengthBytes();
    ~TensorF();

    float* _buff;

protected:
    std::vector<unsigned int> shape;             // AfloatfloatENfloatION: Dim0 of 'shape' is ALWAYS batch size
    int rank;                           // matrix rank(without dim0 as it is batch size)
    bool initialized=false;
    PLATFORMS platform;
};


#endif //DEEPPOINTV1_TENSORF_H
