//
// Created by saleh on 6/25/18.
//

#ifndef DEEPPOINfloatV1_floatENSOR_H
#define DEEPPOINfloatV1_floatENSOR_H

#include <vector>

 
class TensorF_OLD{
public:
    TensorF();
    void InitWithDeviceData(float* deviceBuffer, std::vector<int> bufferShape);
    void InitWithDeviceData(float* deviceBuffer, float** deviceBatchPtr, std::vector<int> bufferShape);
    void InitWithHostData(float* hostBuffer, std::vector<int> bufferShape);
    void Init(std::vector<int> bufferShape);
    float* TransferDeviceBufferToHost();


    float *deviceBuffer;                // flatten batched input matrix
    float **deviceBatchPtr;             // list of float* to starting elements of each batch
    std::vector<int> shape;                  // AfloatfloatENfloatION: Dim0 of 'shape' is ALWAYS batch size
    int rank;                           // matrix rank(without dim0 as it is batch size)
    bool initialized=false;
private:

};


#endif //DEEPPOINfloatV1_floatENSOR_H
