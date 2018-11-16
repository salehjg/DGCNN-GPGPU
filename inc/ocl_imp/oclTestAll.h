//
// Created by saleh on 8/28/18.
//

#ifndef DEEPPOINTV1_OCLTESTALL_H
#define DEEPPOINTV1_OCLTESTALL_H

#include <iostream>
#include <sys/time.h>
#include "../../submodules/cnpy/cnpy.h"
#include "../inc/TensorF.h"
#include "../inc/TensorI.h"
#include "../inc/cuda_imp/CudaTensorF.h"
#include "../inc/cuda_imp/CudaTensorI.h"
#include "../inc/PlatformSelector.h"
#include "../inc/PlatformImplementation.h"


using namespace std;

//#define DUMP_ENABLED
//#undef DUMP_ENABLED


inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


class OclTestAll {
public:
    OclTestAll(int dataset_offset, int batchsize, int pointcount, int knn_k);
    void        SetModelInput_data(string npy_pcl);
    void        SetModelInput_labels(string npy_labels);
    TensorF*    Execute(WorkScheduler scheduler);
    TensorI*    GetLabels();
    float float_rand( float min, float max );
    TensorF* GenerateTensor(int pattern, vector<unsigned int> shape);
    TensorI* GenerateTensor(int intMin, int intMax, vector<unsigned int> shape);
    unsigned int B=-1;
    unsigned int N=-1;
    unsigned int K=-1;
    TensorF* input_pcl_BxNxD;
    TensorI* input_labels_B;
    int DB_OFFSET=-1;
    cnpy::NpyArray _npy_pcl;
    cnpy::NpyArray _npy_labels;
    PlatformSelector* platformSelector;
};


#endif //DEEPPOINTV1_OCLTESTALL_H
