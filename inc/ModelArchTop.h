//
// Created by saleh on 8/28/18.
//

#ifndef DEEPPOINTV1_MODELARCHTOP_H
#define DEEPPOINTV1_MODELARCHTOP_H

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
#undef DUMP_ENABLED

struct ModelInfo{
    string Version="";
    string ModelType="";
    string DesignNotes="";
    string ExperimentNotes="";
    string ToDo="";
    string Date="";
};

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}


class ModelArchTop {
public:
    ModelArchTop(int dataset_offset, int batchsize, int pointcount, int knn_k);
    ModelInfo   GetModelInfo();
    void        SetModelInput_data(string npy_pcl);
    void        SetModelInput_labels(string npy_labels);
    TensorF*    FullyConnected_Forward(WorkScheduler scheduler, TensorF* input_BxD, TensorF* weights, TensorF* biases);
    TensorF*    Batchnorm_Forward(WorkScheduler scheduler, TensorF* input, TensorF* gamma, TensorF* beta, TensorF* ema_ave, TensorF* ema_var);
    TensorF*    GetEdgeFeatures(WorkScheduler scheduler, TensorF* input_BxNxD, TensorI* knn_output_BxNxK);
    TensorF*    PairwiseDistance(WorkScheduler scheduler, TensorF* input_BxNxD);
    TensorF*    TransformNet(WorkScheduler scheduler, TensorF* edgeFeatures);
    TensorF*    Execute(WorkScheduler scheduler);

private:
    unsigned int B=-1;
    unsigned int N=-1;
    unsigned int K=-1;
    int DB_OFFSET=-1;
    TensorF* input_pcl_BxNxD;
    TensorI* input_labels_B; ///TODO: CHANGE TYPE
    cnpy::NpyArray _npy_pcl;
    cnpy::NpyArray _npy_labels;
    PlatformSelector* platformSelector;
};


#endif //DEEPPOINTV1_MODELARCHTOP_H
