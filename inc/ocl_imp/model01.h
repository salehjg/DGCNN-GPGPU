//
// Created by saleh on 5/31/18.
//

#ifndef DEEPPOINTV1_MODEL01_H
#define DEEPPOINTV1_MODEL01_H

#include <string>
using namespace std;

struct ModelInfo{
    string Version="";
    string DesignNotes="";
    string ExperimentNotes="";
    string ToDo="";
    string Date="";
};

class ModelArch {
public:
    ModelArch(int batchsize, int pointcount, int knn_k);
    ModelInfo GetModelInfo();
    void SetModelInput(float *points_BxNx3, int batch_size,int point_count,int feature_count);
    void SetModelInput(int *labels_B, int batch_size);
private:
    int B;
    int N;
    int K;

};


#endif //DEEPPOINTV1_MODEL01_H
