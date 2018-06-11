//
// Created by saleh on 5/31/18.
//

#include "model01.h"

ModelArch::ModelArch(int batchsize, int pointcount, int knn_k) {
    B = batchsize;
    N = pointcount;
    K = knn_k;
}

ModelInfo ModelArch::GetModelInfo() {
    ModelInfo tmplt;
    tmplt.Version="0.10";
    tmplt.DesignNotes="";
    tmplt.ExperimentNotes="";
    tmplt.ToDo=""
               "";
    tmplt.Date="97.3.10";
    return tmplt;
}

void ModelArch::SetModelInput(float *points_BxNx3, int batch_size, int point_count, int feature_count) {

}

void ModelArch::SetModelInput(int *labels_B, int batch_size) {

}
