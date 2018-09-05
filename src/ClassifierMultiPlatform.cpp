//
// Created by saleh on 9/3/18.
//
#include "../inc/ModelArchTop.h"
#include <iostream>
using namespace std;

int main(){
    WorkScheduler scheduler;
    int batchsize=5;
    ModelArchTop modelArchTop(0,batchsize,1024,20);

    modelArchTop.SetModelInput_data("/home/saleh/00_repos/tensorflow_repo/"
                                    "00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/"
                                    "data/dataset/"
                                    "dataset_B5_pcl.npy");

    modelArchTop.SetModelInput_labels("/home/saleh/00_repos/tensorflow_repo/"
                                      "00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/"
                                      "data/dataset/"
                                      "dataset_B5_labels_int32.npy");

    double timerStart = seconds();
    modelArchTop.Execute(scheduler);
    cout<< "Total model execution time with "<< batchsize <<" as batchsize: " << seconds() -timerStart<<" S"<<endl;
}

