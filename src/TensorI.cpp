//
// Created by saleh on 8/23/18.
//

#include "../inc/TensorI.h"

TensorI::TensorI() {
    initialized = false;
    platform = PLATFORMS::DEFAULT; //Till it's not initialized, keep it general
}

TensorI::TensorI(std::vector<unsigned int> shape) {
    Init(shape);
}

TensorI::TensorI(std::vector<unsigned int> shape, int *buff) {
    Init(shape,buff);
}

void TensorI::Init(std::vector<unsigned int> shape) {
    if(initialized){
        delete(_buff);
    }
    this->shape = shape;
    _buff = new int[getLength()];
    initialized = true;
    platform = PLATFORMS::CPU;
}

void TensorI::Init(std::vector<unsigned int> shape, int* buff){
    if(initialized){
        delete(_buff);
    }
    this->shape = shape;
    _buff = buff;
    initialized = true;
    platform = PLATFORMS::CPU;
}

std::vector<unsigned int> TensorI::getShape(){
    return shape;
}

int TensorI::getRank() {
    return rank;
}

PLATFORMS TensorI::getPlatform(){
    return platform;
}

unsigned long TensorI::getLength() {
    if(initialized) {
        unsigned long len = 1;
        for (int i = 0; i < shape.size(); i++) {
            len = len * shape[i];
        }
        return len;
    }else{
        return 0;
    }
}

unsigned long TensorI::getLengthBytes() {
    if(initialized) {
        unsigned long len = 1;
        for(int i = 0;i<shape.size();i++){
            len = len * shape[i];
        }
        return len*sizeof(int);
    }else{
        return 0;
    }
}

TensorI::~TensorI() {
    if(initialized){
        delete(_buff);
    }
}