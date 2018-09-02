//
// Created by saleh on 8/23/18.
//

#include "../inc/TensorF.h"

TensorF::TensorF() {
    initialized = false;
    platform = PLATFORMS::DEFAULT; //Till it's not initialized, keep it general
}

TensorF::TensorF(std::vector<unsigned int> shape) {
    Init(shape);
}

TensorF::TensorF(std::vector<unsigned int> shape, float *buff) {
    Init(shape,buff);
}

void TensorF::Init(std::vector<unsigned int> shape) {
    if(initialized){
        delete(_buff);
    }
    this->shape = shape;
    _buff = new float[getLength()];
    initialized = true;
    platform = PLATFORMS::CPU;
}

void TensorF::Init(std::vector<unsigned int> shape, float* buff){
    if(initialized){
        delete(_buff);
    }
    this->shape = shape;
    _buff = buff;
    initialized = true;
    platform = PLATFORMS::CPU;
}

std::vector<unsigned int> TensorF::getShape(){
    return shape;
}

int TensorF::getRank() {
    return rank;
}

PLATFORMS TensorF::getPlatform(){
    return platform;
}

unsigned long TensorF::getLength() {
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

unsigned long TensorF::getLengthBytes() {
    if(initialized) {
        unsigned long len = 1;
        for(int i = 0;i<shape.size();i++){
            len = len * shape[i];
        }
        return len*sizeof(float);
    }else{
        return 0;
    }
}

TensorF::~TensorF() {
    if(initialized){
        delete(_buff);
    }
}