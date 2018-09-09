//
// Created by saleh on 8/23/18.
//
#include <cuda_runtime.h>
#include "../inc/TensorF.h"
#include "../inc/cuda_imp/CudaTensorF.h"
#include "../submodules/cnpy/cnpy.h"
#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

using namespace std;

map<string,int> strToIndexMap;
TensorF** weightsCPU;
vector<cnpy::NpyArray> _weight_cnpy;

// https://stackoverflow.com/a/33373704/8296604
void LoadFromDisk(string weightsBaseDir, string pathToTxtFnameList) {
    std::ifstream inFile(pathToTxtFnameList);
    long weightCount = std::count(std::istreambuf_iterator<char>(inFile), std::istreambuf_iterator<char>(), '\n');
    weightsCPU = new TensorF*[weightCount];

    ifstream txtfile (pathToTxtFnameList);
    if (!txtfile.is_open())
    {
        cout<<"Failed to open text file!";
        return;
    }

    string line; int idx=-1;

    while (std::getline(txtfile, line)) {
        string weight_npy_path = weightsBaseDir + line;
        _weight_cnpy.push_back( cnpy::npy_load(weight_npy_path) );
        vector<unsigned int> __shape(_weight_cnpy.back().shape.begin(), _weight_cnpy.back().shape.end());
        idx++;
        strToIndexMap.insert( std::make_pair(line, idx) );

        //(*weightsCPU)[idx] = new TensorF(__shape, current.data<float>()) ;
        //weightsCPU.emplace_back(__shape, current.data<float>());
        weightsCPU[idx] = new TensorF(__shape, _weight_cnpy.back().data<float>()) ;
    }

    txtfile.close();
}

TensorF* Access(const string &name){
    int idx = strToIndexMap[name];
    return weightsCPU[ idx ];
}

int main(){
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);
    cudaDeviceSynchronize();
    //-----------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------
    {
        TensorF *tnc01, *tnc02;
        tnc01 = new CudaTensorF({10, 10, 10});
        cout << "Tensor tnc01 len: " << tnc01->getLength() << endl;
        cout << "Tensor tnc01 lenBytes: " << tnc01->getLengthBytes() << endl;
        TensorF *trnsfrd = ((CudaTensorF *) tnc01)->TransferToHost();
        cout << "Tensor trnsfrd len: " << trnsfrd->getLength() << endl;

        tnc02 = new TensorF({10, 10, 10});
        cout << "Tensor tnc02 len: " << tnc02->getLength() << endl;
        cout << "Tensor tnc02 lenBytes: " << tnc02->getLengthBytes() << endl;
    }

    //-----------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------
    {
        LoadFromDisk("/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                     "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights_testfiles/",

                     "/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                     "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights_testfiles/filelist.txt");

    }

    //-----------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------
    {
        TensorF* testTensor01 = Access("test01_5x100_float32.npy");
        cout<<"Test Tensor:\nName: test01_5x100_float32.npy\nLength: "<< testTensor01->getLength()<<endl;
        unsigned int dim0,dim1;
        unsigned long sidx;
        dim0=testTensor01->getShape()[0];
        dim1=testTensor01->getShape()[1];
        for(unsigned int d0=0;d0<dim0;d0++){
            for(unsigned int d1=0;d1<dim1;d1++){
                sidx = (d0 * dim1 + d1);
                if(testTensor01->_buff[sidx] != d1/( powf(10.0f,d0+0.f) )  ){
                    cout<<"MISMATCH AT "<<d0<<","<<d1<< "\t\t"<<
                        testTensor01->_buff[sidx] << " Should've been "<<  d1/( powf(10.0f,d0+0.0f) ) <<endl;
                }
            }
        }
    }

    //-----------------------------------------------------------------------------------------------------------------
    //-----------------------------------------------------------------------------------------------------------------


}