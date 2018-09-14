#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <TensorF.h>
#include <CudaTensorF.h>
using namespace std;

extern
void tile_try03(
        float* g_i,
        float* g_o,
        unsigned int dim0,
        unsigned int dim1,
        unsigned int dim2,
        unsigned int dim3,
        int rank,
        int tileAxis,
        int tileCount);


TensorF* Tile(TensorF *inputTn, int tileAxis, int tileCount) {
    //Makes new tensor with same rank as inputTn's with tileAxis, tileCount times multiplied
    //tileAxis is in respect to the input tensor's axes.
    //----------------------------------------------------------------------------------------
    // inputTn       rsltTn         tileAxis        inputTn's Rank
    // BxNx1xD ----> BxNxKxD        2               4
    // BxNx1   ----> BxNxK          2               3
    // Bx1xN   ----> BxKxN          1               3
    // 1xD     ----> KxD            0               2

    unsigned long indxS1,indxD;
    if(inputTn->getRank()==4 && tileAxis==2) {
        unsigned int B,N,K,D;
        B = inputTn->getShape()[0];
        N = inputTn->getShape()[1];
        D = inputTn->getShape()[3];
        K = (unsigned int)tileCount;

        //tile ing input of shape BxNxD into BxNxKxD.
        TensorF* rsltTn = new TensorF({B, N, K, D});

        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                indxS1 = b * N * D + n * D + 0; //beginning of dim2 of input
                for (int k = 0; k < K; k++) {
                    indxD = b * N * K * D + n * K * D + k * D + 0;
                    std::copy(inputTn->_buff + indxS1,
                              inputTn->_buff + indxS1 + D,
                              rsltTn->_buff + indxD);
                }
            }
        }
        return rsltTn;
    }

    if(inputTn->getRank()==3 && tileAxis==2) { //BxN = BxNx1   ------->  BxNxK  (PAGE 221 of my design notebook)
        unsigned int B,N,K,D;
        B = inputTn->getShape()[0];
        N = inputTn->getShape()[1];
        K = (unsigned int)tileCount;

        //tile ing input of shape BxN or BxNx1 into BxNxK.
        TensorF* rsltTn = new TensorF({B, N, K});

        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                indxS1 = b*N + n;
                for(int k=0;k<K;k++){
                    indxD = b*N*K + n*K + k;
                    rsltTn->_buff[indxD] = inputTn->_buff[indxS1];
                }
            }
        }
        return rsltTn;
    }

    if(inputTn->getRank()==3 && tileAxis==1) { //BxN = Bx1xN   ------->  BxKxN  (PAGE 221 of my design notebook)
        unsigned int B,N,K,D;
        B = inputTn->getShape()[0];
        N = inputTn->getShape()[2];
        K = (unsigned int)tileCount;

        //tile ing input of shape BxN or Bx1xN into BxKxN.
        TensorF* rsltTn = new TensorF({B, K, N});

        for (int b = 0; b < B; b++) {
            for(int k=0;k<K;k++){
                for (int n = 0; n < N; n++) {
                    indxD  = b*K*N + k*N + n;
                    indxS1 = b*1*N + n;
                    rsltTn->_buff[indxD] = inputTn->_buff[indxS1];
                }
            }
        }
        return rsltTn;
    }

    if(inputTn->getRank()==2 && tileAxis==0) {
        unsigned int K,D;
        D = inputTn->getShape()[1];
        K = (unsigned int)tileCount;

        //tile ing input of shape BxNxD into BxNxKxD.
        TensorF* rsltTn = new TensorF({K, D});


        for (int k = 0; k < K; k++) {
            indxD = k * D + 0;
            std::copy(inputTn->_buff ,
                      inputTn->_buff + D,
                      rsltTn->_buff + indxD);
        }

        return rsltTn;
    }
}

int Test_Try03_4D() //FFTF
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\ndevice %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    unsigned int
    dim0 = 16 ,
    dim1 = 1024 ,
    dim2 = 1 ,
    dim3 = 1024 ,

    tileCount = 16;
    int tileAxis = 2;

    TensorF* hostSrcTn = new TensorF({dim0,dim1,dim2,dim3});
    TensorF* hostDstTn ;

    {
        int d0,d1,d2,indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3;d3++) {
                        indx = d0*dim1*dim2*dim3 + d1*dim2*dim3 + d2*dim3 + d3;
                        hostSrcTn->_buff[indx] = (float) (rand());
                    }
                }
            }
        }
    }

    hostDstTn = Tile(hostSrcTn,tileAxis,tileCount);

    CudaTensorF* deviceSrcTn = new CudaTensorF();
    CudaTensorF* deviceDstTn = new CudaTensorF();
    deviceSrcTn->InitWithHostData(hostSrcTn->getShape(),hostSrcTn->_buff);
    deviceDstTn->Init(hostDstTn->getShape());


    double iStart, iElaps;
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    tile_try03(
            deviceSrcTn->_buff,
            deviceDstTn->_buff,
            dim0,
            dim1,
            dim2,
            dim3,
            hostSrcTn->getRank(),
            tileAxis,
            tileCount);
    

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    TensorF* hostRsltTn = deviceDstTn->TransferToHost();

    if(hostRsltTn->getShape() != hostDstTn->getShape()){
        cout<< "ERROR: Result shapes do not match!"<<endl;
    }

    dim0 = (tileAxis==0)?tileCount:dim0;
    dim1 = (tileAxis==1)?tileCount:dim1;
    dim2 = (tileAxis==2)?tileCount:dim2;
    dim3 = (tileAxis==3)?tileCount:dim3;

    bool match0=true;
    {
        int indx=0;

        for(int d0 = 0 ; d0 < dim0      && match0; d0++){
        for(int d1 = 0 ; d1 < dim1      && match0; d1++){
        for(int d2 = 0 ; d2 < dim2      && match0; d2++){
        for(int d3 = 0 ; d3 < dim3      && match0; d3++){
            indx = d0*dim1*dim2*dim3 + d1*dim2*dim3 + d2*dim3 +d3;
            if(hostRsltTn->_buff[indx] != hostDstTn->_buff[indx] ){
                cout<< "Mismatch: "<< hostDstTn->_buff[indx] <<" should be "<< hostRsltTn->_buff[indx] <<endl;
                cout<< "Mismatch: "<< d0 << ", " <<  d1 << ", " <<  d2 << ", " <<  d3 << endl;
                match0=false;
                break;
            }
        }}}}
    }


    cout << "-------------------------------------------------------" << endl;
    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout << "Tensor Size: dim0 x dim1 x dim2 " << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 << " Float32" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "Comparison Results:" << endl;
    cout << "\t-Matches: " << match0 << endl;
    cout << "-------------------------------------------------------" << endl;

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


int Test_Try03_3D() //FFTF
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\ndevice %d: %s \n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    unsigned int
            dim0 = 16 ,
            dim1 = 1222 ,
            dim2 = 1 ,
            tileCount = 106;
    int tileAxis = 2;

    TensorF* hostSrcTn = new TensorF({dim0,dim1,dim2});
    TensorF* hostDstTn ;

    {
        int d0,d1,d2,indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                    indx = d0*dim1*dim2 + d1*dim2 + d2;
                    hostSrcTn->_buff[indx] = (float) (rand());
                }
            }
        }
    }

    hostDstTn = Tile(hostSrcTn,tileAxis,tileCount);

    CudaTensorF* deviceSrcTn = new CudaTensorF();
    CudaTensorF* deviceDstTn = new CudaTensorF();
    deviceSrcTn->InitWithHostData(hostSrcTn->getShape(),hostSrcTn->_buff);
    deviceDstTn->Init(hostDstTn->getShape());


    double iStart, iElaps;
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();

    tile_try03(
            deviceSrcTn->_buff,
            deviceDstTn->_buff,
            dim0,
            dim1,
            dim2,
            0,
            hostSrcTn->getRank(),
            tileAxis,
            tileCount);


    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;

    TensorF* hostRsltTn = deviceDstTn->TransferToHost();

    if(hostRsltTn->getShape() != hostDstTn->getShape()){
        cout<< "ERROR: Result shapes do not match!"<<endl;
    }

    dim0 = (tileAxis==0)?tileCount:dim0;
    dim1 = (tileAxis==1)?tileCount:dim1;
    dim2 = (tileAxis==2)?tileCount:dim2;

    bool match0=true;
    {
        int indx=0;

        for(int d0 = 0 ; d0 < dim0      && match0; d0++){
            for(int d1 = 0 ; d1 < dim1      && match0; d1++){
                for(int d2 = 0 ; d2 < dim2      && match0; d2++){
                        indx = d0*dim1*dim2 + d1*dim2 + d2 ;
                        if(hostRsltTn->_buff[indx] != hostDstTn->_buff[indx] ){
                            cout<< "Mismatch: "<< hostDstTn->_buff[indx] <<" should be "<< hostRsltTn->_buff[indx] <<endl;
                            cout<< "Mismatch: "<< d0 << ", " <<  d1 << ", " <<  d2  << endl;
                            match0=false;
                            break;
                        }
                    }}}
    }


    cout << "-------------------------------------------------------" << endl;
    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout << "Tensor Size: dim0 x dim1 x dim2 " << dim0 << "x" << dim1 << "x" << dim2 << " Float32" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "Comparison Results:" << endl;
    cout << "\t-Matches: " << match0 << endl;
    cout << "-------------------------------------------------------" << endl;

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


int main(){
    //Test_Try03_4D();
    Test_Try03_3D();
}