#include <iostream>
#include <cassert>
#include <stdio.h>
#include "LA_CUDA.h"
#include <cstring>
#include <cstdio>

using namespace std;





void PrintMatrix(string mname, float* mat, int h, int w){
    cout<<"==========================================================================\n";
    cout<<"Matrix Dump of "<< mname<<" : "<<endl;
    for(int j = 0 ; j< h;j++){
        cout<<"\t";
        for(int i=0;i<w;i++){
            cout<<mat[j*w +i]<<",\t";
        }
        cout<<"\n";
    }
    cout<<"==========================================================================\n";
}




//void Test_LA_MatMul(int mode){
//    /* Testing Steps:
//     * 1. create a cublas handle
//     * 2. create necessary matrices on host & compute expected result on host(cpu)
//     * 3. copy matrices into device(gpu) memory
//     * 4. feed them into LinearAlgebra method under test
//     * 5. copy resulted matrix back into host(cpu) memory
//     * 6. compare cpu and gpu results.
//     * 7. print out statistics and timings into console.
//     */
//    //------------------------------------------------------------------------------------------------------------------
//    cublasHandle_t myhandle;
//    cublasStatus_t cublas_result;
//    cudaError_t cuda_stat;
//    int B=2; // batch size
//    int M=20; // first  matrix is MxK
//    int N=20; // second matrix is KxN
//    int K=20; // result matrix is MxN
//    //------------------------------------------------------------------------------------------------------------------
//    // 1. create a cublas handle
//    cublas_result = cublasCreate(&myhandle);
//    assert(cublas_result == CUBLAS_STATUS_SUCCESS);
//
//    //------------------------------------------------------------------------------------------------------------------
//    //2. create necessary matrices on host
//    float* matA_host = new float[B*M*K];
//    float* matB_host = new float[B*K*N]; //is an Identity Matrix
//    float* matR_host = new float[B*M*N]; //should be equal to matA_solo
//
//    //matrices on host memory are ALWAYS row-major, but on device(gpu) are column-major!
//    for(int b=0;b<B;b++)
//    {
//        int rows = M;
//        int cols = K;
//        for (int row = 0; row < rows; row++) { // height
//            for (int col = 0; col < cols; col++) { //width
//                matA_host[b*rows*cols+ row * cols + col] = (float)(row * cols + col);
//            }
//        }
//    }
//    for(int b=0;b<B;b++)
//    {
//        int rows = K;
//        int cols = M;
//        for (int row = 0; row < rows; row++) { // height
//            for (int col = 0; col < cols; col++) { //width
//                matB_host[b*rows*cols+ row * cols + col] = (row==col?1.0f:0.0f); //row-major matrix
//            }
//        }
//    }
//    for(int b=0;b<B;b++)
//    {
//        int rows = M;
//        int cols = K;
//        for (int row = 0; row < rows; row++) { // height
//            for (int col = 0; col < cols; col++) { //width
//                matR_host[b*rows*cols+ row * cols + col] = 0;//matA_host[row * cols + col]; ///TODO : THIS IS JUST FOR TEST, FIX IT
//            }
//        }
//    }
//
//    PrintMatrix("matA_solo",matA_host,M,K);
//    PrintMatrix("matB_solo",matB_host,K,N);
//    PrintMatrix("matR_solo",matR_host,M,N);
//
//    //------------------------------------------------------------------------------------------------------------------
//    //3. copy matrices into device(gpu) memory
//    float** matA_ptr_device=ConvertHost1D_to_Device2D(matA_host,B,M,K);
//    float** matB_ptr_device=ConvertHost1D_to_Device2D(matB_host,B,K,N);;
//    float** matR_ptr_device=ConvertHost1D_to_Device2D(matR_host,B,M,N);;
//
//
//    cudaCheckErrors("ERROR4");
//    //------------------------------------------------------------------------------------------------------------------
//    //4. feed them into LinearAlgebra method under test
//    LA_CUDA pLA;
//    pLA.MatMul(myhandle,matA_ptr_device,matB_ptr_device,matR_ptr_device,B,M,N,K,1.0f,0.0f,true,true);
//    //pLA.GPU_Multi(matA,matB,matR,M,N,K,B,1.0f,0.0f);
//    //------------------------------------------------------------------------------------------------------------------
//    //5. copy resulted matrix back into host(cpu) memory
//    float * gpu_results = new float[B*M*N];
//    float *start_addr = new float[2];
//    cudaMemcpy(start_addr,matR_ptr_device, sizeof(float*),cudaMemcpyDeviceToHost);
//
//    cudaMemcpy(gpu_results, *matR_ptr_device, B*M*N*sizeof(float), cudaMemcpyDeviceToHost);
//    cudaCheckErrors("ERROR5");
//    //------------------------------------------------------------------------------------------------------------------
//    //6. compare cpu and gpu results.
//    int indx=0;
//    for(int b=0;b<B;b++){
//        PrintMatrix("GPU_Result_Batch"+to_string(b),gpu_results+b*M*N,M,N);
//        for(int j=0;j<N;j++){
//            for(int i=0;i<M;i++){
//                indx = b*M*N + j * N + i;
//                /*
//                if(gpu_results[indx] != matA_host[indx]){
//                    std::cout<<"Discrepancy at (j,i)=("<<j<<","<<i<<")\n";
//                    std::cout<<"Expected " << matR_host[indx] << std::endl;
//                    std::cout<<"Got      " << gpu_results[indx] << std::endl<< std::endl;
//                }*/
//
//            }
//        }
//    }
//
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//    //------------------------------------------------------------------------------------------------------------------
//}
void Test_LA_MatMul(){
    cublasHandle_t myhandle;
    cublasStatus_t cublas_result;
    cudaError_t cuda_stat;
    int B=2; // batch size
    int M=20; // first  matrix is MxK
    int N=20; // second matrix is KxN
    int K=20; // result matrix is MxN
    Tensor matA;
    Tensor matB;
    Tensor matR;
    //------------------------------------------------------------------------------------------------------------------
    // 1. create a cublas handle
    cublas_result = cublasCreate(&myhandle);
    assert(cublas_result == CUBLAS_STATUS_SUCCESS);

    //------------------------------------------------------------------------------------------------------------------
    //2. create necessary matrices on host
    float* matA_host = new float[B*M*K];
    float* matB_host = new float[B*K*N]; //is an Identity Matrix
    float* matR_host = new float[B*M*N]; //should be equal to matA_solo

    //matrices on host memory are ALWAYS row-major, but on device(gpu) are column-major!
    for(int b=0;b<B;b++)
    {
        int rows = M;
        int cols = K;
        for (int row = 0; row < rows; row++) { // height
            for (int col = 0; col < cols; col++) { //width
                matA_host[b*rows*cols+ row * cols + col] = (float)(row * cols + col);
            }
        }
    }
    for(int b=0;b<B;b++)
    {
        int rows = K;
        int cols = M;
        for (int row = 0; row < rows; row++) { // height
            for (int col = 0; col < cols; col++) { //width
                matB_host[b*rows*cols+ row * cols + col] = (row==col?1.0f:0.0f); //row-major matrix
            }
        }
    }
    for(int b=0;b<B;b++)
    {
        int rows = M;
        int cols = K;
        for (int row = 0; row < rows; row++) { // height
            for (int col = 0; col < cols; col++) { //width
                matR_host[b*rows*cols+ row * cols + col] = 0;//matA_host[row * cols + col]; ///TODO : THIS IS JUST FOR TEST, FIX IT
            }
        }
    }

    PrintMatrix("matA_solo",matA_host,M,K);
    PrintMatrix("matB_solo",matB_host,K,N);
    PrintMatrix("matR_solo",matR_host,M,N);

    matA.InitWithHostData(matA_host,{B,M,K});
    matB.InitWithHostData(matB_host,{B,K,N});
    matR.InitWithHostData(matR_host,{B,M,N});

    LA_CUDA pLA;
    pLA.MatMul(myhandle,matA,matB,matR,1.0f,0.0f,true,true);
    pLA.MatMul(myhandle,matA,matB,matR,1.0f,0.0f,true,true);

    float *gpu_results = matR.TransferDeviceBufferToHost();
    for(int b=0;b<B;b++) {
        PrintMatrix("GPU_Result_Batch" + to_string(b), gpu_results + b * M * N, M, N);
    }
}

int main(){
    Test_LA_MatMul();
}