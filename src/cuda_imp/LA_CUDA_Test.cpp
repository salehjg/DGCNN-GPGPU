#include <iostream>
#include <cassert>
#include <stdio.h>
#include "LA_CUDA.h"
#include <cstring>
#include <cstdio>

using namespace std;

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
        } \
    } while (0)



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

void Test_LA_MatMul(int mode){
    /* Testing Steps:
     * 1. create a cublas handle
     * 2. create necessary matrices on host & compute expected result on host(cpu)
     * 3. copy matrices into device(gpu) memory
     * 4. feed them into LinearAlgebra method under test
     * 5. copy resulted matrix back into host(cpu) memory
     * 6. compare cpu and gpu results.
     * 7. print out statistics and timings into console.
     */
    //------------------------------------------------------------------------------------------------------------------
    cublasHandle_t myhandle;
    cublasStatus_t cublas_result;
    cudaError_t cuda_stat;
    int B=200; // batch size
    int M=30; // first  matrix is MxK
    int N=30; // second matrix is KxN
    int K=30; // result matrix is MxN
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

    //------------------------------------------------------------------------------------------------------------------
    //3. copy matrices into device(gpu) memory
    float** matA_ptr_device;
    float** matB_ptr_device;
    float** matR_ptr_device;

    float** matA_ptr_host = new float*[B];
    float** matB_ptr_host = new float*[B];
    float** matR_ptr_host = new float*[B];

    float* matA_device;
    float* matB_device;
    float* matR_device;


    //create device side pointer list
    cuda_stat = cudaMalloc((void**)&matA_ptr_device, B*sizeof(float *));
    assert(cuda_stat==cudaSuccess);
    cuda_stat = cudaMalloc((void**)&matB_ptr_device, B*sizeof(float *));
    assert(cuda_stat==cudaSuccess);
    cuda_stat = cudaMalloc((void**)&matR_ptr_device, B*sizeof(float *));
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR1");

    //assign pointers in list to respective locations of contiguous device memory buffer.
    cuda_stat = cudaMalloc((void**)&matA_device, B*M*K*sizeof(float));
    assert(cuda_stat==cudaSuccess);
    cuda_stat = cudaMalloc((void**)&matB_device, B*K*N*sizeof(float));
    assert(cuda_stat==cudaSuccess);
    cuda_stat = cudaMalloc((void**)&matR_device, B*M*N*sizeof(float));
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR2");

    //copy cpu batched buffers to gpu batched buffers
    cuda_stat =cudaMemcpy(matA_device,matA_host,B*M*K*sizeof(float),cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);
    cuda_stat =cudaMemcpy(matB_device,matB_host,B*K*N*sizeof(float),cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);
    //cuda_stat =cudaMemcpy(matR_device,matR_host,B*M*N*sizeof(float),cudaMemcpyHostToDevice);
    assert(cuda_stat==cudaSuccess);
    cudaCheckErrors("ERROR3");

    for(int b=0;b<B;b++){
        matA_ptr_host[b] = matA_device + b*M*K;
        matB_ptr_host[b] = matB_device + b*K*N;
        matR_ptr_host[b] = matR_device + b*M*N;
    }

    cudaMemcpy(matA_ptr_device, matA_ptr_host, B*sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(matB_ptr_device, matB_ptr_host, B*sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(matR_ptr_device, matR_ptr_host, B*sizeof(float *), cudaMemcpyHostToDevice);
    cudaCheckErrors("ERROR4");
    //------------------------------------------------------------------------------------------------------------------
    //4. feed them into LinearAlgebra method under test
    LA_CUDA pLA;
    pLA.MatMul(myhandle,matA_ptr_device,matB_ptr_device,matR_ptr_device,B,M,N,K);
    //pLA.GPU_Multi(matA,matB,matR,M,N,K,B,1.0f,0.0f);
    //------------------------------------------------------------------------------------------------------------------
    //5. copy resulted matrix back into host(cpu) memory
    float * gpu_results = new float[B*M*N];
    cudaMemcpy(gpu_results, matR_device, B*M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("ERROR5");
    //------------------------------------------------------------------------------------------------------------------
    //6. compare cpu and gpu results.
    int indx=0;
    for(int b=0;b<B;b++){
        PrintMatrix("GPU_Result_Batch"+to_string(b),gpu_results+b*M*N,M,N);
        for(int j=0;j<N;j++){
            for(int i=0;i<M;i++){
                indx = b*M*N + j * N + i;
                /*
                if(gpu_results[indx] != matA_host[indx]){
                    std::cout<<"Discrepancy at (j,i)=("<<j<<","<<i<<")\n";
                    std::cout<<"Expected " << matR_host[indx] << std::endl;
                    std::cout<<"Got      " << gpu_results[indx] << std::endl<< std::endl;
                }*/

            }
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
    //------------------------------------------------------------------------------------------------------------------
}

int main(){
    Test_LA_MatMul(0);
}