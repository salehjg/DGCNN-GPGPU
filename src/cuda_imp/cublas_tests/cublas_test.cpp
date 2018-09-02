#include <iostream>
# include <cstdlib>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# define n 600000000 // length of x

int main(){
    cudaError_t cudaStat ;          // cudaMalloc status
    cublasStatus_t stat ;           // CUBLAS functions status
    cublasHandle_t handle ;         // CUBLAS context
    int j;                          // index of elements
    float * x;                      // n- vector on the host
    //---------------------------------------------------------------------------

    x=( float *) malloc(n* sizeof (*x ));  // host memory alloc
    for(j=0;j<n;j++)
        x[j]=( float )j;                    // x={0 ,1 ,2 ,3 ,4 ,5}
    //---------------------------------------------------------------------------

    //printf ("x: ");
    //for(j=0;j<n;j++)
    //    printf (" %4.0f,",x[j]); // print x
    //printf ("\n");
    //---------------------------------------------------------------------------

    // on the device
    float * d_x; // d_x - x on the device
    cudaStat = cudaMalloc(( void **)& d_x ,n* sizeof (*x )); // device
    // memory alloc for x
    //---------------------------------------------------------------------------

    stat = cublasCreate(& handle ); // initialize CUBLAS context
    stat = cublasSetVector(n, sizeof (*x),x ,1,d_x ,1); // cp x ->d_x
    //---------------------------------------------------------------------------

    int result ;
    stat=cublasIsamax(handle,n,d_x,1,&result);
    std::cout << "MAX: " << fabs(x[ result -1]) << std::endl;
    //---------------------------------------------------------------------------

    stat=cublasIsamin(handle,n,d_x,1,&result);
    std::cout << "MIN: " << fabs(x[ result -1]) << std::endl;
    //---------------------------------------------------------------------------

    cudaFree(d_x ); // free device memory
    cublasDestroy( handle ); // destroy CUBLAS context
    free (x); // free host memory
    //---------------------------------------------------------------------------

    return EXIT_SUCCESS ;
}