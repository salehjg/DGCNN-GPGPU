#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

using namespace std;

extern void tile_try02(
        float* g_i,
        float* g_o,
        const unsigned int dim0,
        const unsigned int dim1,
        const unsigned int dim2,
        const unsigned int tileAxis,
        const unsigned int count);


float* LA_Tile(float *input_BxNxD,int B,int N,int D,int K)
{
    unsigned int indxS1,indxD;
    //tile ing input of shape BxNxD into BxNxKxD.
    float *point_cloud_central = new float[B * N * K * D];

    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N; n++) {
            indxS1 = b * N * D + n * D + 0; //beginning of dim2 of input
            for (int k = 0; k < K; k++) {
                indxD = b * N * K * D + n * K * D + k * D + 0;
                std::copy(input_BxNxD + indxS1,
                          input_BxNxD + indxS1 + D,
                          point_cloud_central + indxD);
            }
        }
    }
    return point_cloud_central;
}






int Test() //FFTF
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\ndevice %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    const unsigned int  dim0 = 16 ,
                        dim1 = 1024 ,
                        dim2 = 1024 ,
                        tileCount = 16;

    unsigned int size = dim0 * dim1 * dim2;
    printf("\nwith array size %d  \n", size);

    // allocate host memory
    size_t fbytes_input =      (dim0*dim1*dim2) * sizeof(float);
    size_t fbytes_output = (dim0*dim1*dim2 * tileCount) * sizeof(float);

    float *h_f_idata  = (float *) malloc(fbytes_input);
    float *h_f_odata = (float *) malloc(fbytes_output);

    {
        int d0,d1,d2,indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){ 
                    indx = d0*dim1*dim2 + d1*dim2 + d2;
                    h_f_idata[indx] = (float) (rand());
                }
            }
        }
    }

    
    
    

    float *h_f_result = LA_Tile(h_f_idata,dim0,dim1,dim2,tileCount);

    
    

    
    
    double iStart, iElaps;

    // allocate device memory
    float *d_f_idata = NULL;
    float *d_f_odata = NULL;
    CHECK(cudaMalloc((void **) &d_f_idata, fbytes_input));
    CHECK(cudaMalloc((void **) &d_f_odata, fbytes_output));

    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes_input, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();


    tile_try02(
        d_f_idata,
        d_f_odata,
        dim0,
        dim1,
        dim2,
        2,
        tileCount);
    
    

    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;


    CHECK(cudaMemcpy(h_f_odata, d_f_odata, fbytes_output,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY






    ///TODO: CHECK RESULTS WITH CORRECT ONES!
    bool match0=true, match1=true, match2=true;
    {
        int indx=0;
        // FFTF Reduce Max Check:
        for(int d0 = 0 ; d0 < dim0      && match0; d0++){
        for(int d1 = 0 ; d1 < dim1      && match0; d1++){
        for(int d2 = 0 ; d2 < tileCount && match0; d2++){
        for(int d3 = 0 ; d3 < dim2      && match0; d3++){
            indx = d0*dim1*tileCount*dim2 + d1*tileCount*dim2 + d2*dim2 +d3;
            if(h_f_odata[indx] != h_f_result[indx] ){
                cout<< "FFTF: Mismatch: "<< h_f_odata[indx] <<" should be "<< h_f_result[indx] <<endl;
                cout<< "FFTF: Mismatch: "<< d0 << ", " <<  d1 << ", " <<  d2 << ", " <<  d3 << endl;
                match0=false;
                break;
            }
        }}}}
    }


    cout << "-------------------------------------------------------" << endl;
    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout << "Tensor Size: dim0 x dim1 x dim2 " << dim0 << "x" << dim1 << "x" << dim2 << " Float32" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "Comparison Results:" << endl;
    cout << "\t-TileAxis=2 (RANK=4) Matches: " << match0 << endl;
    cout << "-------------------------------------------------------" << endl;




    // free host memory
    free(h_f_idata);
    free(h_f_odata);

    // free device memory
    CHECK(cudaFree(d_f_idata));
    CHECK(cudaFree(d_f_odata));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


int main(){
    Test();
}