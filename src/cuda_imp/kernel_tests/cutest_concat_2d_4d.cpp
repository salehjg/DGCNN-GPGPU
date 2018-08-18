//
// Created by saleh on 7/23/18.
//

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

using namespace std;

extern
void concat_try01(
        float* g_iA,
        float* g_iB,
        float* g_o,
        const unsigned int dim0A,
        const unsigned int dim1A,
        const unsigned int dim2A,
        const unsigned int dim3A,

        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const unsigned int concatAxis);
//concat 2 matrices
// [matA, matB]
float* LA_Concat2(
        float* matA,
        float* matB,
        int rank,
        int concat_dim,
        int dimA0,
        int dimA1,
        int dimA2,
        int dimA3,
        int dimB0,
        int dimB1,
        int dimB2,
        int dimB3
){

    if(rank==4){
        int dimR0,dimR1,dimR2,dimR3;
        int mat2_offset_dim0=0;
        int mat2_offset_dim1=0;
        int mat2_offset_dim2=0;
        int mat2_offset_dim3=0;

        if(concat_dim==0){
            dimR0 = dimA0 + dimB0;
            dimR1 = dimA1;
            dimR2 = dimA2;
            dimR3 = dimA3;
            mat2_offset_dim0=dimA0;
        }
        if(concat_dim==1){
            dimR0 = dimA0;
            dimR1 = dimA1 + dimB1;
            dimR2 = dimA2;
            dimR3 = dimA3;
            mat2_offset_dim1=dimA1;
        }
        if(concat_dim==2){
            dimR0 = dimA0;
            dimR1 = dimA1;
            dimR2 = dimA2 + dimB2;
            dimR3 = dimA3;
            mat2_offset_dim2=dimA2;
        }
        if(concat_dim==3){
            dimR0 = dimA0;
            dimR1 = dimA1;
            dimR2 = dimA2;
            dimR3 = dimA3 + dimB3;
            mat2_offset_dim3=dimA3;
        }

        float* rslt = new float[dimR0*dimR1*dimR2*dimR3];
        int indxS1,indxS2,indxD;

        for(int d0=0;d0<dimA0;d0++){
            for(int d1=0;d1<dimA1;d1++){
                for(int d2=0;d2<dimA2;d2++){
                    for(int d3=0;d3<dimA3;d3++){
                        indxS1 = d0*dimA1*dimA2*dimA3 +
                                 d1*dimA2*dimA3+
                                 d2*dimA3+
                                 d3;
                        indxD = (d0)*dimR1*dimR2*dimR3 +
                                (d1)*dimR2*dimR3+
                                (d2)*dimR3+
                                (d3);
                        rslt[indxD] = matA[indxS1];
                    }
                }
            }
        }

        for(int d0=0;d0<dimB0;d0++){
            for(int d1=0;d1<dimB1;d1++){
                for(int d2=0;d2<dimB2;d2++){
                    for(int d3=0;d3<dimB3;d3++){
                        indxS2 = d0*dimB1*dimB2*dimB3 +
                                 d1*dimB2*dimB3+
                                 d2*dimB3+
                                 d3;
                        indxD  = (d0+mat2_offset_dim0)*dimR1*dimR2*dimR3 +
                                 (d1+mat2_offset_dim1)*dimR2*dimR3+
                                 (d2+mat2_offset_dim2)*dimR3+
                                 (d3+mat2_offset_dim3);
                        rslt[indxD] = matB[indxS2];
                    }
                }
            }
        }

        return rslt;


    }

    cout<<"LA_Concat2: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
}

int Test_4d() //Over Axis 3
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\ndevice %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    const unsigned int  dim0    = 16 ,
                        dim1    = 1024 ,
                        dim2    = 16,
                        dim3A   = 512,
                        dim3B   = 512,
                        dim3    = dim3A+dim3B;

    unsigned int size = dim0 * dim1 * dim2 * dim3A;
    printf("\nwith array size %d  \n", size);

    // allocate host memory
    size_t fbytes_inputA =      (dim0*dim1*dim2*dim3A) * sizeof(float);
    size_t fbytes_inputB =      (dim0*dim1*dim2*dim3B) * sizeof(float);
    size_t fbytes_output = (dim0*dim1*dim2*dim3) * sizeof(float);

    float *h_f_idataA  = (float *) malloc(fbytes_inputA);
    float *h_f_idataB  = (float *) malloc(fbytes_inputB);
    float *h_f_odata = (float *) malloc(fbytes_output);

    {
        int indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3A;d3++){
                        indx = d0*dim1*dim2*dim3A + d1*dim2*dim3A + d2*dim3A + d3;
                        h_f_idataA[indx] = (float)(1.5f);
                    }
                }
            }
        }

        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3B;d3++){
                        indx = d0*dim1*dim2*dim3B + d1*dim2*dim3B + d2*dim3B + d3;
                        h_f_idataB[indx] = (float)(2.5f);
                    }
                }
            }
        }
    }

    float *h_f_result = LA_Concat2(h_f_idataA,h_f_idataB,4,3,dim0,dim1,dim2,dim3A,dim0,dim1,dim2,dim3B);



    double iStart, iElaps;

    // allocate device memory
    float *d_f_idataA = NULL;
    float *d_f_idataB = NULL;
    float *d_f_odata = NULL;
    CHECK(cudaMalloc((void **) &d_f_idataA, fbytes_inputA));
    CHECK(cudaMalloc((void **) &d_f_idataB, fbytes_inputB));
    CHECK(cudaMalloc((void **) &d_f_odata, fbytes_output));

    CHECK(cudaMemcpy(d_f_idataA, h_f_idataA, fbytes_inputA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_f_idataB, h_f_idataB, fbytes_inputB, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();


    ///TODO: ADD KERNEL LAUNCHER HERE!
    //reduce_variance_4d_try01(d_f_idata, d_f_odata, dim0, dim1, dim2, dim3, true,true,true,false);
    concat_try01(d_f_idataA,d_f_idataB,d_f_odata,dim0,dim1,dim2,dim3A,dim0,dim1,dim2,dim3B,3);


    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;


    CHECK(cudaMemcpy(h_f_odata, d_f_odata, fbytes_output,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY



    ///TODO: CHECK RESULTS WITH CORRECT ONES!
    bool match0=true, match1=true, match2=true;
    {
        int indx=0;
        // FFTF Reduce Max Check:
        for(int d3 = 0 ; d3 < dim3 && match0; d3++){
            indx = d3;
            if(h_f_odata[indx] != h_f_result[indx] ){
                cout<< "FFTF: Mismatch: "<< h_f_odata[indx] <<" should be "<< h_f_result[indx] <<endl;
                match0=false;
                break;
            }
        }
    }


    cout << "-------------------------------------------------------" << endl;
    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout << "Tensor Size: dim0 x dim1 x dim2 x dim3 " << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 << " Float32" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "Comparison Results:" << endl;
    cout << "\t-OverAxis 3(RANK=4) Matches: " << match0 << endl;
    cout << "-------------------------------------------------------" << endl;




    // free host memory
    free(h_f_idataA);
    free(h_f_odata);

    // free device memory
    CHECK(cudaFree(d_f_idataA));
    CHECK(cudaFree(d_f_idataB));
    CHECK(cudaFree(d_f_odata));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}



int Test_2d_FTF() //FTF
{

}

int main(){
    Test_4d ();
    //Test_3d_FTF();
}