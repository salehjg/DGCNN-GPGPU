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
void reduce_max_3d_try01(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2);

extern
void reduce_max_4d_try01(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        int dim3,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3);


float* LA_ReduceMax(
        float* input,
        int axis,
        int rank,
        int dim0,
        int dim1,
        int dim2,
        int dim3){
    cout<< "reduceMax: "<< "Rank: " << rank << ", Axis: "<< axis <<", Dim: " << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 << endl;
    if(rank==4){
        if(axis==3){
            float* rslt= new float[dim0*dim1*dim2];
            int indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= -numeric_limits<float>::infinity();

            for(int d0=0;d0<dim0;d0++){
                for(int d1=0;d1<dim1;d1++){
                    for(int d2=0;d2<dim2;d2++){
                        indxD = d0*dim1*dim2+
                                d1*dim2+
                                d2;
                        max = max_cte;
                        for(int d3=0;d3<dim3;d3++){
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<input[indxS]){
                                max = input[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            return rslt;
        }




        if(axis==2){
            float* rslt= new float[dim0*dim1*dim3];
            int indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= 0;

            for(int d0=0;d0<dim0;d0++){
                for(int d1=0;d1<dim1;d1++){
                    for(int d3=0;d3<dim3;d3++){
                        indxD = d0*dim1*dim3+
                                d1*dim3+
                                d3;
                        max = max_cte;

                        for(int d2=0;d2<dim2;d2++)
                        {
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<input[indxS]){
                                max = input[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            return rslt;
        }

        if(axis==1){
            float* rslt= new float[dim0*dim2*dim3];
            int indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= 0;

            for(int d0=0;d0<dim0;d0++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3;d3++){
                        indxD = d0*dim2*dim3+
                                d2*dim3+
                                d3;
                        max = max_cte;


                        for(int d1=0;d1<dim1;d1++)
                        {
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<input[indxS]){
                                max = input[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            return rslt;
        }
        cout<<"LA_ReduceMax: ERROR_UNDEFINED_POOLING_AXIS"<<endl;
        return nullptr;
    }

    cout<<"LA_ReduceMax: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
}


int Test_4d_FFTF() //FFTF
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\ndevice %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    const unsigned int  dim0 = 60 ,
                        dim1 = 1024 ,
                        dim2 = 20,
                        dim3 = 128;

    unsigned int size = dim0 * dim1 * dim2 * dim3;
    printf("\nwith array size %d  \n", size);

    // allocate host memory
    size_t fbytes_input =      (dim0*dim1*dim2*dim3) * sizeof(float);
    size_t fbytes_output_FFTF = (dim0*dim1*dim3) * sizeof(float);

    float *h_f_idata  = (float *) malloc(fbytes_input);
    float *h_f_odataFFTF = (float *) malloc(fbytes_output_FFTF);

    {
        int d0,d1,d2,indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3;d3++){
                        indx = d0*dim1*dim2*dim3 + d1*dim2*dim3 + d2*dim3 + d3;
                        h_f_idata[indx] = (float)(rand());
                    }
                }
            }
        }
    }

    float *h_f_resultFFTF = LA_ReduceMax(h_f_idata,2,4,dim0,dim1,dim2,dim3); //cpu results FFTF



    double iStart, iElaps;

    // allocate device memory
    float *d_f_idata = NULL;
    float *d_f_odata_FFTF = NULL;
    CHECK(cudaMalloc((void **) &d_f_idata, fbytes_input));
    CHECK(cudaMalloc((void **) &d_f_odata_FFTF, fbytes_output_FFTF));

    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes_input, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();


    reduce_max_4d_try01(d_f_idata, d_f_odata_FFTF, dim0, dim1, dim2, dim3, false,false,true,false);



    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;


    CHECK(cudaMemcpy(h_f_odataFFTF, d_f_odata_FFTF, fbytes_output_FFTF,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY






    ///TODO: CHECK RESULTS WITH CORRECT ONES!
    bool match0=true, match1=true, match2=true;
    {
        int indx=0;
        // FFTF Reduce Max Check:
        for(int d0 = 0 ; d0 < dim0 && match0; d0++){
        for(int d1 = 0 ; d1 < dim1 && match0; d1++){
        for(int d3 = 0 ; d3 < dim3 && match0; d3++){
            indx = d0*dim1*dim3 + d1*dim3 + d3;
            if(h_f_odataFFTF[indx] != h_f_resultFFTF[indx] ){
                cout<< "FFTF: Mismatch: "<< h_f_odataFFTF[indx] <<" should be "<< h_f_resultFFTF[indx] <<endl;
                match0=false;
                break;
            }
        }}}
    }


    cout << "-------------------------------------------------------" << endl;
    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout << "Tensor Size: dim0 x dim1 x dim2 x dim3 " << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 << " Float32" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "Comparison Results:" << endl;
    cout << "\t-OverAxis FFTF(RANK=4) Matches: " << match0 << endl;
    cout << "-------------------------------------------------------" << endl;




    // free host memory
    free(h_f_idata);
    free(h_f_odataFFTF);

    // free device memory
    CHECK(cudaFree(d_f_idata));
    CHECK(cudaFree(d_f_odata_FFTF));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}



int Test_3d_FTF() //FTF
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\ndevice %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));


    const unsigned int
            dim0 = 60 ,
            dim1 = 1024 ,
            dim2 = 1024;

    unsigned int size = dim0 * dim1 * dim2 ;
    printf("\nwith array size %d  \n", size);

    // allocate host memory
    size_t fbytes_input =      (dim0*dim1*dim2) * sizeof(float);
    size_t fbytes_output_FTF = (dim0*dim2) * sizeof(float);

    float *h_f_idata  = (float *) malloc(fbytes_input);
    float *h_f_odataFTF = (float *) malloc(fbytes_output_FTF);

    {
        int d0,d1,d2,indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                        indx = d0*dim1*dim2 + d1*dim2 + d2;
                        h_f_idata[indx] = (float)(rand());
                }
            }
        }
    }

    float *h_f_resultFTF = LA_ReduceMax(h_f_idata,1,4,dim0,dim1,dim2,1); //cpu results FFTF



    double iStart, iElaps;

    // allocate device memory
    float *d_f_idata = NULL;
    float *d_f_odata_FFTF = NULL;
    CHECK(cudaMalloc((void **) &d_f_idata, fbytes_input));
    CHECK(cudaMalloc((void **) &d_f_odata_FFTF, fbytes_output_FTF));

    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes_input, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();


    reduce_max_3d_try01(d_f_idata, d_f_odata_FFTF, dim0, dim1, dim2, false,true,false);



    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;


    CHECK(cudaMemcpy(h_f_odataFTF, d_f_odata_FFTF, fbytes_output_FTF,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY






    ///TODO: CHECK RESULTS WITH CORRECT ONES!
    bool match0=true, match1=true, match2=true;
    {
        int indx=0;
        // FTF Reduce Max Check:
        for(int d0 = 0 ; d0 < dim0 && match0; d0++){
                for(int d2 = 0 ; d2 < dim2 && match0; d2++){
                    indx = d0*dim2 + d2;
                    if(h_f_odataFTF[indx] != h_f_resultFTF[indx] ){
                        cout<< "FTF: Mismatch: "<< h_f_odataFTF[indx] <<" should be "<< h_f_resultFTF[indx] <<endl;
                        match0=false;
                        break;
                    }
                }}
    }


    cout << "-------------------------------------------------------" << endl;
    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout << "Tensor Size: dim0 x dim1 x dim2 x dim3 " << dim0 << "x" << dim1 << "x" << dim2 << " Float32" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "Comparison Results:" << endl;
    cout << "\t-OverAxis FFTF(RANK=4) Matches: " << match0 << endl;
    cout << "-------------------------------------------------------" << endl;




    // free host memory
    free(h_f_idata);
    free(h_f_odataFTF);

    // free device memory
    CHECK(cudaFree(d_f_idata));
    CHECK(cudaFree(d_f_odata_FFTF));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

int main(){
    //Test_4d_FFTF();
    Test_3d_FTF();
}