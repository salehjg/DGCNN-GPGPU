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
void reduce_variance_4d_try01(
        float* g_idata,
        float* g_odata,
        unsigned long dim0,
        unsigned long dim1,
        unsigned long dim2,
        unsigned long dim3,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3);

extern
void reduce_variance_4d_try02(
        float* g_idata,
        float* g_odata,
        unsigned long dim0,
        unsigned long dim1,
        unsigned long dim2,
        unsigned long dim3,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2,
        bool overaxis3);



//[axis0,axis1,axis2]
float* LA_Sum(float* mat1,
                         bool over_axis0,
                         bool over_axis1,
                         bool over_axis2,
                         int dim0,
                         int dim1,
                         int dim2){
    int indxS=0;
    int indxD=0;
    float* rslt;


    if(over_axis0==true &&
       over_axis1==true &&
       over_axis2==true )
    {
        float sum = 0;
        rslt = &sum;
        int limit = dim0*dim1*dim2;

        for(int b=0;b<limit;b++) {
            (*rslt) += mat1[b];
        }
        return rslt;
    }

    if(over_axis0==true &&
       over_axis1==false &&
       over_axis2==false )
    {
        rslt = new float[dim2*dim1];
        float sum=0;
        for(int d1=0; d1<dim1;d1++){
            for(int d2=0;d2<dim2;d2++){
                sum=0;
                indxD = d1 * dim2 + d2;
                //sum over dim of interest
                for(int dx=0;dx<dim0;dx++)
                {
                    indxS = dx * dim1*dim2 + d1 * dim2 + d2;
                    sum+=mat1[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        return rslt;
    }

    if(over_axis0==false &&
       over_axis1==true &&
       over_axis2==false )
    {
        rslt = new float[dim2*dim0];
        float sum=0;
        for(int d0=0; d0<dim0;d0++){
            for(int d2=0;d2<dim2;d2++){
                sum=0;
                indxD = d0 *dim2 + d2;
                //sum over dim of interest
                for(int dx=0;dx<dim1;dx++)
                {
                    indxS = d0 * dim1*dim2 + dx * dim2 + d2;
                    sum+=mat1[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        return rslt;
    }

    if(over_axis0==false &&
       over_axis1==false &&
       over_axis2==true )
    {
        rslt = new float[dim1*dim0];
        float sum=0;
        for(int d0=0; d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                sum=0;
                indxD = d0 * dim1 + d1;
                //sum over dim of interest
                for(int dx=0;dx<dim2;dx++)
                {
                    indxS = d0 * dim1*dim2 + d1 * dim2 + dx;
                    sum+=mat1[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        return rslt;
    }

    cout<<"LA_SUM:ERROR_UNDEFINED_AXES_COMB"<<endl;
    return nullptr;
}

//[axis0,axis1,axis2,axis3] //No batch op, uses data as is(as a matrix)
float* LA_Sum4D(float* mat1,
                           bool over_axis0,
                           bool over_axis1,
                           bool over_axis2,
                           bool over_axis3,
                           int dim0,
                           int dim1,
                           int dim2,
                           int dim3){
    //cout<<"**LA_SUM4D: "<<dim0<<","<<dim1<<","<<dim2<<","<<dim3<<";\n";
    int indxS=0;
    int indxD=0;
    float* rslt;

    if(over_axis0==true &&
       over_axis1==true &&
       over_axis2==true &&
       over_axis3==false )
    {
        rslt = new float[dim3];
        float sum=0;
        for (int d3 = 0; d3 < dim3; d3++)
        {
            sum=0;
            indxD = d3;
            for (int d0 = 0; d0 < dim0; d0++) {
                for (int d1 = 0; d1 < dim1; d1++) {
                    for (int d2 = 0; d2 < dim2; d2++) {

                        indxS = d0*dim1*dim2*dim3+
                                d1*dim2*dim3+
                                d2*dim3+
                                d3;

                        sum += mat1[indxS];
                    }
                }
            }

            rslt[indxD] = sum;
        }

        return rslt;
    }

    cout<<"LA_SUM4D:ERROR_UNDEFINED_AXES_COMB"<<endl;
    return nullptr;
}

// rslt = MAT1 * scalar
float* LA_MatMul(float* mat1,float scalar,
                            int batchsize, int matrix_rank,
                            int matrixH1,int matrixW1){
    if(matrix_rank!=3){cout<<"LA_MatMul: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    float* rslt = new float[batchsize*matrixH1*matrixW1];
    int limit = batchsize*matrixH1*matrixW1;

    for(int b=0;b<limit;b++) {
        rslt[b] = mat1[b] * scalar;
    }
    return rslt;
}

float* LA_Mean(
        float* mat,
        int rank,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3,
        int dim0,
        int dim1,
        int dim2,
        int dim3){

    if(rank==4){
        if(!mean_axis3 && mean_axis0 && mean_axis1 && mean_axis2){
            float* sum = LA_Sum4D(mat,
                                  mean_axis0,
                                  mean_axis1,
                                  mean_axis2,
                                  mean_axis3,
                                  dim0,
                                  dim1,
                                  dim2,
                                  dim3);
            float *mean = new float[dim3];

            for(int d3=0;d3<dim3;d3++){
                mean[d3] = (sum[d3])/(float)(dim0*dim1*dim2);
            }
            free(sum);
            return mean;
        }
        cout<<"LA_MEAN: ERROR_UNDEFINED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==2){ //dim0 is batch, dim1 is fc layer output, ex.: for B=1 --> output=[1,256]
        if(!mean_axis1 && mean_axis0 ){
            float* sum = LA_Sum(mat,false,true,false,1,dim0,dim1); //result is of shape dim1
            float *mean ;
            mean = LA_MatMul(sum,(1.0f/dim0),1,3,1,dim1);
            free(sum);
            return mean;
        }
        cout<<"LA_MEAN: ERROR_UNDEFINED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==1){
        float* sum = LA_Sum(mat,true,true,true,1,1,dim0);
        float *mean = (float*)malloc(sizeof(float) * 1);
        *mean = (*sum)/(float)(dim0);
        free(sum);
        return mean;
    }
    cout<<"LA_MEAN: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
}



float* LA_Variance(
        float* mat,
        int rank,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3,
        int dim0,
        int dim1,
        int dim2,
        int dim3

){
    if(rank==4){
        if(!variance_axis3 && variance_axis0 && variance_axis1 && variance_axis2) {
            float *mean = LA_Mean(mat,
                                  rank,
                                  variance_axis0,
                                  variance_axis1,
                                  variance_axis2,
                                  false,
                                  dim0,
                                  dim1,
                                  dim2,
                                  dim3);

            float* variance = new float[dim3];

            int indxS1,indxS2,indxD;
            for (int d3 = 0; d3 < dim3; d3++) { //over last-dim
                variance[d3]=0;


                for (int d0 = 0; d0 < dim0; d0++) {
                    for (int d1 = 0; d1 < dim1; d1++) {
                        for (int d2 = 0; d2 < dim2; d2++) {
                            indxS1 = d0*dim1*dim2*dim3+
                                     d1*dim2*dim3+
                                     d2*dim3+
                                     d3;

                            float delta = (mat[indxS1]-mean[d3]);
                            variance[d3] += delta*delta;
                        }
                    }
                }
            }

            float* variance_final = LA_MatMul(variance,(float)(1.0f/(dim0*dim1*dim2)),1,3,1,dim3);

            free(variance);
            free(mean);

            return variance_final;
        }
        cout<<"LA_MEAN: ERROR_UNDEFINED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==2){
        if(!variance_axis1 && variance_axis0 ) {
            float *mean = LA_Mean(mat,
                                  2,
                                  true,
                                  false,
                                  false,
                                  false,
                                  dim0,
                                  dim1,
                                  0,
                                  0);

            float* variance = new float[dim1];

            int indxS1,indxS2,indxD;
            for (int d1 = 0; d1 < dim1; d1++) { //over last-dim
                variance[d1]=0;

                for (int d0 = 0; d0 < dim0; d0++) {
                    indxS1 = d0*dim1 + d1;

                    float delta = (mat[indxS1]-mean[d1]);
                    variance[d1] += delta*delta;
                }
            }

            float* variance_final = LA_MatMul(variance,(float)(1.0f/dim0),1,3,1,dim1);

            free(variance);
            free(mean);

            return variance_final;
        }
        cout<<"LA_MEAN: ERROR_UNDEFINED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==1){
        float *mean = LA_Mean(mat,
                              rank,
                              true,
                              true,
                              true,
                              true,
                              dim0,
                              1,
                              1,
                              1);

        float *variance = (float*)malloc(sizeof(float) * 1);

        for (int d0 = 0; d0 < dim0; d0++) {
            float delta = (mat[d0]-mean[0]);
            *variance += delta*delta;
        }

        *variance = *variance * (float)(dim0);
        return variance;

    }
    cout<<"LA_MEAN: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
}

float float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

int Test_4d_FFTF() //FFTF
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\ndevice %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

/**
 *
 *   * Variance: , Shape1=5x1024x20x64x,    , Combination=1-1-1-0-, Test Result: 1
     * Variance: , Shape1=5x1024x20x128x,   , Combination=1-1-1-0-, Test Result: 1
     * Variance: , Shape1=5x256x,           , Combination=1-0-0-0-, Test Result: 1
     * Variance: , Shape1=5x512x,           , Combination=1-0-0-0-, Test Result: 1
     * Variance: , Shape1=5x1024x1x1024x,   , Combination=1-1-1-0-, Test Result: 1
     * Variance: , Shape1=5x1024x20x128x,   , Combination=1-1-1-0-, Test Result: 1
     * Variance: , Shape1=5x1024x20x64x,    , Combination=1-1-1-0-, Test Result: 1
     * Variance: , Shape1=5x1024x20x64x,    , Combination=1-1-1-0-, Test Result: 1
     * Variance: , Shape1=5x256x,           , Combination=1-0-0-0-, Test Result: 1
     * Variance: , Shape1=5x512x,           , Combination=1-0-0-0-, Test Result: 1
     * Variance: , Shape1=5x1024x1x1024x,   , Combination=1-1-1-0-, Test Result: 1
     * Variance: , Shape1=5x1024x20x64x,    , Combination=1-1-1-0-, Test Result: 1
 */
    const unsigned int  dim0 = 5 ,
                        dim1 = 1024 ,
                        dim2 = 20,
                        dim3 = 512;

    unsigned int size = dim0 * dim1 * dim2 * dim3;
    printf("\nwith array size %d  \n", size);

    // allocate host memory
    size_t fbytes_input =      (dim0*dim1*dim2*dim3) * sizeof(float);
    size_t fbytes_output_TTTF = (dim3) * sizeof(float);

    float *h_f_idata  = (float *) malloc(fbytes_input);
    float *h_f_odataTTTF = (float *) malloc(fbytes_output_TTTF);

    {
        int d0,d1,d2,indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3;d3++){
                        indx = d0*dim1*dim2*dim3 + d1*dim2*dim3 + d2*dim3 + d3;
                        h_f_idata[indx] = (float)(1.5f) + float_rand(-5.5f,5.5f);
                    }
                }
            }
        }
    }

    float *h_f_resultTTTF = LA_Variance(h_f_idata,4, true,true,true,false ,dim0,dim1,dim2,dim3); //cpu results FFTF



    double iStart, iElaps;

    // allocate device memory
    float *d_f_idata = NULL;
    float *d_f_odata_TTTF = NULL;
    CHECK(cudaMalloc((void **) &d_f_idata, fbytes_input));
    CHECK(cudaMalloc((void **) &d_f_odata_TTTF, fbytes_output_TTTF));

    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes_input, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();


    reduce_variance_4d_try01(d_f_idata, d_f_odata_TTTF, dim0, dim1, dim2, dim3, true,true,true,false);



    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;


    CHECK(cudaMemcpy(h_f_odataTTTF, d_f_odata_TTTF, fbytes_output_TTTF,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY






    ///TODO: CHECK RESULTS WITH CORRECT ONES!
    bool match0=true, match1=true, match2=true;
    {
        int indx=0;
        // FFTF Reduce Max Check:
        for(int d3 = 0 ; d3 < dim3 && match0; d3++){
            indx = d3;
            if(h_f_odataTTTF[indx] != h_f_resultTTTF[indx] ){
                cout<< "FFTF: Mismatch: "<< h_f_odataTTTF[indx] <<" should be "<< h_f_resultTTTF[indx] <<endl;
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
    cout << "\t-OverAxis FFTF(RANK=4) Matches: " << match0 << endl;
    cout << "-------------------------------------------------------" << endl;




    // free host memory
    free(h_f_idata);
    free(h_f_odataTTTF);

    // free device memory
    CHECK(cudaFree(d_f_idata));
    CHECK(cudaFree(d_f_odata_TTTF));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}



int Test_2d_FTF() //FTF
{

}

int main(){
    Test_4d_FFTF();
    //Test_3d_FTF();
}