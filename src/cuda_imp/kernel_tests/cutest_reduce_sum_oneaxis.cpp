//
// Created by saleh on 7/23/18.
//

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;
/*
 * An example of using CUDA shuffle instructions to optimize performance of a
 * parallel reduction.
 */



extern void reduce_sum_3d_try01(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        int overaxis0,
        int overaxis1,
        int overaxis2);

extern void reduce_sum_3d_try02(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        int overaxis0,
        int overaxis1,
        int overaxis2);

extern void reduce_sum_3d_try03(
        float* g_idata,
        float* g_odata,
        int dim0,
        int dim1,
        int dim2,
        bool overaxis0,
        bool overaxis1,
        bool overaxis2);

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

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("\n%s starting reduction at ", argv[0]);
    printf("\ndevice %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // initialization
    const unsigned int  dim0 = 2 , //_LIMIT FOR OVER DIM0 IS SOMEHOW 2x8x16 , STH IS WRONG WHIT THAT KERNEL!
                        dim1 = 2 ,
                        dim2 = 4;

    unsigned int size = dim0 * dim1 * dim2;
    printf("\nwith array size %d  \n", size);

    // allocate host memory
    size_t fbytes_input =      (dim0*dim1*dim2) * sizeof(float);
    size_t fbytes_output_overdim0 = (dim1*dim2) * sizeof(float);
    size_t fbytes_output_overdim1 = (dim0*dim2) * sizeof(float);
    size_t fbytes_output_overdim2 = (dim0*dim1) * sizeof(float);

    float *h_f_idata  = (float *) malloc(fbytes_input);
    float *h_f_odata0 = (float *) malloc(fbytes_output_overdim0);
    float *h_f_odata1 = (float *) malloc(fbytes_output_overdim1);
    float *h_f_odata2 = (float *) malloc(fbytes_output_overdim2);

    {
        int d0,d1,d2,indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                    indx = d0*dim1*dim2 + d1*dim2 + d2;
                    h_f_idata[indx] = (float)1.0;
                }
            }
        }
    }

    float *h_f_result0 = LA_Sum(h_f_idata,true,false,false,dim0,dim1,dim2); //cpu results
    float *h_f_result1 = LA_Sum(h_f_idata,false,true,false,dim0,dim1,dim2); //cpu results
    float *h_f_result2 = LA_Sum(h_f_idata,false,false,true,dim0,dim1,dim2); //cpu results



    double iStart, iElaps;

    // allocate device memory
    float *d_f_idata = NULL;
    float *d_f_odata_overdim0 = NULL;
    float *d_f_odata_overdim1 = NULL;
    float *d_f_odata_overdim2 = NULL;
    CHECK(cudaMalloc((void **) &d_f_idata, fbytes_input));
    CHECK(cudaMalloc((void **) &d_f_odata_overdim0, fbytes_output_overdim0));
    CHECK(cudaMalloc((void **) &d_f_odata_overdim1, fbytes_output_overdim1));
    CHECK(cudaMalloc((void **) &d_f_odata_overdim2, fbytes_output_overdim2));

    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes_input, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();




    //reduce_sum_3d_try01(d_f_idata, d_f_odata_overdim0, dim0, dim1, dim2, 1,0,0);
    //reduce_sum_3d_try02(d_f_idata, d_f_odata_overdim1, dim0, dim1, dim2, 0,1,0);
    //reduce_sum_3d_try02(d_f_idata, d_f_odata_overdim2, dim0, dim1, dim2, 0,0,1);


    //reduce_sum_3d_try03(d_f_idata, d_f_odata_overdim0, dim0, dim1, dim2, true,false,false);
    reduce_sum_3d_try03(d_f_idata, d_f_odata_overdim1, dim0, dim1, dim2, false,true,false);
    //reduce_sum_3d_try03(d_f_idata, d_f_odata_overdim2, dim0, dim1, dim2, false,false,true);





    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;


    CHECK(cudaMemcpy(h_f_odata0, d_f_odata_overdim0, fbytes_output_overdim0,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY
    CHECK(cudaMemcpy(h_f_odata1, d_f_odata_overdim1, fbytes_output_overdim1,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY
    CHECK(cudaMemcpy(h_f_odata2, d_f_odata_overdim2, fbytes_output_overdim2,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY





    ///TODO: CHECK RESULTS WITH CORRECT ONES!
    bool match0=true, match1=true, match2=true;
    {
        // over dim0 result check:
        for(int indx = 0 ; indx < dim1*dim2; indx++){
            if(h_f_odata0[indx] != h_f_result0[indx] ){ cout<< "overDim0: Mismatch: "<< h_f_odata0[indx] <<" should be "<< h_f_result0[indx] <<endl;match0=false; break;}
        }

        // over dim1 result check:
        for(int indx = 0 ; indx < dim0*dim2; indx++){
            if(h_f_odata1[indx] != h_f_result1[indx] ){ cout<< "overDim1: Mismatch: "<< h_f_odata1[indx] <<" should be "<< h_f_result1[indx] <<endl; match1=false; break;}
        }

        // over dim2 result check:
        for(int indx = 0 ; indx < dim0*dim1; indx++){
            if(h_f_odata2[indx] != h_f_result2[indx] ){ cout<< "overDim2: Mismatch: "<< h_f_odata2[indx] <<" should be "<< h_f_result2[indx] <<endl; match2=false; break;}
        }
    }


    cout << "-------------------------------------------------------" << endl;
    cout << "Kernel Execution Time: " << iElaps*1000000 << " uS" << endl;
    cout << "Tensor Size: dim0xdim1xdim2 " << dim0 << "x" << dim1 << "x" << dim2 << " Float32" << endl;
    cout << "-------------------------------------------------------" << endl;
    cout << "Comparison Results:" << endl;
    cout << "\t-OverAxis0 Matches: " << match0 << endl;
    cout << "\t-OverAxis1 Matches: " << match1 << endl;
    cout << "\t-OverAxis2 Matches: " << match2 << endl;
    cout << "-------------------------------------------------------" << endl;




    // free host memory
    free(h_f_idata);
    free(h_f_odata0);
    free(h_f_odata1);
    free(h_f_odata2);

    // free device memory
    CHECK(cudaFree(d_f_idata));
    CHECK(cudaFree(d_f_odata_overdim0));
    CHECK(cudaFree(d_f_odata_overdim1));
    CHECK(cudaFree(d_f_odata_overdim2));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


