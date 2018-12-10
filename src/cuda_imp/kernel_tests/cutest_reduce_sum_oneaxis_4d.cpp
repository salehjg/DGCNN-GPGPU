//
// Created by saleh on 7/23/18.
//

#include "common.h"
#include "../../../submodules/cnpy/cnpy.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

extern
void reduce_sum_4d_try03(
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

extern
void reduce_sum_4d_try04(
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
void reduce_sum_4d_try05(
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

template<typename T> int DumpMatrix(
        string npy_fname,
        int rank,
        T* mat,
        int dim0,
        int dim1,
        int dim2,
        int dim3,
        int dim4){
    string npy_dir = "/home/saleh/00_repos/tensorflow_repo/00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/data/matrix_dumps/";
    if(rank==1){
        cnpy::npy_save<T>(npy_dir+npy_fname,&mat[0],{(long unsigned int)dim0},"w");
    }
    if(rank==2){
        cnpy::npy_save<T>(npy_dir+npy_fname,&mat[0],{(long unsigned int)dim0,(long unsigned int)dim1},"w");
    }
    if(rank==3){
        cnpy::npy_save<T>(npy_dir+npy_fname,&mat[0],{(long unsigned int)dim0,
                                          (long unsigned int)dim1,
                                          (long unsigned int)dim2},"w");
    }
    if(rank==4){
        cnpy::npy_save<T>(npy_dir+npy_fname,&mat[0],{(long unsigned int)dim0,
                                          (long unsigned int)dim1,
                                          (long unsigned int)dim2,
                                          (long unsigned int)dim3},"w");
    }
    if(rank==5){
        cnpy::npy_save<T>(npy_dir+npy_fname,&mat[0],{(long unsigned int)dim0,
                                                  (long unsigned int)dim1,
                                                  (long unsigned int)dim2,
                                                  (long unsigned int)dim3,
                                                  (long unsigned int)dim4},"w");
    }
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
    /*      60,1024,20,64  ---------> 1.4788 ms
     *      60,1024,20,128 ---------> 2.9376 ms
     *      60,1024,1 ,1024---------> 1.1873 ms
     *
     */

    const unsigned int
            dim0 = 60 ,
            dim1 = 1024 ,
            dim2 = 1,
            dim3 = 1024;

    unsigned int size = dim0 * dim1 * dim2 * dim3;
    printf("\nwith array size %d  \n", size);

    // allocate host memory
    size_t fbytes_input =      (dim0*dim1*dim2*dim3) * sizeof(float);
    size_t fbytes_output_overdim012 = (dim3) * sizeof(float);

    float *h_f_idata  = (float *) malloc(fbytes_input);
    float *h_f_odata012 = (float *) malloc(fbytes_output_overdim012);

    {
        int d0,d1,d2,indx;
        for(int d0=0;d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3;d3++){
                        indx = d0*dim1*dim2*dim3 + d1*dim2*dim3 + d2*dim3 + d3;
                        h_f_idata[indx] = (float)(1.0);
                    }
                }
            }
        }
    }

    float *h_f_result012 = LA_Sum4D(h_f_idata,true,true,true,false,dim0,dim1,dim2,dim3); //cpu results



    double iStart, iElaps;

    // allocate device memory
    float *d_f_idata = NULL;
    float *d_f_odata_overdim012 = NULL;
    CHECK(cudaMalloc((void **) &d_f_idata, fbytes_input));
    CHECK(cudaMalloc((void **) &d_f_odata_overdim012, fbytes_output_overdim012));

    CHECK(cudaMemcpy(d_f_idata, h_f_idata, fbytes_input, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();



    //reduce_sum_4d_try02(d_f_idata, d_f_odata_overdim012, dim0, dim1, dim2, dim3, true,true,true,false);
    //reduce_sum_4d_try03(d_f_idata, d_f_odata_overdim012, dim0, dim1, dim2, dim3, true,true,true,false);

    //reduce_sum_4d_try04(d_f_idata, d_f_odata_overdim012, dim0, dim1, dim2, dim3, true,true,true,false);

    reduce_sum_4d_try05(d_f_idata, d_f_odata_overdim012, dim0, dim1, dim2, dim3, true,true,true,false);




    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;


    CHECK(cudaMemcpy(h_f_odata012, d_f_odata_overdim012, fbytes_output_overdim012,cudaMemcpyDeviceToHost)); ///TODO: CHANGE BUFF SIZE TO CPY




    DumpMatrix<float>("reducesum_src.npy",2,h_f_idata,5,256,0,0,0);
    DumpMatrix<float>("reducesum_cpu.npy",1,h_f_result012,256,0,0,0,0);
    DumpMatrix<float>("reducesum_gpu.npy",1,h_f_odata012,256,0,0,0,0);

    ///TODO: CHECK RESULTS WITH CORRECT ONES!
    bool match0=true, match1=true, match2=true;
    {
        // over dim3 result check:
        for(int indx = 0 ; indx < dim3; indx++){
            if(h_f_odata012[indx] != h_f_result012[indx] ){
                cout<< "overDim012: Mismatch: "<< h_f_odata012[indx] <<" should be "<< h_f_result012[indx] <<endl;
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
    cout << "\t-OverAxis0,1 and 2 Matches: " << match0 << endl;
    cout << "-------------------------------------------------------" << endl;




    // free host memory
    free(h_f_idata);
    free(h_f_odata012);

    // free device memory
    CHECK(cudaFree(d_f_idata));
    CHECK(cudaFree(d_f_odata_overdim012));

    // reset device
    CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}


