//
// Created by saleh on 9/11/18.
//

#ifndef DEEPPOINTV1_KERNELS_H
#define DEEPPOINTV1_KERNELS_H

extern
void concat_try01(
        float* g_iA,
        float* g_iB,
        float* g_o,
        unsigned int dim0A,
        unsigned int dim1A,
        unsigned int dim2A,
        unsigned int dim3A,

        unsigned int dim0B,
        unsigned int dim1B,
        unsigned int dim2B,
        unsigned int dim3B,
        unsigned int concatAxis);
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
void reduce_sum_3d_try03(
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

extern
void reduce_mean_4d_try02(
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
void mat_ops_try01(
        float* g_iA,
        float* g_iB,
        float* g_o,
        unsigned int rankA,
        unsigned int dim0A,
        unsigned int dim1A,
        unsigned int dim2A,
        unsigned int dim3A,
        unsigned int rankB,
        unsigned int dim0B,
        unsigned int dim1B,
        unsigned int dim2B,
        unsigned int dim3B,
        int operationMode);

#endif //DEEPPOINTV1_KERNELS_H
