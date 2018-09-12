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
        const unsigned int dim0A,
        const unsigned int dim1A,
        const unsigned int dim2A,
        const unsigned int dim3A,

        const unsigned int dim0B,
        const unsigned int dim1B,
        const unsigned int dim2B,
        const unsigned int dim3B,
        const unsigned int concatAxis);

#endif //DEEPPOINTV1_KERNELS_H
