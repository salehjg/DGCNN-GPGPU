//
// Created by saleh on 10/8/18.
//

kernel void kernel_sqrt_float(global const float * __restrict__ g_idata, global float * __restrict__ g_odata, const unsigned long len){
    unsigned long idx = get_group_id(0) * get_local_size(0) + get_local_id(0);//blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<len){
        g_odata[idx] = sqrt(g_idata[idx]);
    }
}
/*
void sqrt_float(
        const float *g_idata,
        float *g_odata,
        unsigned long len){
    unsigned long blocksize, gridsize;
    blocksize = 256;
    gridsize = (len + blocksize -1 )/blocksize;
    kernel_sqrt_float<<<gridsize,blocksize>>>(g_idata,g_odata,len);
}*/