//
// Created by saleh on 10/8/18.
//

__global__ void kernel_relu(const float * __restrict__ g_idata, float * __restrict__ g_odata, unsigned long len){
    unsigned long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<len){
        g_odata[idx] = g_idata[idx]>0 ? g_idata[idx] : 0;
    }
}

void activation_relu(
        const float *g_idata,
        float *g_odata,
        unsigned long len){
    unsigned long blocksize, gridsize;
    blocksize = 256;
    gridsize = (len + blocksize -1 )/blocksize;
    kernel_relu<<<gridsize,blocksize>>>(g_idata,g_odata,len);
}