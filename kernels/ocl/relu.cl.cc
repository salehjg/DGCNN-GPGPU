//
// Created by saleh on 10/8/18.
//

kernel void kernel_relu(global const float * __restrict__ g_idata, global float * __restrict__ g_odata, const unsigned long len){
    unsigned long idx = get_group_id(0) * get_local_size(0) + get_local_id(0);
    if(idx<len){
        g_odata[idx] = g_idata[idx]>0 ? g_idata[idx] : 0;
    }
}

/*
void activation_relu(
        const float *g_idata,
        float *g_odata,
        unsigned long len){
    unsigned long blocksize, gridsize;
    blocksize = 256;
    gridsize = (len + blocksize -1 )/blocksize;
    kernel_relu<<<gridsize,blocksize>>>(g_idata,g_odata,len);
}*/