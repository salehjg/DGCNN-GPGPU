
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define BIN_SIZE 256


__kernel
void LA_MatMul_3 (__global const float* matA,
                  __global const float* matB,
                  __global const float* matD,
                  int batch_size,
                  int dimA0,
                  int dimA1,
                  int dimB0,
                  int dimB1){

    int rdim0 = batch_size;
    int rdim1 = dimA0;
    int rdim2 = dimB1;

    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t groupId = get_group_id(0);
    size_t groupSize = get_local_size(0);

    for(int b=b_offset; b<rdim0; b++){
        for(int j=j_offset; j<rdim1; j++){
            for(int i=i_offset; i<rdim2; i++){

            }
        }
    }
}
