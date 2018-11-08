// C = AB
// https://stackoverflow.com/questions/8888718/how-to-declare-local-memory-in-opencl
__kernel void kernel_batch_matmul(
        const __global float *  __restrict__ matA,
        const __global float *  __restrict__ matB,
        __global float *  __restrict__ matC,
        __local float *smem,
        int dim0,
        int dim1A, int dim2A,
        int dim1B, int dim2B,
        int dim1C, int dim2C,
        unsigned int worksize_x,
        unsigned int worksize_y,
        unsigned int worksize_z){

    int gtid_x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    int gtid_y = get_group_id(1) * get_local_size(1) + get_local_id(1);
    int gtid_z = get_group_id(2) * get_local_size(2) + get_local_id(2);

    //Padded because non-uniform work-group is not available in OCL1.2 (Trying to implement dynamic shared memory)
    if(gtid_x < worksize_x && gtid_y < worksize_y && gtid_z < worksize_z) {


        const unsigned int len_subA = get_local_size(1) * dim2A, len_subB =
                get_local_size(0) * dim1B; //len of sub matrices of A and B.
        const unsigned long
                len_A = dim0 * dim1A * dim2A,
                len_B = dim0 * dim1B * dim2B,
                len_C = dim0 * dim1C * dim2C;
        const unsigned long
                len_A_signleBatch = dim1A * dim2A,
                len_B_signleBatch = dim1B * dim2B,
                len_C_signleBatch = dim1C * dim2C;
        const unsigned int BLOCKSIZE_P2 = get_local_size(0) * get_local_size(1);

        //smemA = smem + 0;
        //smemB = smem + len_subA;


        // Block index
        unsigned int bx = get_group_id(0); // mapped to the sub-matrices of output
        unsigned int by = get_group_id(1); // mapped to the sub-matrices of output
        unsigned int bz = get_group_id(2); // batch index

        // Thread index
        unsigned int tx = get_local_id(0);
        unsigned int ty = get_local_id(1);

        unsigned int c_pos_x, c_pos_y;
        c_pos_x = bx * get_local_size(0) + tx;
        c_pos_y = by * get_local_size(1) + ty;

        unsigned long gidx1, gidx2;
        unsigned int _d1, _d2;

        //printf("## bx:%u, by:%u, tx:%u, ty:%u, c_pos_x:%u, c_pos_y:%u\n",bx,by,tx,ty,c_pos_x,c_pos_y);


        unsigned long offsetA = (by * get_local_size(1)) * dim2A;
        unsigned long offsetB = (bx * get_local_size(0)); //first row (d1=0)

        // Load sub matrices from global memory into shared memory

        unsigned long idxA, idxB;
        idxA = ty * get_local_size(0) + tx;
        idxB = ty * get_local_size(0) + tx;

        //printf("*** bx:%u, by:%u, tx:%u, ty:%u ,idxA:%ld, idxB:%ld\n",bx,by,tx,ty,idxA,idxB);

        while (idxA < len_subA) {//Block-stride loop
            gidx1 = offsetA + idxA;
            if (idxA < len_subA && gidx1 < len_A) {
                smem[idxA] = matA[bz * len_A_signleBatch + gidx1];
                /*printf("bx:%u, by:%u, tx:%u, ty:%u ,idxA:%ld, gidx1:%ld\n",bx,by,tx,ty,idxA,gidx1);*/
            } else {
                smem[idxA] = 0;
            }
            idxA += BLOCKSIZE_P2;
        }

        ///TODO: It might be better to store transposed subMatB in shared memory to avoid shared memory read conflict.
        ///      But then we might get shared memory write conflict. (?)
        while (idxB < len_subB) {//Block-stride loop
            //gidx2 = offsetB + (bx*BLOCK_SIZE)*dim2B + (idxB % dim2B);
            _d2 = idxB % get_local_size(0);
            _d1 = (idxB / get_local_size(0));
            gidx2 = offsetB + _d1 * dim2B + _d2;
            if (idxB < len_subB && _d1 < dim1B && _d2 < dim2B) {
                smem[len_subA + idxB] = matB[bz * len_B_signleBatch + gidx2];
                /*printf("* bx:%u, by:%u ,tx:%u, ty:%u ,idxB:%ld, _d1:%d, _d2:%d, gidx2:%ld\n",bx,by,tx,ty,idxB,_d1,_d2,gidx2);*/
            } else {
                smem[len_subA + idxB] = 0;
            }
            idxB += BLOCKSIZE_P2;
        }


        barrier(CLK_LOCAL_MEM_FENCE);




        // Multiply and add each result to produce output element of current thread in the thread block.
        if (c_pos_x < dim2C && c_pos_y < dim1C) {
            float output_element = 0.0f;

            //dim2A=dim1B is common equal dimension of 2 matrices  --- block-stride loop
            for (int k = 0; k < dim2A; k++) {
                output_element += smem[ty * dim2A + k] * smem[len_subA + k * get_local_size(0) + tx];
                /*printf("###bz:%d, c_pos_x:%d, c_pos_y:%d, smem[%d]=%f, smem[%d]=%f\n",
                        bz,c_pos_x,c_pos_y,
                        ty*dim2A+k,smem[ty*dim2A+k],
                        len_subA+ k*BLOCK_SIZE+tx,smem[len_subA+ k*BLOCK_SIZE+tx]);*/
            }

            ///TODO: Check matC index to not to exceed the len of matC!
            matC[bz * len_C_signleBatch + c_pos_y * dim2C + c_pos_x] = output_element;

        }


    }
}