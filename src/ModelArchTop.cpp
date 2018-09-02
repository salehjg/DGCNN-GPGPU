//
// Created by saleh on 8/28/18.
//

#include "../inc/ModelArchTop.h"


ModelArchTop::ModelArchTop(int dataset_offset, int batchsize, int pointcount, int knn_k) {
    platformSelector = new PlatformSelector(PLATFORMS::CPU,{PLATFORMS::CPU});
    DB_OFFSET = dataset_offset;
    B = batchsize;
    N = pointcount;
    K = knn_k;
}

ModelInfo ModelArchTop::GetModelInfo() {
    ModelInfo tmplt;
    tmplt.ModelType="Classifier"
    tmplt.Version="1.0";
    tmplt.DesignNotes="";
    tmplt.ExperimentNotes="";
    tmplt.ToDo=""
               "";
    tmplt.Date="97.6.10";
    return tmplt;
}

void ModelArchTop::SetModelInput_data(string npy_pcl) {
    _npy_pcl = cnpy::npy_load(npy_pcl);
    input_pcl_BxNxD = new TensorF({B,N,3}, _npy_pcl.data<float>())
    input_pcl_BxNxD += DB_OFFSET*N*3;///TODO: CHECK COMPATIBILITY FOR GPU TENSOR(CUDA & OCL)
    assert( N == (int)(_npy_pcl.shape[1]) );
}

void ModelArchTop::SetModelInput_labels(string npy_labels) {
    //dataType of npy file should be int32, NOT uchar8!
    // use dataset_B5_labels_int32.npy
    _npy_labels = cnpy::npy_load(npy_labels);
    input_labels_B = new TensorI({B},_npy_labels.data<int>())
    input_labels_B->_buff += DB_OFFSET;
}

TensorF* ModelArchTop::FullyConnected_Forward(WorkScheduler scheduler, TensorF* input_BxD, TensorF* weights, TensorF* biases){
    int ch_out = (int)weights->getShape().back();
    //TensorF* tiledBias = platformSelector->Tile(platformSelector->defaultPlatform,scheduler,biases,0,B);
    //TensorF* rsltTn = platformSelector->MatAdd(platformSelector->defaultPlatform,scheduler,tmp,tiledBias);
    TensorF* tmp = platformSelector->MatMul(platformSelector->defaultPlatform,scheduler,input_BxD,weights);
    TensorF* rsltTn = platformSelector->MatAddTiled(platformSelector->defaultPlatform,scheduler,tmp,biases);
    /*
    float* tmp = LA_MatMul(input_BxD,w_ptr,1,3,B,input_last_dim,input_last_dim,ch_out);
    for(int b=0;b<B;b++){
        for(int ch=0;ch<ch_out;ch++){
            tmp->_buff[b*ch_out+ch] = tmp->_buff[b*ch_out+ch] + biases->_buff[ch];
        }
    }
    */
    return rsltTn;
}

TensorF* ModelArchTop::Batchnorm_Forward(WorkScheduler scheduler, TensorF* input, TensorF* gamma, TensorF* beta, TensorF* ema_ave, TensorF* ema_var){
    int dim0,dim1,dim2,dim3,rank;
    float bn_decay = 0.5f;
    unsigned long indxS1;
    unsigned long indxS2;
    unsigned long indxD;

    TensorF* mu;
    TensorF* var;

    rank = input->getRank();
    dim0 = input->getShape()[0];
    dim1 = input->getShape()[1];

                float* gamma_ptr = weights_map[gamma_key];
                vector<size_t> gamma_shape = weightsshape_map[gamma_key];
                float* beta_ptr = weights_map[beta_key];
                vector<size_t> beta_shape = weightsshape_map[beta_key];

                float* ema_var = weights_map[ema_var_key];
                vector<size_t> ema_var_shape = weightsshape_map[ema_var_key];
                float* ema_ave = weights_map[ema_ave_key];
                vector<size_t> ema_ave_shape = weightsshape_map[ema_ave_key];


    ///TODO: Rank4 and Rank2 branches could be merged into 1 branch with just mu and var lines conditioned!

    if(rank==4){
        dim2 = input->getShape()[2];
        dim3 = input->getShape()[3];

        //mu and var is of shape (dim3)
        mu = platformSelector->Mean(platformSelector->defaultPlatform, scheduler, input, true, true, true, false);
        var = platformSelector->Variance(platformSelector->defaultPlatform, scheduler, input, true, true, true, false);
        //mu = LA_Mean(input,4,true,true,true,false,dim0,dim1,dim2,dim3);
        //var = LA_Variance(input,4,true,true,true,false,dim0,dim1,dim2,dim3);
        //------------------------------------------
        // Exponential Moving Average for mu and var
        TensorF *update_delta_ave, *update_delta_var;
        TensorF *update_delta_ave2, *update_delta_var2;

        update_delta_ave = platformSelector->MatSub(platformSelector->defaultPlatform, scheduler, ema_ave, mu);
        update_delta_ave2 = platformSelector->MatMul(platformSelector->defaultPlatform, scheduler, update_delta_ave, bn_decay);
        update_delta_var = platformSelector->MatSub(platformSelector->defaultPlatform, scheduler, ema_var, var);
        update_delta_var2 = platformSelector->MatMul(platformSelector->defaultPlatform, scheduler, update_delta_var, bn_decay);

        //float *final_ave =  LA_SUB(ema_ave,update_delta_ave2,3,1,1,dim3,0);
        TensorF *final_ave =  platformSelector->MatSub(platformSelector->defaultPlatform, scheduler, ema_ave, update_delta_ave2);

        //float *final_var =  LA_SUB(ema_var,update_delta_var2,3,1,1,dim3,0);
        TensorF *final_var =  platformSelector->MatSub(platformSelector->defaultPlatform, scheduler, ema_var, update_delta_var2);
        //------------------------------------------
        TensorF* xNormTmp1 = platformSelector->MatSubTiled(platformSelector->defaultPlatform,scheduler,input,final_ave);
        TensorF* xNormTmp2 = platformSelector->MatAddTiled(platformSelector->defaultPlatform,scheduler,final_var,1e-8);
        TensorF* xNormTmp3 = platformSelector->Sqrt(platformSelector->defaultPlatform,scheduler,xNormTmp2);
        TensorF* xNorm = platformSelector->DivideTiled(platformSelector->defaultPlatform,scheduler,xNormTmp1,xNormTmp3);
        TensorF* rsltTmp1 = platformSelector->MultiplyTiled(platformSelector->defaultPlatform,scheduler,xNorm,gamma);
        TensorF* rsltTn = platformSelector->MatAddTiled(platformSelector->defaultPlatform,scheduler,rsltTmp1,beta);
        /*
        //mu and var is of shape (dim3)
        for(int d3=0;d3<dim3;d3++) {
            for (int d0 = 0; d0 < dim0; d0++) {
                for (int d1 = 0; d1 < dim1; d1++) {
                    for (int d2 = 0; d2 < dim2; d2++) {
                        indxS1 = d0*dim1*dim2*dim3+
                                 d1*dim2*dim3+
                                 d2*dim3+
                                 d3;

                        X_norm[indxS1] = (float) ((input[indxS1]-final_ave[d3]) / sqrt(final_var[d3]+1e-8));
                        rslt[indxS1] = gamma_ptr[d3] * X_norm[indxS1] + beta_ptr[d3];
                    }
                }
            }
        }
        free(mu);
        free(var);
        free(X_norm);
        free(final_ave);
        free(final_var);
        free(update_delta_ave);
        free(update_delta_ave2);
        free(update_delta_var);
        free(update_delta_var2);
        */

        return rsltTn;
    }

    if(rank==2){

        //mu and var is of shape (dim1)
        mu = platformSelector->Mean(platformSelector->defaultPlatform, scheduler, input, true, false, false, false);
        var = platformSelector->Variance(platformSelector->defaultPlatform, scheduler, input, true, false, false, false);
        //mu = LA_Mean(input,2,true,false,false,false,dim0,dim1,0,0);
        //var = LA_Variance(input,2,true,false,false,false,dim0,dim1,0,0);
        //------------------------------------------
        // Exponential Moving Average for mu and var
        TensorF *update_delta_ave, *update_delta_var;
        TensorF *update_delta_ave2, *update_delta_var2;

        update_delta_ave = platformSelector->MatSub(platformSelector->defaultPlatform, scheduler, ema_ave, mu);
        update_delta_ave2 = platformSelector->MatMul(platformSelector->defaultPlatform, scheduler, update_delta_ave, bn_decay);
        update_delta_var = platformSelector->MatSub(platformSelector->defaultPlatform, scheduler, ema_var, var);
        update_delta_var2 = platformSelector->MatMul(platformSelector->defaultPlatform, scheduler, update_delta_var, bn_decay);

        //float *final_ave =  LA_SUB(ema_ave,update_delta_ave2,3,1,1,dim3,0);
        TensorF *final_ave =  platformSelector->MatSub(platformSelector->defaultPlatform, scheduler, ema_ave, update_delta_ave2);

        //float *final_var =  LA_SUB(ema_var,update_delta_var2,3,1,1,dim3,0);
        TensorF *final_var =  platformSelector->MatSub(platformSelector->defaultPlatform, scheduler, ema_var, update_delta_var2);
        //------------------------------------------
        TensorF* xNormTmp1 = platformSelector->MatSubTiled(platformSelector->defaultPlatform,scheduler,input,final_ave);
        TensorF* xNormTmp2 = platformSelector->MatAddTiled(platformSelector->defaultPlatform,scheduler,final_var,1e-8);
        TensorF* xNormTmp3 = platformSelector->Sqrt(platformSelector->defaultPlatform,scheduler,xNormTmp2);
        TensorF* xNorm = platformSelector->DivideTiled(platformSelector->defaultPlatform,scheduler,xNormTmp1,xNormTmp3);
        TensorF* rsltTmp1 = platformSelector->MultiplyTiled(platformSelector->defaultPlatform,scheduler,xNorm,gamma);
        TensorF* rsltTn = platformSelector->MatAddTiled(platformSelector->defaultPlatform,scheduler,rsltTmp1,beta);
        /*
        for (int d0 = 0; d0 < dim0; d0++) {
            for (int d1 = 0; d1 < dim1; d1++) {
                indxS1 = d0*dim1 + d1;
                X_norm[indxS1] = (float) ((input[indxS1]-final_ave[d1]) / sqrt(final_var[d1]+1e-8));
                rslt[indxS1] = gamma_ptr[d1] * X_norm[indxS1] + beta_ptr[d1];
            }
        }

        free(final_ave);
        free(final_var);
        free(X_norm);
        free(update_delta_ave);
        free(update_delta_ave2);
        free(update_delta_var);
        free(update_delta_var2);
        free(mu);
        free(var);
        */
        return rsltTn;
    }

}

TensorF* ModelArchTop::GetEdgeFeatures(WorkScheduler scheduler, TensorF *input_BxNxD, TensorI *knn_output_BxNxK) {

    //Gather knn's indices from input array.
    /*
    float* point_cloud_neighbors = new float[B*N*K*D];
    for(int b=0;b<B;b++){
        for(int n=0;n<N;n++){
            for(int k=0;k<K;k++){
                indxS1 = b*N*K + n*K + k;
                for(int d=0;d<D;d++)
                {
                    indxD = b*N*K*D + n*K*D + k*D + d;
                    indxS2 = b*N*D +
                             knn_output_BxNxK[indxS1]*D +
                             d;
                    point_cloud_neighbors[indxD] = input_BxNxD[indxS2];
                }
            }
        }
    }
    */
    TensorF* point_cloud_neighbors = platformSelector->Gather(platformSelector->defaultPlatform,scheduler,input_BxNxD,knn_output_BxNxK,1);
    //DumpMatrix<DType>("tmp1.npy",4,point_cloud_neighbors,B,N,K,D,0);


    //tile ing input of shape BxNxD into BxNxKxD..
    TensorF* point_cloud_central = platformSelector->Tile(platformSelector->defaultPlatform,scheduler,input_BxNxD,2,K);
    /*
    float* point_cloud_central=new float[B*N*K*D];
    for(int b=0;b<B;b++){
        for(int n=0;n<N;n++){
            indxS1= b*N*D + n*D + 0; //beginning of dim2 of input
            for(int k=0;k<K;k++){
                indxD = b*N*K*D + n*K*D + k*D + 0;
                std::copy(input_BxNxD+indxS1,
                          input_BxNxD+indxS1+D,
                          point_cloud_central+indxD);
            }
        }
    }*/
    //DumpMatrix<DType>("tmp2.npy",4,point_cloud_central,B,N,K,D,0);


    //float* features= LA_SUB(point_cloud_neighbors,point_cloud_central,4,B,N,K,D);
    TensorF* features = platformSelector->MatSub(platformSelector->defaultPlatform,scheduler, point_cloud_neighbors,point_cloud_central);
    //DumpMatrix<DType>("tmp3.npy",4,features,B,N,K,D,0);


    //concatenate centrals and features (BxNxKxD) and (BxNxKxD)
    //float* edge_feature = LA_Concat2(point_cloud_central,features,4,3,B,N,K,D,B,N,K,D);
    TensorF* edge_feature = platformSelector->Concat2(platformSelector->defaultPlatform,scheduler, point_cloud_central,features);
    //DumpMatrix<DType>("tmp4.npy",4,edge_feature,B,N,K,2*D,0);


    /*
    free(point_cloud_central);
    free(point_cloud_neighbors);
    free(features);
     */

    return edge_feature;
}

TensorF* ModelArchTop::PairwiseDistance(WorkScheduler scheduler, TensorF *input_BxNxD) {

    TensorF* point_cloud_transpose = platformSelector->Transpose(platformSelector->defaultPlatform,scheduler,input_BxNxD);
    TensorF* point_cloud_inner =  platformSelector->MatMul(platformSelector->defaultPlatform,scheduler,input_BxNxD,point_cloud_transpose);
    TensorF* point_cloud_inner2 = platformSelector->MatMul(platformSelector->defaultPlatform,scheduler,point_cloud_inner,-2.0f);
    TensorF* point_cloud_inner2p2 = platformSelector->Square(platformSelector->defaultPlatform,scheduler,input_BxNxD);
    TensorF* point_cloud_sum = platformSelector->ReduceSum(platformSelector->defaultPlatform,scheduler,point_cloud_inner2p2,false,false,true);

    //2D Matrix fed into function with virutal batch size of 1
    TensorF* point_cloud_sum_transpose = platformSelector->Transpose(platformSelector->defaultPlatform,scheduler,point_cloud_sum);  //changed dims

    TensorF* point_cloud_sum_tiled =  platformSelector->Tile(platformSelector->defaultPlatform,scheduler,point_cloud_sum,2,N); //result is BxNxK for k=N
    TensorF* point_cloud_sum_transpose_tiled =  platformSelector->Tile(platformSelector->defaultPlatform,scheduler,point_cloud_sum_transpose,1,N); //result is BxkxN for k=N
    TensorF* rsltTmpTn = platformSelector->MatAdd(platformSelector->defaultPlatform,scheduler,point_cloud_sum_tiled,point_cloud_sum_transpose_tiled); //both input tensors are BxNxN
    TensorF* rsltTn = platformSelector->MatAdd(platformSelector->defaultPlatform,scheduler,rsltTmpTn,point_cloud_inner2); //both input tensors are BxNxN

    /*
    float*rslt=new float[B*N*N];
    {
        int indxS1,indxS2,indxS3;

        //[b,n1,n2] ,shapes:
        //point_cloud_sum           =[B,N,1] -->s1
        //point_cloud_inner2        =[B,N,N] -->s2
        //point_cloud_sum_transpose =[B,1,N] -->s3

        for(int b=0;b<B;b++){
            for(int n1=0;n1<N;n1++){
                indxS1 = b*N+n1;
                for(int n2=0;n2<N;n2++){
                    indxS2 = b*N*N+n1*N+n2;
                    indxS3 = b*N+n2;
                    rslt[indxS2] =  point_cloud_sum[indxS1]+
                                    point_cloud_inner2[indxS2]+
                                    point_cloud_sum_transpose[indxS3];
                }
            }
        }
    }



    free(point_cloud_transpose);
    free(point_cloud_inner);
    free(point_cloud_inner2);
    free(point_cloud_inner2p2);
    free(point_cloud_sum);
    free(point_cloud_sum_transpose);

    */
    return rsltTn;
}
