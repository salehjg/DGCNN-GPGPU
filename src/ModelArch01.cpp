//
// Created by saleh on 8/22/18.
//

#include "../inc/ModelArch01.h"
#include <iostream>
#include "../../submodules/cnpy/cnpy.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>


ModelArch01::ModelArch(int dataset_offset, int batchsize, int pointcount, int knn_k) {
    DB_OFFSET = dataset_offset;
    B = batchsize;
    N = pointcount;
    K = knn_k;
}

ModelInfo ModelArch01::GetModelInfo() {
    ModelInfo tmplt;
    tmplt.Version="0.10";
    tmplt.DesignNotes="";
    tmplt.ExperimentNotes="";
    tmplt.ToDo=""
               "";
    tmplt.Date="97.3.10";
    return tmplt;
}

void ModelArch01::SetModelInput_data(string npy_pcl) {
    _npy_pcl = cnpy::npy_load(npy_pcl);
    input_pcl_BxNxD = _npy_pcl.data<float>();
    input_pcl_BxNxD += DB_OFFSET*N*3;
    //B = (int)(_npy_pcl.shape[0]);
    N = (int)(_npy_pcl.shape[1]);

}

void ModelArch01::SetModelInput_labels(string npy_labels) {
    _npy_labels = cnpy::npy_load(npy_labels);
    input_labels_B = _npy_labels.data<unsigned char>();
    input_labels_B += DB_OFFSET;
    //if(B == -1)
    //    B = batch_size;
    //else
    //    if(B!=batch_size)
    //        cout<<"Error, inconsistent batchsize for label's npy file respect to the pcl npy file."<<endl;
}

int ModelArch01::LoadWeights(string weights_base_dir, string path_txt_fnamelist) {
    ifstream txtfile (path_txt_fnamelist);

    if (!txtfile.is_open())
    {
        cout<<"Failed to open text file!";
        return -1;
    }

    string line; int i=0;
    while (std::getline(txtfile, line)) {
        //cout<<"Line:"<<i++<<":"<<line<<endl;
        string weight_npy_path = weights_base_dir + line;
        _weights_vector.push_back(cnpy::npy_load(weight_npy_path));
        weights_map.insert(std::make_pair(line,_weights_vector.back().data<float>()));
        weightsshape_map.insert(std::make_pair(line,_weights_vector.back().shape));
    }

    txtfile.close();
}

float* ModelArch01::TransformNet(
        float* edge_features, //of shape B x N x K x InputLastDim
        int k){

    int InputLastDim=6;
    int ConvOutputLastDim=-1;
    float* net;
    float* transform;
    //----------------------------------------------------------------------------
    {
        float *net1 = Conv2D(
                edge_features,
                InputLastDim,
                "transform_net1.tconv1.weights.npy",
                "transform_net1.tconv1.biases.npy",
                &ConvOutputLastDim);
        DumpMatrix<DType>("A01_tnet_conv.npy",4,net1,B,N,K,ConvOutputLastDim,0);

        float *net2 = Batchnorm_Forward(
                net1,
                "transform_net1.tconv1.bn.gamma.npy",
                "transform_net1.tconv1.bn.beta.npy",
                "transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "transform_net1.tconv1.bn.transform_net1.tconv1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                4,
                B, N, K, ConvOutputLastDim);
        free(net1);
        DumpMatrix<DType>("A02_tnet_bn.npy",4,net2,B,N,K,ConvOutputLastDim,0);

        float *net3 = ReLU(net2, B * N * K * ConvOutputLastDim);
        free(net2);
        DumpMatrix<DType>("A03_tnet_relu.npy",4,net3,B,N,K,ConvOutputLastDim,0);

        //free(net1);
        //free(net2);

        net = net3;
    }

    //----------------------------------------------------------------------------
    {
        float *net1 = Conv2D(
                net,
                ConvOutputLastDim,
                "transform_net1.tconv2.weights.npy",
                "transform_net1.tconv2.biases.npy",
                &ConvOutputLastDim);
        DumpMatrix<DType>("A04_tnet_conv.npy",4,net1,B,N,K,ConvOutputLastDim,0);

        float *net2 = Batchnorm_Forward(
                net1,
                "transform_net1.tconv2.bn.gamma.npy",
                "transform_net1.tconv2.bn.beta.npy",
                "transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "transform_net1.tconv2.bn.transform_net1.tconv2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                4,
                B, N, K, ConvOutputLastDim);
        free(net1);
        DumpMatrix<DType>("A05_tnet_bn.npy",4,net2,B,N,K,ConvOutputLastDim,0);

        float *net3 = ReLU(net2, B * N * K * ConvOutputLastDim);
        free(net2);
        DumpMatrix<DType>("A06_tnet_relu.npy",4,net3,B,N,K,ConvOutputLastDim,0);

        //free(net1);
        //free(net2);

        net = net3;
    }

    //----------------------------------------------------------------------------
    {
        float *net1 = LA_ReduceMax(net,2,4,B,N,K,ConvOutputLastDim);
        DumpMatrix<DType>("A07_tnet_pool.npy",3,net1,B,N,ConvOutputLastDim,0,0);

        net = net1;
    }
    //----------------------------------------------------------------------------
    {
        //ATTENTION: DIM2 (zero based index) of input is 1, NOT 'K'
        float *net1 = Conv2D(
                net,
                ConvOutputLastDim,
                "transform_net1.tconv3.weights.npy",
                "transform_net1.tconv3.biases.npy",
                &ConvOutputLastDim,
                1);
        DumpMatrix<DType>("A08_tnet_conv.npy",4,net1,B,N,1,ConvOutputLastDim,0);

        //ATTENTION: DIM2 (zero based index) of input is 1, NOT 'K'
        float *net2 = Batchnorm_Forward(
                net1,
                "transform_net1.tconv3.bn.gamma.npy",
                "transform_net1.tconv3.bn.beta.npy",
                "transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "transform_net1.tconv3.bn.transform_net1.tconv3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                4,
                B, N, 1, ConvOutputLastDim);
        free(net1);
        DumpMatrix<DType>("A09_tnet_bn.npy",4,net2,B,N,1,ConvOutputLastDim,0);

        //ATTENTION: DIM2 (zero based index) of input is 1, NOT 'K'
        float *net3 = ReLU(net2, B * N * 1 * ConvOutputLastDim);
        free(net2);
        DumpMatrix<DType>("A10_tnet_relu.npy",4,net3,B,N,1,ConvOutputLastDim,0);

        //free(net1);
        //free(net2);

        net = net3;
    }

    //----------------------------------------------------------------------------
    {
        float *net1 = LA_ReduceMax(net,1,4,B,N,1,ConvOutputLastDim);
        DumpMatrix<DType>("A11_tnet_pool.npy",2,net1,B,ConvOutputLastDim,0,0,0);
        net = net1;
    }

    //----------------------------------------------------------------------------
    //FC
    // net is Bx1024
    {
        float *net1 = FullyConnected_Forward(
                net,
                "transform_net1.tfc1.weights.npy",
                "transform_net1.tfc1.biases.npy",
                ConvOutputLastDim);
        DumpMatrix<DType>("A12_tnet_fc.npy",2,net1,B,512,0,0,0);
        ConvOutputLastDim = 512;

        //net1 is of shape Bx1x1x512
        float *net2 = Batchnorm_Forward(
                net1,
                "transform_net1.tfc1.bn.gamma.npy",
                "transform_net1.tfc1.bn.beta.npy",
                "transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "transform_net1.tfc1.bn.transform_net1.tfc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                2,
                B, (1 * 1 * ConvOutputLastDim), 0, 0);
        free(net1);
        DumpMatrix<DType>("A13_tnet_bn.npy",2,net2,B,512,0,0,0);

        float *net3 = ReLU(net2, B * 1 * 1 * ConvOutputLastDim);
        free(net2);
        DumpMatrix<DType>("A13_tnet_relu.npy",2,net3,B,512,0,0,0);

        //free(net1);
        //free(net2);


        net = net3;
    }

    //----------------------------------------------------------------------------
    //FC
    // net is Bx1024
    {
        float *net1 = FullyConnected_Forward(
                net,
                "transform_net1.tfc2.weights.npy",
                "transform_net1.tfc2.biases.npy",
                ConvOutputLastDim);
        ConvOutputLastDim = 256;
        DumpMatrix<DType>("A14_tnet_fc.npy",2,net1,B,256,0,0,0);

        //net1 is of shape Bx1x1x256
        float *net2 = Batchnorm_Forward(
                net1,
                "transform_net1.tfc2.bn.gamma.npy",
                "transform_net1.tfc2.bn.beta.npy",
                "transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "transform_net1.tfc2.bn.transform_net1.tfc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                2,
                B, (1 * 1 * ConvOutputLastDim), 0, 0);
        free(net1);
        DumpMatrix<DType>("A15_tnet_bn.npy",2,net2,B,256,0,0,0);

        float *net3 = ReLU(net2, B * 1 * 1 * ConvOutputLastDim);
        free(net2);
        DumpMatrix<DType>("A16_tnet_relu.npy",2,net3,B,256,0,0,0);

        //free(net1);
        //free(net2);


        net = net3;
    }
    //----------------------------------------------------------------------------
    {
        float* weights = weights_map["transform_net1.transform_XYZ.weights.npy"];
        float* biases  = weights_map["transform_net1.transform_XYZ.biases.npy"];
        vector<size_t> weights_shape = weightsshape_map["transform_net1.transform_XYZ.weights.npy"];
        vector<size_t> biases_shape  = weightsshape_map["transform_net1.transform_XYZ.biases.npy"];

        float eye[] = {1,0,0,
                       0,1,0,
                       0,0,1};
        //eye matrix
        for(int i =0 ; i<9;i++){
            biases[i]+=eye[i]; ///TODO: CLONE POINTER, DONOT EDIT SOURCE OF WEIGHTS!
        }
        DumpMatrix<DType>("A17_biass_added.npy",1,biases,9,0,0,0,0);

        transform = LA_MatMul(
                net,
                weights,
                1,
                3,
                B,ConvOutputLastDim, //2D net's shape
                ConvOutputLastDim,9  //2D weight's shape
        );
        DumpMatrix<DType>("A18_transform_batch.npy",2,transform,B,9,0,0,0);



        //float* transform2 = LA_ADD(transform,biases,3,1,B,9); // WAS ORIGINALLY THIS
        //float* transform2 = LA_ADD(transform,biases,3,B,1,9);
        for(int b=0;b<B;b++){
            for(int i = 0;i<9;i++){
                transform[b*9+i] += biases[i];
            }
        }

        DumpMatrix<DType>("A19_transform_batch_bias.npy",2,transform,B,9,0,0,0);



        //free(transform);
        //free(net);
        //transform = transform2;
    }

    //----------------------------------------------------------------------------

    return transform;
}

float ModelArch01::execute() {

    float* net; float* net_BxNx3;
    float* endpoint_0;float* endpoint_1;float* endpoint_2;float* endpoint_3;
    int dim_endpoint_0, dim_endpoint_1, dim_endpoint_2, dim_endpoint_3;

    int InputLastDim;
    int ConvOutputLastDim;

    bool *correct = new bool[B];
    float accu =0;

    //----------------------------------------------------------------------------------------
    cout<<"Starting Process..."<<endl;
    cout<<"Batch Size  = "<< B << endl;
    cout<<"Point Count = "<< N << endl;

    //----------------------------------------------------------------------------------------
    // TransferNet(net_BxNx3 is this layer's input)
    {
        net_BxNx3 = input_pcl_BxNxD;
        InputLastDim = 3;
        DumpMatrix<DType>("B00_input_pcl_BxNxD.npy",3,input_pcl_BxNxD,B,N,3,0,0);
        float *adj_matrix = Pairwise_Distance(net_BxNx3,InputLastDim);
        DumpMatrix<DType>("B01_tnet_adj_matrix.npy",3,adj_matrix,B,N,N,0,0);
        int *nn_idx = KNN(adj_matrix);
        DumpMatrix<int>("B02_tnet_nn_idx.npy",3,nn_idx,B,N,K,0,0);
        float *edge_features = Get_Edge_Features(net_BxNx3, nn_idx, InputLastDim);
        free(nn_idx);
        free(adj_matrix);
        DumpMatrix<float >("B03_tnet_edgef.npy",4,edge_features,B,N,K,6,0);
        float *transform =TransformNet(edge_features,3);
        free(edge_features);
        DumpMatrix<DType>("B04_tnet_3x3.npy",3,transform,B,3,3,0,0); //Confirmed

        net = LA_MatMul(
                net_BxNx3,
                transform,
                B,
                3,      // rank
                N,3,    // pcl's shape
                3,3);   // transform's shape

        free(transform);
        DumpMatrix<DType>("C01_pcl.npy",3,net,B,N,3,0,0);
    }

    //----------------------------------------------------------------------------------------
    // DGCNN Layer #0
    cout<<"STATUS: "<<"DGCCN0 Started"<<endl;
    {
        InputLastDim = 3;
        float *adj_matrix = Pairwise_Distance(net,InputLastDim);
        int *nn_idx = KNN(adj_matrix);

        float *edge_features = Get_Edge_Features(
                net,
                nn_idx,
                InputLastDim);
        free(nn_idx);
        free(adj_matrix);

        float *net1 = Conv2D(
                edge_features,
                2 * InputLastDim,
                "dgcnn1.weights.npy",
                "dgcnn1.biases.npy",
                &ConvOutputLastDim);
        free(edge_features);
        DumpMatrix<DType>("C02_dg1_conv.npy",4,net1,B,N,K,ConvOutputLastDim,0); //CONFIRMED

        float *net2 = Batchnorm_Forward(
                net1,
                "dgcnn1.bn.gamma.npy",
                "dgcnn1.bn.beta.npy",
                "dgcnn1.bn.dgcnn1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "dgcnn1.bn.dgcnn1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                4,
                B,N,K,ConvOutputLastDim);
        free(net1);
        DumpMatrix<DType>("C03_dg1_bn.npy",4,net2,B,N,K,ConvOutputLastDim,0); //CONFIRMED

        float *net3 = ReLU(net2,B*N*K*ConvOutputLastDim);
        free(net2);

        float *net4 = LA_ReduceMax(net3,2,4,B,N,K,ConvOutputLastDim);
        free(net3);

        DumpMatrix<DType>("B05_dg1_pool.npy",3,net4,B,N,ConvOutputLastDim,0,0);//CONFIRMED

        //free(adj_matrix);
        //free(nn_idx);
        //free(edge_features);
        //free(net1);
        //free(net1);
        //free(net2);
        //free(net3);
        net = net4;
        endpoint_0 = net;
        dim_endpoint_0 = ConvOutputLastDim;
    }
    //----------------------------------------------------------------------------------------
    // DGCNN Layer #1
    cout<<"STATUS: "<<"DGCCN1 Started"<<endl;
    {
        float *adj_matrix = Pairwise_Distance(net,ConvOutputLastDim);
        int *nn_idx = KNN(adj_matrix);
        float *edge_features = Get_Edge_Features(net, nn_idx, ConvOutputLastDim);
        free(nn_idx);
        free(adj_matrix);

        float *net1 = Conv2D(
                edge_features,
                2 * ConvOutputLastDim,
                "dgcnn2.weights.npy",
                "dgcnn2.biases.npy",
                &ConvOutputLastDim);
        free(edge_features);

        float *net2 = Batchnorm_Forward(
                net1,
                "dgcnn2.bn.gamma.npy",
                "dgcnn2.bn.beta.npy",
                "dgcnn2.bn.dgcnn2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "dgcnn2.bn.dgcnn2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                4,
                B,N,K,ConvOutputLastDim);
        free(net1);

        float *net3 = ReLU(net2,B*N*K*ConvOutputLastDim);
        free(net2);

        float *net4 = LA_ReduceMax(net3,2,4,B,N,K,ConvOutputLastDim);
        free(net3);

        DumpMatrix<DType>("B06_dg2_pool.npy",3,net4,B,N,ConvOutputLastDim,0,0);//CONFIRMED
        //free(adj_matrix);
        //free(nn_idx);
        //free(edge_features);
        //free(net1);
        //free(net1);
        //free(net2);
        //free(net3);
        net = net4;
        endpoint_1 = net;
        dim_endpoint_1 = ConvOutputLastDim;
    }
    //----------------------------------------------------------------------------------------
    // DGCNN Layer #2
    cout<<"STATUS: "<<"DGCCN2 Started"<<endl;
    {
        float *adj_matrix = Pairwise_Distance(net,ConvOutputLastDim);
        int *nn_idx = KNN(adj_matrix);
        float *edge_features = Get_Edge_Features(net, nn_idx, ConvOutputLastDim);
        free(adj_matrix);
        free(nn_idx);


        float *net1 = Conv2D(
                edge_features,
                2 * ConvOutputLastDim,
                "dgcnn3.weights.npy",
                "dgcnn3.biases.npy",
                &ConvOutputLastDim);
        free(edge_features);

        float *net2 = Batchnorm_Forward(
                net1,
                "dgcnn3.bn.gamma.npy",
                "dgcnn3.bn.beta.npy",
                "dgcnn3.bn.dgcnn3.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "dgcnn3.bn.dgcnn3.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                4,
                B,N,K,ConvOutputLastDim);
        free(net1);

        float *net3 = ReLU(net2,B*N*K*ConvOutputLastDim);
        free(net2);

        float *net4 = LA_ReduceMax(net3,2,4,B,N,K,ConvOutputLastDim);
        free(net3);
        DumpMatrix<DType>("B07_dg3_pool.npy",3,net4,B,N,ConvOutputLastDim,0,0);//CONFIRMED
        //free(adj_matrix);
        //free(nn_idx);
        //free(edge_features);
        //free(net1);
        //free(net1);
        //free(net2);
        //free(net3);
        net = net4;
        endpoint_2 = net;
        dim_endpoint_2 = ConvOutputLastDim;
    }
    //----------------------------------------------------------------------------------------
    // DGCNN Layer #3
    cout<<"STATUS: "<<"DGCCN3 Started"<<endl;
    {
        float *adj_matrix = Pairwise_Distance(net,ConvOutputLastDim);
        int *nn_idx = KNN(adj_matrix);
        float *edge_features = Get_Edge_Features(net, nn_idx, ConvOutputLastDim);
        free(adj_matrix);
        free(nn_idx);

        float *net1 = Conv2D(
                edge_features,
                2 * ConvOutputLastDim,
                "dgcnn4.weights.npy",
                "dgcnn4.biases.npy",
                &ConvOutputLastDim);
        free(edge_features);


        float *net2 = Batchnorm_Forward(
                net1,
                "dgcnn4.bn.gamma.npy",
                "dgcnn4.bn.beta.npy",
                "dgcnn4.bn.dgcnn4.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "dgcnn4.bn.dgcnn4.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                4,
                B,N,K,ConvOutputLastDim);
        free(net1);

        float *net3 = ReLU(net2,B*N*K*ConvOutputLastDim);
        free(net2);

        float *net4 = LA_ReduceMax(net3,2,4,B,N,K,ConvOutputLastDim);
        free(net3);

        DumpMatrix<DType>("B08_dg4_pool.npy",3,net4,B,N,ConvOutputLastDim,0,0);//CONFIRMED
        //free(adj_matrix);
        //free(nn_idx);
        //free(edge_features);
        //free(net1);
        //free(net1);
        //free(net2);
        //free(net3);
        net = net4;
        endpoint_3 = net;
        dim_endpoint_3 = ConvOutputLastDim;
    }
    //----------------------------------------------------------------------------------------
    // concat layer
    cout<<"STATUS: "<<"Agg Layer Started"<<endl;
    {
        float *concatA = LA_Concat2(
                endpoint_0,
                endpoint_1,
                4,
                3,
                B,N,1,dim_endpoint_0,
                B,N,1,dim_endpoint_1);
        free(endpoint_0);
        free(endpoint_1);

        float *concatB = LA_Concat2(
                concatA,
                endpoint_2,
                4,
                3,
                B,N,1,dim_endpoint_0+dim_endpoint_1,
                B,N,1,dim_endpoint_2);
        free(endpoint_2);
        free(concatA);

        float *concatC = LA_Concat2(
                concatB,
                endpoint_3,
                4,
                3,
                B,N,1,dim_endpoint_0+dim_endpoint_1+dim_endpoint_2,
                B,N,1,dim_endpoint_3);
        free(concatB);
        free(endpoint_3);

        //COMFIRMED
        DumpMatrix<DType>("B09_agg_concat.npy",4,concatC,B,N,1,dim_endpoint_0+dim_endpoint_1+dim_endpoint_2+dim_endpoint_3,0);

        //free(concatA);
        //free(concatB);


        //ATTENTION>> DIM2(K) of concatenated matrix is ONE, NOT 'K'
        float* net1 = Conv2D(
                concatC,
                dim_endpoint_0+dim_endpoint_1+dim_endpoint_2+dim_endpoint_3,
                "agg.weights.npy",
                "agg.biases.npy",
                &ConvOutputLastDim,
                1);
        free(concatC);

        DumpMatrix<DType>("B10_agg_conv.npy",4,net1,B,N,1,ConvOutputLastDim,0);//CONFIRMED

        float *net2 = Batchnorm_Forward(
                net1,
                "agg.bn.gamma.npy",
                "agg.bn.beta.npy",
                "agg.bn.agg.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "agg.bn.agg.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                4,
                B,N,1,ConvOutputLastDim);
        free(net1);
        DumpMatrix<DType>("B11_agg_bn.npy",4,net2,B,N,1,ConvOutputLastDim,0);//CONFIRMED

        float *net3 = ReLU(net2,B*N*1*ConvOutputLastDim);
        free(net2);

        float *net4 = LA_ReduceMax(net3,1,4,B,N,1,ConvOutputLastDim);
        free(net3);
        DumpMatrix<DType>("B12_agg_pool.npy",4,net4,B,1,1,ConvOutputLastDim,0);//CONFIRMED

        //free(concatC);
        //free(net1);
        //free(net2);
        //free(net3);

        net = net4;
    }



    //----------------------------------------------------------------------------------------
    //RESHAPING TO (Bx-1)
    //----------------------------------------------------------------------------------------
    //FC1
    //net is of shape Bx1x1x1024
    cout<<"STATUS: "<<"FC Layer1 Started"<<endl;
    {
        float *net1 = FullyConnected_Forward(
                net,
                "fc1.weights.npy",
                "fc1.biases.npy",
                ConvOutputLastDim);
        ConvOutputLastDim = 512;

        DumpMatrix<DType>("B13_fc.npy",2,net1,B,ConvOutputLastDim,0,0,0);//CONFIRMED

        //net1 is of shape Bx1x1x512
        float *net2 = Batchnorm_Forward(
                net1,
                "fc1.bn.gamma.npy",
                "fc1.bn.beta.npy",
                "fc1.bn.fc1.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "fc1.bn.fc1.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                2,
                B, (1 * 1 * ConvOutputLastDim), 0, 0);
        free(net1);

        DumpMatrix<DType>("B14_fc.npy",2,net2,B,ConvOutputLastDim,0,0,0);

        float *net3 = ReLU(net2, B * 1 * 1 * ConvOutputLastDim);
        free(net2);

        //free(net1);
        //free(net2);


        net = net3;
    }
    //----------------------------------------------------------------------------------------
    //FC2
    //net is of shape Bx1x1x512
    cout<<"STATUS: "<<"FC Layer2 Started"<<endl;
    {
        float *net1 = FullyConnected_Forward(
                net,
                "fc2.weights.npy",
                "fc2.biases.npy",
                ConvOutputLastDim);
        ConvOutputLastDim = 256;

        DumpMatrix<DType>("B15_fc.npy",2,net1,B,ConvOutputLastDim,0,0,0);//CONFIRMED

        //net1 is of shape Bx1x1x512
        float *net2 = Batchnorm_Forward(
                net1,
                "fc2.bn.gamma.npy",
                "fc2.bn.beta.npy",
                "fc2.bn.fc2.bn.moments.Squeeze.ExponentialMovingAverage.npy",
                "fc2.bn.fc2.bn.moments.Squeeze_1.ExponentialMovingAverage.npy",
                2,
                B, (1 * 1 * ConvOutputLastDim), 0, 0);
        free(net1);

        DumpMatrix<DType>("B16_fc.npy",2,net2,B,ConvOutputLastDim,0,0,0);//CONFIRMED

        float *net3 = ReLU(net2, B * 1 * 1 * ConvOutputLastDim);
        free(net2);

        //free(net1);
        //free(net2);


        net = net3;
    }
    //----------------------------------------------------------------------------------------
    //FC3
    //net is of shape Bx1x1x256
    cout<<"STATUS: "<<"FC Layer3 Started"<<endl;
    {
        float *net1 = FullyConnected_Forward(
                net,
                "fc3.weights.npy",
                "fc3.biases.npy",
                ConvOutputLastDim);

        ConvOutputLastDim = 40;

        DumpMatrix<DType>("B17_fc.npy",2,net1,B,ConvOutputLastDim,0,0,0);//CONFIRMED

        net = net1;
    }
    //----------------------------------------------------------------------------------------
    //find argmax(net) and compute bool array of corrects.
    cout<<"STATUS: "<<"Computing Accuracy..."<<endl;
    {
        float max_cte = -numeric_limits<float>::infinity();
        float max = 0;
        int max_indx=-1;
        int *a1 = new int[B];


        for(int b=0;b<B;b++){

            max=max_cte;
            for(int c=0;c<ConvOutputLastDim;c++){
                if(max<net[b*ConvOutputLastDim+c]){
                    max=net[b*ConvOutputLastDim+c];
                    max_indx=c;
                }
            }

            //set maximum score for current batch index
            a1[b]=max_indx;
        }

        for(int b=0;b<B;b++){
            if(a1[b]==(int)input_labels_B[b]){
                correct[b]=true;
            }
            else{
                correct[b]=false;
            }
        }

        free(a1);
    }
    //----------------------------------------------------------------------------------------
    // compute accuracy using correct array.
    {
        float correct_cnt=0;
        for(int b=0;b<B;b++){
            if(correct[b]==true) correct_cnt++;
        }
        accu = correct_cnt / (float)B;

        cout<<"Correct Count: "<< correct_cnt <<endl;
        cout<<"Accuracy: "<< accu<<endl;
    }
    //----------------------------------------------------------------------------------------
    cout<<"Stopping Process..."<<endl;
    return accu;
}

float* ModelArch01::Pairwise_Distance(float* input_BxNxD,int input_last_dim){
/* INPUTS:
*      -input_BxNxD
* OUTPUTS:
*      -BxNxN
* */
    float* point_cloud_transpose = LA_transpose(input_BxNxD,B,3,N,input_last_dim);
    float* point_cloud_inner = LA_MatMul(input_BxNxD,point_cloud_transpose,B,3,N,input_last_dim,input_last_dim,N);
    float* point_cloud_inner2 = LA_MatMul(point_cloud_inner,-2.0f,B,3,N,N);
    float* point_cloud_inner2p2 = LA_Square(input_BxNxD,B,3,N,input_last_dim);
    float* point_cloud_sum = LA_Sum(point_cloud_inner2p2,false,false,true,B,N,input_last_dim);

    //2D Matrix fed into function with virutal batch size of 1
    //float* point_cloud_sum_transpose = LA_transpose(point_cloud_sum,1,3,B,N);  //Was originally this.
    float* point_cloud_sum_transpose = LA_transpose(point_cloud_sum,B,3,N,1);  //changed dims


    //float* sum1 = LA_ADD(point_cloud_inner2,point_cloud_sum,3,B,N,N);
    //float* sum2 = LA_ADD(sum1,point_cloud_sum_transpose,3,B,N,N);

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


    return rslt;
}

float* ModelArch01::Get_Edge_Features(float* input_BxNxD,
                                      int*   knn_output_BxNxK,
                                      int    D){


    int indxS1=0;
    int indxS2=0;
    int indxD=0;


    //Gather knn's indices from input array.
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
    //DumpMatrix<DType>("tmp1.npy",4,point_cloud_neighbors,B,N,K,D,0);

    //tile ing input of shape BxNxD into BxNxKxD.
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
    }
    //DumpMatrix<DType>("tmp2.npy",4,point_cloud_central,B,N,K,D,0);

    float* features= LA_SUB(point_cloud_neighbors,point_cloud_central,4,B,N,K,D);
    //DumpMatrix<DType>("tmp3.npy",4,features,B,N,K,D,0);

    //concatenate centrals and features (BxNxKxD) and (BxNxKxD)
    float* edge_feature=LA_Concat2(point_cloud_central,features,4,3,B,N,K,D,B,N,K,D);
    //DumpMatrix<DType>("tmp4.npy",4,edge_feature,B,N,K,2*D,0);

    free(point_cloud_central);
    free(point_cloud_neighbors);
    free(features);

    return edge_feature;
}


float* ModelArch01::Batchnorm_Forward(
        float* input,
        string gamma_key,
        string beta_key,
        string ema_ave_key,
        string ema_var_key,
        int rank,
        int dim0,
        int dim1,
        int dim2,
        int dim3){

    float* gamma_ptr = weights_map[gamma_key];
    vector<size_t> gamma_shape = weightsshape_map[gamma_key];
    float* beta_ptr = weights_map[beta_key];
    vector<size_t> beta_shape = weightsshape_map[beta_key];

    float* ema_var = weights_map[ema_var_key];
    vector<size_t> ema_var_shape = weightsshape_map[ema_var_key];
    float* ema_ave = weights_map[ema_ave_key];
    vector<size_t> ema_ave_shape = weightsshape_map[ema_ave_key];

    float bn_decay = 0.5f;

    float* mu;
    float* var;
    int indxS1;
    int indxS2;
    int indxD;

    if(rank==4){
        float* X_norm = new float[dim0*dim1*dim2*dim3];
        float* rslt = new float[dim0*dim1*dim2*dim3];
        mu = LA_Mean(input,4,true,true,true,false,dim0,dim1,dim2,dim3);
        var = LA_Variance(input,4,true,true,true,false,dim0,dim1,dim2,dim3);
        //------------------------------------------
        // Exponential Moving Average for mu and var
        float* update_delta_ave,*update_delta_var;
        float* update_delta_ave2,*update_delta_var2;

        update_delta_ave = LA_SUB(ema_ave,mu,3,1,1,dim3,0);
        update_delta_ave2 = LA_MatMul(update_delta_ave,bn_decay,1,3,1,dim3);

        update_delta_var = LA_SUB(ema_var,var,3,1,1,dim3,0);
        update_delta_var2 = LA_MatMul(update_delta_var,bn_decay,1,3,1,dim3);

        float *final_ave =  LA_SUB(ema_ave,update_delta_ave2,3,1,1,dim3,0);
        float *final_var =  LA_SUB(ema_var,update_delta_var2,3,1,1,dim3,0);
        //------------------------------------------


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

        return rslt;
    }
    /*
    if(rank==1){//DEPRICATED!!!!!!!!!!!!
        float* X_norm = new float[dim0];
        float* rslt = new float[dim0];

        mu = LA_Sum(input,true,true,true,dim0,1,1); //scalar ptr
        var = LA_Variance(input,1,true,true,true,true,dim0,1,1,1);//scalar ptr
        //mu and var is of shape (dim0)
        //------------------------------------------
        // Exponential Moving Average for mu and var
        float* update_delta_ave,*update_delta_var;
        float* update_delta_ave2,*update_delta_var2;

        update_delta_ave = LA_SUB(ema_ave,mu,3,1,1,dim0,0);
        update_delta_ave2 = LA_MatMul(update_delta_ave,bn_decay,1,3,1,dim0);

        update_delta_var = LA_SUB(ema_var,var,3,1,1,dim0,0);
        update_delta_var2 = LA_MatMul(update_delta_var,bn_decay,1,3,1,dim0);

        float *final_ave =  LA_SUB(ema_ave,update_delta_ave2,3,1,1,dim0,0);
        float *final_var =  LA_SUB(ema_var,update_delta_var2,3,1,1,dim0,0);
        //------------------------------------------



        for (int d0 = 0; d0 < dim0; d0++) {
            X_norm[d0] = (float) ((input[d0]-final_ave[d0]) / sqrt(final_var[d0]+1e-8));
            rslt[d0] = gamma_ptr[d0] * X_norm[d0] + beta_ptr[d0];
        }

        return rslt;
    }*/

    if(rank==2){
        float* X_norm = new float[dim0*dim1];
        float* rslt = new float[dim0*dim1];
        mu = LA_Mean(input,2,true,false,false,false,dim0,dim1,0,0);
        var = LA_Variance(input,2,true,false,false,false,dim0,dim1,0,0);
        //mu and var is of shape (dim1)

        //------------------------------------------
        // Exponential Moving Average for mu and var
        float* update_delta_ave,*update_delta_var;
        float* update_delta_ave2,*update_delta_var2;

        update_delta_ave = LA_SUB(ema_ave,mu,3,1,1,dim1,0);
        update_delta_ave2 = LA_MatMul(update_delta_ave,bn_decay,1,3,1,dim1);

        update_delta_var = LA_SUB(ema_var,var,3,1,1,dim1,0);
        update_delta_var2 = LA_MatMul(update_delta_var,bn_decay,1,3,1,dim1);

        float *final_ave =  LA_SUB(ema_ave,update_delta_ave2,3,1,1,dim1,0);
        float *final_var =  LA_SUB(ema_var,update_delta_var2,3,1,1,dim1,0);
        //------------------------------------------

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

        return rslt;
    }

}


float* ModelArch01::FullyConnected_Forward(
        float* input_BxD,
        string weight_key,
        string bias_key,
        int input_last_dim){


    float* w_ptr = weights_map[weight_key];
    vector<size_t> w_shape = weightsshape_map[weight_key];
    float* b_ptr = weights_map[bias_key];
    vector<size_t> b_shape = weightsshape_map[bias_key];

    int ch_out = (int)w_shape.back();

    float* tmp = LA_MatMul(input_BxD,w_ptr,1,3,B,input_last_dim,input_last_dim,ch_out);

    for(int b=0;b<B;b++){
        for(int ch=0;ch<ch_out;ch++){
            tmp[b*ch_out+ch] = tmp[b*ch_out+ch] + b_ptr[ch];
        }
    }

    return tmp;
}



template<typename T> int ModelArch01::DumpMatrix(
        string npy_fname,
        int rank,
        T* mat,
        int dim0,
        int dim1,
        int dim2,
        int dim3,
        int dim4,
        string npy_dir){
#ifdef DUMP_ENABLED
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
#endif
}
