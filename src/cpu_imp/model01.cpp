//
// Created by saleh on 5/31/18.
//
#include <iostream>
#include "../../submodules/cnpy/cnpy.h"
#include "model01.h"
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

/* INPUTS:
 *      -input_BxNxD
 * OUTPUTS:
 *      -BxNxN
 * */
float* ModelArch01::Pairwise_Distance(float* input_BxNxD,int input_last_dim){
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

int* ModelArch01::KNN(float* adj_matrix_BxNxN){
    //we will use std::sort in ascending order.
    int indxS=0;
    int* rslt = new int[B*N*K];

    float tmp_array[N];
    int indices[N];
    for(int i = 0 ;i<N;i++)
        indices[i]=i;

    for(int b=0;b<B;b++){
        for(int n=0;n<N;n++){
            indxS = b*N*N + n*N + 0;
            //start of dim2
            std::copy(adj_matrix_BxNxN+indxS,adj_matrix_BxNxN+indxS+N,tmp_array);
            //std::sort(tmp_array,tmp_array+N);


            std::sort(  indices,
                        indices+N,
                        [&](int i1, int i2) { return tmp_array[i1] < tmp_array[i2]; } );

            std::copy(indices,indices+K,rslt+(b*N*K + n*K + 0));
        }
    }

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

float* ModelArch01::Conv2D(
        float* input,
        int input_last_dim,
        string weight_key,
        string bias_key,
        int *out_lastdim,
        int overrided_dim2){

    int OverridedK = (overrided_dim2==-1)?K:overrided_dim2;
    vector<size_t> w_shape = weightsshape_map[weight_key]; //sth like 1,1,6,64
    float* w_ptr = weights_map[weight_key];

    //------------------------------------------------------
    vector<size_t> b_shapes = weightsshape_map[bias_key];
    float* b_ptr = weights_map[bias_key];

    //------------------------------------------------------
    int ch_out = (int)w_shape.back();
    *out_lastdim = ch_out;
    float* rslt = new float[B*N*OverridedK*ch_out];
    int D = input_last_dim;
    int indxS1,indxS2,indxD;

    //------------------------------------------------------
    for(int b=0;b<B;b++){
        for(int n=0;n<N;n++){
            for(int k=0;k<OverridedK;k++){
                indxS1 = b*N*OverridedK*D + n*OverridedK*D + k*D + 0;
                for(int ch=0;ch<ch_out;ch++){
                    float sum=0;
                    for(int d=0;d<D;d++){
                        indxS2 = d*ch_out + ch;
                        //if(indxS1+d >= B*N*OverridedK*input_last_dim){
                        //    cout<< "indx1: " << (indxS1+d)<<endl;
                        //    cout<< "indx2: " << (indxS2)<<endl<<endl;
                        //}
                        sum += input[indxS1+d] * w_ptr[indxS2];
                    }
                    indxD=b*N*OverridedK*ch_out+ n*OverridedK*ch_out+ k*ch_out+ ch;

                    //if(indxD >= B*N*OverridedK*ch_out){
                    //    cout<< "indx3: " << (indxD)<<endl;
                    //}

                    rslt[indxD] = sum + b_ptr[ch];
                }
            }
        }
    }

    //------------------------------------------------------
    return rslt;

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

float* ModelArch01::ReLU(
        float* input,
        int dim){

    float* tmp = new float[dim];

    for(int i=0;i<dim;i++){
        tmp[i] = (input[i]>0)?input[i]:0;
    }
    return tmp;
}









float* ModelArch01::LA_transpose(float* input,int batchsize, int matrix_rank, int matrixH, int matrixW){
    if(matrix_rank!=3){cout<<"LA_transpose: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    float* rslt = new float[batchsize*matrixW*matrixH];
    int indxS=0;
    int indxD=0;

    for(int b=0;b<batchsize;b++)
    {
        for (int j = 0; j < matrixH; j++) {
            for (int i = 0; i < matrixW; i++) {
                indxS = b * matrixH * matrixW + j * matrixW + i;
                indxD = b * matrixH * matrixW + i * matrixH + j;
                rslt[indxD] = input[indxS];
            }
        }
    }
    return rslt;
}

// rslt = MAT1 * MAT2
float* ModelArch01::LA_MatMul(float* mat1,float* mat2,
                            int batchsize, int matrix_rank,
                            int matrixH1,int matrixW1,
                            int matrixH2,int matrixW2){
    if(matrix_rank!=3){cout<<"LA_MatMul: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    if(matrixW1!=matrixH2){cout<<"LA_MatMul: ERROR_BAD_MATRIX_DIMs"<<endl;return nullptr;}
    float* rslt = new float[batchsize*matrixH1*matrixW2];
    int indxS1=0;
    int indxS2=0;
    int indxD=0;

    for(int b=0;b<batchsize;b++) {
        // for element of output of matrixH1 x matrixW2
        for(int j=0;j<matrixH1;j++){
            for(int i=0;i<matrixW2;i++){
                //mat1: select row j
                //mat2: select col i
                float sum=0;
                for(int mat1_x=0;mat1_x<matrixW1;mat1_x++)
                {
                    indxS1 = b*matrixH1*matrixW1 +
                             j*matrixW1 + mat1_x;
                    /*indxS2 = b*matrixH2*matrixW2 +
                             mat1_x*matrixW1 + j;*/
                    indxS2 = b*matrixH2*matrixW2 +
                             mat1_x*matrixW2 + i;

                    sum += mat1[indxS1] * mat2[indxS2];
                }
                // for element of output of matrixH1 x matrixW2
                indxD = b*matrixH1*matrixW2 +
                        j*matrixW2 + i;
                rslt[indxD] = sum;

            }
        }
    }
    return rslt;
}

// rslt = MAT1 * scalar
float* ModelArch01::LA_MatMul(float* mat1,float scalar,
                            int batchsize, int matrix_rank,
                            int matrixH1,int matrixW1){
    if(matrix_rank!=3){cout<<"LA_MatMul: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    float* rslt = new float[batchsize*matrixH1*matrixW1];
    int limit = batchsize*matrixH1*matrixW1;

    for(int b=0;b<limit;b++) {
        rslt[b] = mat1[b] * scalar;
    }
    return rslt;
}

float* ModelArch01::LA_Square(float* mat1,
                            int batchsize, int matrix_rank,
                            int matrixH1,int matrixW1){
    if(matrix_rank!=3){cout<<"LA_MatMul: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    float* rslt = new float[batchsize*matrixH1*matrixW1];
    int limit = batchsize*matrixH1*matrixW1;

    for(int b=0;b<limit;b++) {
        rslt[b] = mat1[b] * mat1[b];
    }
    return rslt;
}

//[axis0,axis1,axis2]
float* ModelArch01::LA_Sum(float* mat1,
                         bool over_axis0,
                         bool over_axis1,
                         bool over_axis2,
                         int dim0,
                         int dim1,
                         int dim2){
    int indxS=0;
    int indxD=0;
    float* rslt;


    if(over_axis0==true &&
       over_axis1==true &&
       over_axis2==true )
    {
        float sum = 0;
        rslt = &sum;
        int limit = dim0*dim1*dim2;

        for(int b=0;b<limit;b++) {
            (*rslt) += mat1[b];
        }
        return rslt;
    }

    if(over_axis0==true &&
       over_axis1==false &&
       over_axis2==false )
    {
        rslt = new float[dim2*dim1];
        float sum=0;
        for(int d1=0; d1<dim1;d1++){
            for(int d2=0;d2<dim2;d2++){
                sum=0;
                indxD = d1 * dim2 + d2;
                //sum over dim of interest
                for(int dx=0;dx<dim0;dx++)
                {
                    indxS = dx * dim1*dim2 + d1 * dim2 + d2;
                    sum+=mat1[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        return rslt;
    }

    if(over_axis0==false &&
       over_axis1==true &&
       over_axis2==false )
    {
        rslt = new float[dim2*dim0];
        float sum=0;
        for(int d0=0; d0<dim0;d0++){
            for(int d2=0;d2<dim2;d2++){
                sum=0;
                indxD = d0 *dim2 + d2;
                //sum over dim of interest
                for(int dx=0;dx<dim1;dx++)
                {
                    indxS = d0 * dim1*dim2 + dx * dim2 + d2;
                    sum+=mat1[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        return rslt;
    }

    if(over_axis0==false &&
       over_axis1==false &&
       over_axis2==true )
    {
        rslt = new float[dim1*dim0];
        float sum=0;
        for(int d0=0; d0<dim0;d0++){
            for(int d1=0;d1<dim1;d1++){
                sum=0;
                indxD = d0 * dim1 + d1;
                //sum over dim of interest
                for(int dx=0;dx<dim2;dx++)
                {
                    indxS = d0 * dim1*dim2 + d1 * dim2 + dx;
                    sum+=mat1[indxS] ;
                }
                rslt[indxD] = sum;
            }
        }
        return rslt;
    }

    cout<<"LA_SUM:ERROR_UNDEFINED_AXES_COMB"<<endl;
    return nullptr;
}

//[axis0,axis1,axis2,axis3] //No batch op, uses data as is(as a matrix)
float* ModelArch01::LA_Sum4D(float* mat1,
                           bool over_axis0,
                           bool over_axis1,
                           bool over_axis2,
                           bool over_axis3,
                           int dim0,
                           int dim1,
                           int dim2,
                           int dim3){
    //cout<<"**LA_SUM4D: "<<dim0<<","<<dim1<<","<<dim2<<","<<dim3<<";\n";
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


float* ModelArch01::LA_Mean(
        float* mat,
        int rank,
        bool mean_axis0,
        bool mean_axis1,
        bool mean_axis2,
        bool mean_axis3,
        int dim0,
        int dim1,
        int dim2,
        int dim3

){
    //cout      <<"**LA_Mean: Rank: "<< rank << "  dims: "<<dim0<<","<<dim1<<","<<dim2<<","<<dim3<<
    //            "  overaxes: "<<mean_axis0<<","<<mean_axis1<<","<<mean_axis2<<","<<mean_axis3<<";\n";

    if(rank==4){
        if(!mean_axis3 && mean_axis0 && mean_axis1 && mean_axis2){
            float* sum = LA_Sum4D(mat,
                                  mean_axis0,
                                  mean_axis1,
                                  mean_axis2,
                                  mean_axis3,
                                  dim0,
                                  dim1,
                                  dim2,
                                  dim3);
            float *mean = new float[dim3];

            for(int d3=0;d3<dim3;d3++){
                mean[d3] = (sum[d3])/(float)(dim0*dim1*dim2);
            }
            free(sum);
            return mean;
        }
        cout<<"LA_MEAN: ERROR_UNDEFINED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==2){ //dim0 is batch, dim1 is fc layer output, ex.: for B=1 --> output=[1,256]
        if(!mean_axis1 && mean_axis0 ){
            float* sum = LA_Sum(mat,false,true,false,1,dim0,dim1); //result is of shape dim1
            float *mean ;
            mean = LA_MatMul(sum,(1.0f/dim0),1,3,1,dim1);
            free(sum);
            return mean;
        }
        cout<<"LA_MEAN: ERROR_UNDEFINED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==1){
        float* sum = LA_Sum(mat,true,true,true,1,1,dim0);
        float *mean = (float*)malloc(sizeof(float) * 1);
        *mean = (*sum)/(float)(dim0);
        free(sum);
        return mean;
    }
    cout<<"LA_MEAN: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
}

float* ModelArch01::LA_Variance(
        float* mat,
        int rank,
        bool variance_axis0,
        bool variance_axis1,
        bool variance_axis2,
        bool variance_axis3,
        int dim0,
        int dim1,
        int dim2,
        int dim3

){
    //cout      <<"**LA_Variance: Rank: "<< rank << "  dims: "<<dim0<<","<<dim1<<","<<dim2<<","<<dim3<<
    //          "  overaxes: "<<variance_axis0<<","<<variance_axis1<<","<<variance_axis2<<","<<variance_axis3<<";\n";
    if(rank==4){
        if(!variance_axis3 && variance_axis0 && variance_axis1 && variance_axis2) {
            float *mean = LA_Mean(mat,
                                  rank,
                                  variance_axis0,
                                  variance_axis1,
                                  variance_axis2,
                                  false,
                                  dim0,
                                  dim1,
                                  dim2,
                                  dim3);

            float* variance = new float[dim3];

            int indxS1,indxS2,indxD;
            for (int d3 = 0; d3 < dim3; d3++) { //over last-dim
                variance[d3]=0;


                for (int d0 = 0; d0 < dim0; d0++) {
                    for (int d1 = 0; d1 < dim1; d1++) {
                        for (int d2 = 0; d2 < dim2; d2++) {
                            indxS1 = d0*dim1*dim2*dim3+
                                     d1*dim2*dim3+
                                     d2*dim3+
                                     d3;

                            float delta = (mat[indxS1]-mean[d3]);
                            variance[d3] += delta*delta;
                        }
                    }
                }
            }

            float* variance_final = LA_MatMul(variance,(float)(1.0f/(dim0*dim1*dim2)),1,3,1,dim3);

            free(variance);
            free(mean);

            return variance_final;
        }
        cout<<"LA_MEAN: ERROR_UNDEFINED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==2){
        if(!variance_axis1 && variance_axis0 ) {
            float *mean = LA_Mean(mat,
                                  2,
                                  true,
                                  false,
                                  false,
                                  false,
                                  dim0,
                                  dim1,
                                  0,
                                  0);

            float* variance = new float[dim1];

            int indxS1,indxS2,indxD;
            for (int d1 = 0; d1 < dim1; d1++) { //over last-dim
                variance[d1]=0;

                for (int d0 = 0; d0 < dim0; d0++) {
                    indxS1 = d0*dim1 + d1;

                    float delta = (mat[indxS1]-mean[d1]);
                    variance[d1] += delta*delta;
                }
            }

            float* variance_final = LA_MatMul(variance,(float)(1.0f/dim0),1,3,1,dim1);

            free(variance);
            free(mean);

            return variance_final;
        }
        cout<<"LA_MEAN: ERROR_UNDEFINED_AXES_COMB"<<endl;
        return nullptr;
    }

    if(rank==1){
        float *mean = LA_Mean(mat,
                              rank,
                              true,
                              true,
                              true,
                              true,
                              dim0,
                              1,
                              1,
                              1);

        float *variance = (float*)malloc(sizeof(float) * 1);

        for (int d0 = 0; d0 < dim0; d0++) {
            float delta = (mat[d0]-mean[0]);
            *variance += delta*delta;
        }

        *variance = *variance * (float)(dim0);
        return variance;

    }
    cout<<"LA_MEAN: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
}

float* ModelArch01::LA_ADD(
        float* mat1,
        float* mat2,
        int    rank,
        int    dim0,
        int    dim1,
        int    dim2
        ){
    if(rank!=3){cout<<"LA_ADD: ERROR_BAD_MATRIX_RANK"<<endl;return nullptr;}
    int limit = dim0*dim1*dim2;

    float* rslt=new float[limit];

    for(int d0=0;d0<limit;d0++){
        rslt[d0] = mat1[d0] + mat2[d0];
    }

    return rslt;
}

//mat1 - mat2
float* ModelArch01::LA_SUB(
        float* mat1,
        float* mat2,
        int    rank,
        int    dim0,
        int    dim1,
        int    dim2,
        int    dim3
){
    int limit ;
    if(rank==3) {
        limit = dim0 * dim1 * dim2;

        float *rslt = new float[limit];

        for (int d0 = 0; d0 < limit; d0++) {
            rslt[d0] = mat1[d0] - mat2[d0];
        }
        return rslt;
    }

    if(rank==4){
        limit = dim0 * dim1 * dim2*dim3;

        float *rslt = new float[limit];

        for (int d0 = 0; d0 < limit; d0++) {
            rslt[d0] = mat1[d0] - mat2[d0];
        }
        return rslt;
    }

    cout<<"LA_SUB: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
}

//concat 2 matrices
// [matA, matB]
float* ModelArch01::LA_Concat2(
        float* matA,
        float* matB,
        int rank,
        int concat_dim,
        int dimA0,
        int dimA1,
        int dimA2,
        int dimA3,
        int dimB0,
        int dimB1,
        int dimB2,
        int dimB3
        ){
    cout      <<"**LA_Concat2: Rank: "<< rank << "  concatDim: "<<concat_dim<<
              "  dimA: "<<dimA0<<","<<dimA1<<","<<dimA2<<","<<dimA3<<
              "  dimB: "<<dimB0<<","<<dimB1<<","<<dimB2<<","<<dimB3<<";\n";
    if(rank==4){
        int dimR0,dimR1,dimR2,dimR3;
        int mat2_offset_dim0=0;
        int mat2_offset_dim1=0;
        int mat2_offset_dim2=0;
        int mat2_offset_dim3=0;

        if(concat_dim==0){
            dimR0 = dimA0 + dimB0;
            dimR1 = dimA1;
            dimR2 = dimA2;
            dimR3 = dimA3;
            mat2_offset_dim0=dimA0;
        }
        if(concat_dim==1){
            dimR0 = dimA0;
            dimR1 = dimA1 + dimB1;
            dimR2 = dimA2;
            dimR3 = dimA3;
            mat2_offset_dim1=dimA1;
        }
        if(concat_dim==2){
            dimR0 = dimA0;
            dimR1 = dimA1;
            dimR2 = dimA2 + dimB2;
            dimR3 = dimA3;
            mat2_offset_dim2=dimA2;
        }
        if(concat_dim==3){
            dimR0 = dimA0;
            dimR1 = dimA1;
            dimR2 = dimA2;
            dimR3 = dimA3 + dimB3;
            mat2_offset_dim3=dimA3;
        }

        float* rslt = new float[dimR0*dimR1*dimR2*dimR3];
        int indxS1,indxS2,indxD;

        for(int d0=0;d0<dimA0;d0++){
            for(int d1=0;d1<dimA1;d1++){
                for(int d2=0;d2<dimA2;d2++){
                    for(int d3=0;d3<dimA3;d3++){
                        indxS1 = d0*dimA1*dimA2*dimA3 +
                                 d1*dimA2*dimA3+
                                 d2*dimA3+
                                 d3;
                        indxD = (d0)*dimR1*dimR2*dimR3 +
                                (d1)*dimR2*dimR3+
                                (d2)*dimR3+
                                (d3);
                        rslt[indxD] = matA[indxS1];
                    }
                }
            }
        }

        for(int d0=0;d0<dimB0;d0++){
            for(int d1=0;d1<dimB1;d1++){
                for(int d2=0;d2<dimB2;d2++){
                    for(int d3=0;d3<dimB3;d3++){
                        indxS2 = d0*dimB1*dimB2*dimB3 +
                                 d1*dimB2*dimB3+
                                 d2*dimB3+
                                 d3;
                        indxD  = (d0+mat2_offset_dim0)*dimR1*dimR2*dimR3 +
                                 (d1+mat2_offset_dim1)*dimR2*dimR3+
                                 (d2+mat2_offset_dim2)*dimR3+
                                 (d3+mat2_offset_dim3);
                        rslt[indxD] = matB[indxS2];
                    }
                }
            }
        }

        return rslt;


    }

    cout<<"LA_Concat2: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
}

float* ModelArch01::LA_ReduceMax(
        float* input,
        int axis,
        int rank,
        int dim0,
        int dim1,
        int dim2,
        int dim3){
    //cout<< "reduceMax: "<< "Rank: " << rank << ", Axis: "<< axis <<", Dim: " << dim0 << "x" << dim1 << "x" << dim2 << "x" << dim3 << endl;
    if(rank==4){
        if(axis==3){
            float* rslt= new float[dim0*dim1*dim2];
            int indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= -numeric_limits<float>::infinity();

            for(int d0=0;d0<dim0;d0++){
                for(int d1=0;d1<dim1;d1++){
                    for(int d2=0;d2<dim2;d2++){
                        indxD = d0*dim1*dim2+
                                d1*dim2+
                                d2;
                        max = max_cte;
                        for(int d3=0;d3<dim3;d3++){
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<input[indxS]){
                                max = input[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            return rslt;
        }




        if(axis==2){
            float* rslt= new float[dim0*dim1*dim3];
            int indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= 0;

            for(int d0=0;d0<dim0;d0++){
                for(int d1=0;d1<dim1;d1++){
                    for(int d3=0;d3<dim3;d3++){
                        indxD = d0*dim1*dim3+
                                d1*dim3+
                                d3;
                        max = max_cte;

                        for(int d2=0;d2<dim2;d2++)
                        {
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<input[indxS]){
                                max = input[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            return rslt;
        }

        if(axis==1){
            float* rslt= new float[dim0*dim2*dim3];
            int indxS,indxD;
            float max_cte= -numeric_limits<float>::infinity();
            float max= 0;

            for(int d0=0;d0<dim0;d0++){
                for(int d2=0;d2<dim2;d2++){
                    for(int d3=0;d3<dim3;d3++){
                        indxD = d0*dim2*dim3+
                                d2*dim3+
                                d3;
                        max = max_cte;


                        for(int d1=0;d1<dim1;d1++)
                        {
                            indxS = d0*dim1*dim2*dim3+
                                    d1*dim2*dim3+
                                    d2*dim3+
                                    d3;
                            if(max<input[indxS]){
                                max = input[indxS];
                            }
                        }
                        rslt[indxD]=max;
                    }
                }
            }
            return rslt;
        }
        cout<<"LA_ReduceMax: ERROR_UNDEFINED_POOLING_AXIS"<<endl;
        return nullptr;
    }

    cout<<"LA_ReduceMax: ERROR_UNDEFINED_MATRIX_RANK"<<endl;
    return nullptr;
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
