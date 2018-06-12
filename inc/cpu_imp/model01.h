//
// Created by saleh on 5/31/18.
//

#ifndef DEEPPOINTV1_MODEL01_H
#define DEEPPOINTV1_MODEL01_H

#include <string>
using namespace std;

//#define DUMP_ENABLED
//#undef DUMP_ENABLED
#undef DUMP_ENABLED

struct ModelInfo{
    string Version="";
    string DesignNotes="";
    string ExperimentNotes="";
    string ToDo="";
    string Date="";
};

class ModelArch {


public:
    typedef float DType;
    ModelArch(int batchsize, int pointcount, int knn_k);
    ModelInfo   GetModelInfo();
    void        SetModelInput_data(string npy_pcl);
    void        SetModelInput_labels(string npy_labels);
    int         LoadWeights(string base_dir,string path_txt_fnamelist);
    float       execute();
    float*      Pairwise_Distance(float* input_BxNxD,int input_last_dim);
    float*      Get_Edge_Features(float* input_BxNxD,
                                  int*   knn_output_BxNxK,
                                  int    D);
    int*        KNN(float* adj_matrix_BxNxN);

    float*      Conv2D(float* input,
                       int input_last_dim,
                       string weight_key,
                       string bias_key,
                       int *out_lastdim,
                       int overrided_dim2=-1);

    float* Batchnorm_Forward(
            float* input,
            string gamma_key,
            string beta_key,
            string ema_ave_key,
            string ema_var_key,
            int rank,
            int dim0,
            int dim1,
            int dim2,
            int dim3);


    float* FullyConnected_Forward(
            float* input_BxN,
            string weight_key,
            string bias_key,
            int input_last_dim);


    float* ReLU(float* input,int dim);

    float* TransformNet(
            float* edge_features, //of shape B x N x K x InputLastDim
            int k);

    //-------------------------------------------------------------

    template<typename T> int DumpMatrix(
            string npy_fname,
            int rank,
            T* mat,
            int dim0,
            int dim1,
            int dim2,
            int dim3,
            int dim4,
            string npy_dir="/home/saleh/00_repos/tensorflow_repo/00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/data/matrix_dumps/");

private:
    int B=-1;
    int N=-1;
    int K=-1;
    map<string,float*> weights_map;
    map<string, vector<size_t> > weightsshape_map;
    float* input_pcl_BxNxD;
    unsigned char* input_labels_B;
    cnpy::NpyArray _npy_pcl;
    cnpy::NpyArray _npy_labels;
    vector<cnpy::NpyArray> _weights_vector;


    float*      LA_transpose(float* input,int batchsize, int matrix_rank, int matrixH, int matrixW);

    float*      LA_MatMul(float* mat1,float* mat2,
                          int batchsize, int matrix_rank,
                          int matrixH1,int matrixW1,
                          int matrixH2,int matrixW2);

    float*      LA_MatMul(float* mat1,float scalar,
                          int batchsize, int matrix_rank,
                          int matrixH1,int matrixW1);

    float*      LA_Square(float* mat1,
                          int batchsize, int matrix_rank,
                          int matrixH1,int matrixW1);

    float*      LA_Sum(float* mat1,
                       bool over_axis0,
                       bool over_axis1,
                       bool over_axis2,
                       int dim0,
                       int dim1,
                       int dim2);

    float*      LA_Sum4D(float* mat1,
                         bool over_axis0,
                         bool over_axis1,
                         bool over_axis2,
                         bool over_axis3,
                         int dim0,
                         int dim1,
                         int dim2,
                         int dim3);
    float* LA_Mean(
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

    );

    float* LA_Variance(
            float* mat,
            int rank,
            bool mean_axis0,
            bool mean_axis1,
            bool mean_axis2,
            bool mean_axis3,
            int dim0,
            int dim1,
            int dim2,
            int dim3);

    float*      LA_ADD(
            float* mat1,
            float* mat2,
            int    rank,
            int    dim0,
            int    dim1,
            int    dim2
    );

    float*      LA_SUB(
            float* mat1,
            float* mat2,
            int    rank,
            int    dim0,
            int    dim1,
            int    dim2,
            int    dim3
    );

    float*      LA_Concat2(
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
    );


    float* LA_ReduceMax(
            float* input,
            int axis,
            int rank,
            int dim0,
            int dim1,
            int dim2,
            int dim3);

};


#endif //DEEPPOINTV1_MODEL01_H
