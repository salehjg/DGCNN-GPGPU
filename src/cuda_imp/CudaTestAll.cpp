//
// Created by saleh on 8/28/18.
//

#include "cuda_imp/CudaTestAll.h"
#include "UnitTest++/UnitTest++.h"

CudaTestAll *cudaTestAll;
WorkScheduler scheduler;

int main(){
    cudaTestAll = new CudaTestAll(0,5,1024,20);
    cudaTestAll->SetModelInput_data(REPO_DIR "/data/dataset/dataset_B2048_pcl.npy");
    cudaTestAll->SetModelInput_labels(REPO_DIR"/data/dataset/dataset_B2048_labels_int32.npy");

    UnitTest::RunAllTests();
}

SUITE(SingleOperandKernels){

    TEST(Kernel_Transpose){
        TensorF* tensorCpu = cudaTestAll->platformSelector->Transpose(PLATFORMS::CPU,scheduler,cudaTestAll->input_pcl_BxNxD);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Transpose(PLATFORMS::GPU_CUDA,scheduler,cudaTestAll->input_pcl_BxNxD);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Kernel_Square){
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{10,50,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Square(PLATFORMS::CPU,scheduler,tensorSrc1);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Square(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Kernel_Sqrt){
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{10,50,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Sqrt(PLATFORMS::CPU,scheduler,tensorSrc1);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Sqrt(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Kernel_Relu){
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{10,50,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->ReLU(PLATFORMS::CPU,scheduler,tensorSrc1);
        TensorF* tensorGpu = cudaTestAll->platformSelector->ReLU(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

}

SUITE(Kernel_MatMul){
    TEST(Kernel_Matmul1){
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(0,{1,50,20});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(0,{1,20,60});
        TensorF* tensorCpu = cudaTestAll->platformSelector->MatMul(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2);
        TensorF* tensorGpu = cudaTestAll->platformSelector->MatMul(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tensorSrc2);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Kernel_Matmul2){
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{25,1024});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{1024,512});
        TensorF* tensorCpu = cudaTestAll->platformSelector->MatMul(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2);
        TensorF* tensorGpu = cudaTestAll->platformSelector->MatMul(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tensorSrc2);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_MatOps){
    TEST(Rank_4_4){
        //Ranks: 4,4
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(4,{11,12,13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{11,12,13,14});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tensorSrc2,op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_4_3){
        //Ranks: 4,3
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{11,12,13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(5,{12,13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_4_2){
        //Ranks: 4,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{11,12,13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_4_1){
        //Ranks: 4,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{11,12,13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_4_0){
        //Ranks: 4,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{11,12,13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_3_3){
        //Ranks: 3,3
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{12,13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(5,{12,13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_3_2){
        //Ranks: 3,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{12, 13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_3_1){
        //Ranks: 3,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{12,13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_3_0){
        //Ranks: 3,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{12,13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_2_2){
        //Ranks: 2,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_2_1){
        //Ranks: 2,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{13,14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_2_0){
        //Ranks: 2,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_1_1){
        //Ranks: 1,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{14});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_1_0){
        //Ranks: 1,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_1_0V2){
        //Ranks: 1,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{64});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(ETC_1){
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(0,{5,1024,1024});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,0.5f,op);
            TensorF* tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,0.5f,op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(ETC_2){
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(0,{5,1024,1024});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(0,{5,1024,1024});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tensorSrc2,op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(ETC_3){
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(0,{8,256,256});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(0,{8,256,256});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = cudaTestAll->platformSelector->MatOps(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tensorSrc2,op);
            bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }
}

SUITE(Kernel_Concat) {
    TEST (test_1) {
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{5,2,50,20});
        TensorF* tensorSrc2 = cudaTestAll->GenerateTensor(3,{5,2,50,30});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Concat2(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,3);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Concat2(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tensorSrc2,3);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Tile){
    TEST(Rank4_Axis2){
        int tileCount = 8;
        int tileAxis  = 2;
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{5,2,1,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Tile(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tileAxis,tileCount);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Rank3_Axis2){
        int tileCount = 8;
        int tileAxis  = 2;
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{5,20,1});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Tile(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tileAxis,tileCount);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Rank3_Axis1){
        int tileCount = 8;
        int tileAxis  = 1;
        TensorF* tensorSrc1 = cudaTestAll->GenerateTensor(3,{5,1,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Tile(PLATFORMS::GPU_CUDA,scheduler,tensorSrc1,tileAxis,tileCount);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Reduce_Max){

    TEST (Rank4_OverAxis2){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(0,{5,2,50,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->ReduceMax(PLATFORMS::CPU,scheduler,tensorSrc,2);
        TensorF* tensorGpu = cudaTestAll->platformSelector->ReduceMax(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,2);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST (Rank4_OverAxis1){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(0,{5,5,1,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->ReduceMax(PLATFORMS::CPU,scheduler,tensorSrc,1);
        TensorF* tensorGpu = cudaTestAll->platformSelector->ReduceMax(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,1);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_ReduceSum_3D){

    TEST(Rank3_OverAxis0){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(1,{50,25,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc,true,false,false);
        TensorF* tensorGpu = cudaTestAll->platformSelector->ReduceSum(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,true,false,false);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Rank3_OverAxis1){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(1,{50,25,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc,false,true,false);
        TensorF* tensorGpu = cudaTestAll->platformSelector->ReduceSum(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,false,true,false);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Rank3_OverAxis2){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(1,{50,25,50});
        TensorF* tensorCpu = cudaTestAll->platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc,false,false,true);
        TensorF* tensorGpu = cudaTestAll->platformSelector->ReduceSum(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,false,false,true);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_ReduceSum_4D){

    TEST(Rank4_TTTF){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(1,{5,1024,20,256});
        TensorF* tensorCpu = cudaTestAll->platformSelector->ReduceSum4D(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = cudaTestAll->platformSelector->ReduceSum4D(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,true,true,true,false);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Mean){

    TEST(Rank4_TTTF){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(1,{5,1024,20,256});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Mean(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Mean(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,true,true,true,false);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Variance){

    TEST(Rank4_TTTF){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(1,{5,1024,20,256});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Variance(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Variance(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,true,true,true,false);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Conv2D_Mlp){

    TEST(Test_1){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(3,{5,3,3,6});
        TensorF* tensorWeight = cudaTestAll->GenerateTensor(3,{1,1,6,7});
        TensorF* tensorBiases = cudaTestAll->GenerateTensor(-1,{7}); //THE SHAPE SHOULD BE 1D, NOT 4D LIKE {1,1,1,7}
        TensorF* tensorCpu = cudaTestAll->platformSelector->Conv2D(PLATFORMS::CPU,scheduler,tensorSrc,tensorWeight,tensorBiases);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Conv2D(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,tensorWeight,tensorBiases);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Gather){

    TEST(Test_1){
        TensorF* tensorSrc = cudaTestAll->GenerateTensor(3,{5,1024,6});
        TensorI* tensorIndices = cudaTestAll->GenerateTensor(0,1023,{5,1024,20});
        TensorF* tensorCpu = cudaTestAll->platformSelector->Gather(PLATFORMS::CPU,scheduler,tensorSrc,tensorIndices,1);
        TensorF* tensorGpu = cudaTestAll->platformSelector->Gather(PLATFORMS::GPU_CUDA,scheduler,tensorSrc,tensorIndices,1);
        bool comparisonResult = cudaTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_TopK){

    TEST(Test_1) {
        TensorF *tensorSrc = cudaTestAll->GenerateTensor(0, {5, 1024, 1024});
        TensorI *tensorCpu = cudaTestAll->platformSelector->TopK(PLATFORMS::CPU, scheduler, tensorSrc, 2, 20);
        TensorI *tensorGpu = cudaTestAll->platformSelector->TopK(PLATFORMS::GPU_CUDA, scheduler, tensorSrc, 2, 20);
        bool comparisonResult = true;
        if (tensorCpu->getShape() != tensorGpu->getShape()){
            comparisonResult = false;
        }
        else{
            unsigned long len = tensorCpu->getLength();
            TensorI *tensorGpuTransfered = cudaTestAll->platformSelector->CrossThePlatform(tensorGpu,PLATFORMS::CPU);
            for (unsigned long i = 0; i < len; i++) {
                int rCpu = tensorCpu->_buff[i];
                int rGpu = tensorGpuTransfered->_buff[i];
                if(rCpu != rGpu){
                    comparisonResult=false;
                    break;
                }
            }
        }
        CHECK_EQUAL(comparisonResult, true);
    }
}

CudaTestAll::CudaTestAll(int dataset_offset, int batchsize, int pointcount, int knn_k) {
    platformSelector = new PlatformSelector(PLATFORMS::CPU,{PLATFORMS::CPU,PLATFORMS::GPU_CUDA});
    DB_OFFSET = dataset_offset;
    B = (unsigned int)batchsize;
    N = (unsigned int)pointcount;
    K = (unsigned int)knn_k;
}

void CudaTestAll::SetModelInput_data(string npy_pcl) {
    _npy_pcl = cnpy::npy_load(npy_pcl);
    input_pcl_BxNxD = new TensorF({B,N,3}, _npy_pcl.data<float>());
    input_pcl_BxNxD += DB_OFFSET*N*3;
    assert( N == (int)(_npy_pcl.shape[1]) );
}

void CudaTestAll::SetModelInput_labels(string npy_labels) {
    // dataType of npy file should be int32, NOT uchar8!
    // use dataset_B5_labels_int32.npy
    _npy_labels = cnpy::npy_load(npy_labels);
    input_labels_B = new TensorI({B},_npy_labels.data<int>());
    input_labels_B->_buff += DB_OFFSET;
}

TensorI* CudaTestAll::GetLabels(){
    return input_labels_B;
}

TensorF* CudaTestAll::GenerateTensor(int pattern, vector<unsigned int> shape){
    TensorF *testTn = new TensorF(shape);
    unsigned long _len = testTn->getLength();
    if(pattern==-1){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = 0;
        }
    }
    if(pattern==0){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = float_rand(0,2.50f);
        }
    }
    if(pattern==1){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = i %5 ;//+ float_rand(0,2.50f);
        }
    }
    if(pattern==2){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = i %10 + float_rand(0,2.50f);
        }
    }
    if(pattern==3){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = pattern;
        }
    }
    if(pattern==4){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = pattern;
        }
    }
    if(pattern==5){
        for (unsigned long i = 0; i < _len; i++) {
            testTn->_buff[i] = pattern;
        }
    }
    return testTn;
}

TensorI* CudaTestAll::GenerateTensor(int intMin, int intMax, vector<unsigned int> shape){
    TensorI *testTn = new TensorI(shape);
    unsigned long _len = testTn->getLength();

    for (unsigned long i = 0; i < _len; i++) {
        testTn->_buff[i] = (int)float_rand((float)intMin,(float)intMax);
    }

    return testTn;
}

float CudaTestAll::float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}