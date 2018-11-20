//
// Created by saleh on 8/28/18.
//

#include "../../inc/ocl_imp/oclTestAll.h"
#include "UnitTest++/UnitTest++.h"

OclTestAll *oclTestAll;
WorkScheduler scheduler;

int main(){
    oclTestAll = new OclTestAll(0,5,1024,20);
    oclTestAll->SetModelInput_data(REPO_DIR "/data/dataset/dataset_B2048_pcl.npy");
    oclTestAll->SetModelInput_labels(REPO_DIR"/data/dataset/dataset_B2048_labels_int32.npy");

    UnitTest::RunAllTests();
}

SUITE(SingleOperandKernels){

    TEST(Kernel_Transpose){
        TensorF* tensorCpu = oclTestAll->platformSelector->Transpose(PLATFORMS::CPU,scheduler,oclTestAll->input_pcl_BxNxD);
        TensorF* tensorGpu = oclTestAll->platformSelector->Transpose(PLATFORMS::GPU_OCL,scheduler,oclTestAll->input_pcl_BxNxD);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Kernel_Square){
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{10,50,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->Square(PLATFORMS::CPU,scheduler,tensorSrc1);
        TensorF* tensorGpu = oclTestAll->platformSelector->Square(PLATFORMS::GPU_OCL,scheduler,tensorSrc1);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Kernel_Sqrt){
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{10,50,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->Sqrt(PLATFORMS::CPU,scheduler,tensorSrc1);
        TensorF* tensorGpu = oclTestAll->platformSelector->Sqrt(PLATFORMS::GPU_OCL,scheduler,tensorSrc1);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Kernel_Relu){
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{10,50,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->ReLU(PLATFORMS::CPU,scheduler,tensorSrc1);
        TensorF* tensorGpu = oclTestAll->platformSelector->ReLU(PLATFORMS::GPU_OCL,scheduler,tensorSrc1);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

}

SUITE(Kernel_MatMul){
    TEST(Kernel_Matmul1){
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(0,{1,50,20});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(0,{1,20,60});
        TensorF* tensorCpu = oclTestAll->platformSelector->MatMul(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2);
        TensorF* tensorGpu = oclTestAll->platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Kernel_Matmul2){
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{25,1024});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{1024,512});
        TensorF* tensorCpu = oclTestAll->platformSelector->MatMul(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2);
        TensorF* tensorGpu = oclTestAll->platformSelector->MatMul(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_MatOps){
    TEST(Rank_4_4){
        //Ranks: 4,4
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(4,{11,12,13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{11,12,13,14});

        for(MAT_OPS op : ops){
            TensorF* tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,op);
            TensorF* tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_4_3){
        //Ranks: 4,3
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{11,12,13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(5,{12,13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_4_2){
        //Ranks: 4,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{11,12,13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_4_1){
        //Ranks: 4,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{11,12,13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_4_0){
        //Ranks: 4,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{11,12,13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_3_3){
        //Ranks: 3,3
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{12,13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(5,{12,13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_3_2){
        //Ranks: 3,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{12, 13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_3_1){
        //Ranks: 3,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{12,13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_3_0){
        //Ranks: 3,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{12,13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_2_2){
        //Ranks: 2,2
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_2_1){
        //Ranks: 2,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{13,14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_2_0){
        //Ranks: 2,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{13,14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_1_1){
        //Ranks: 1,1
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{14});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, tensorSrc2,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1,
                                                                      tensorSrc2, op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }

    TEST(Rank_1_0){
        //Ranks: 1,0
        vector<MAT_OPS> ops = {MAT_OPS::ADD, MAT_OPS::SUB, MAT_OPS::MUL_ELEMENTWISE, MAT_OPS::DIV_ELEMENTWISE};
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{14});
        for(MAT_OPS op : ops) {
            TensorF *tensorCpu = oclTestAll->platformSelector->MatOps(PLATFORMS::CPU, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            TensorF *tensorGpu = oclTestAll->platformSelector->MatOps(PLATFORMS::GPU_OCL, scheduler, tensorSrc1, 1.5f,
                                                                      op);
            bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU, scheduler, tensorCpu,
                                                                                 tensorGpu);
                    CHECK_EQUAL(comparisonResult, true);
        }
    }
}

SUITE(Kernel_Concat) {
    TEST (test_1) {
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{5,2,50,20});
        TensorF* tensorSrc2 = oclTestAll->GenerateTensor(3,{5,2,50,30});
        TensorF* tensorCpu = oclTestAll->platformSelector->Concat2(PLATFORMS::CPU,scheduler,tensorSrc1,tensorSrc2,3);
        TensorF* tensorGpu = oclTestAll->platformSelector->Concat2(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tensorSrc2,3);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Tile){
    TEST(Rank4_Axis2){
        int tileCount = 8;
        int tileAxis  = 2;
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{5,2,1,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = oclTestAll->platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tileAxis,tileCount);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Rank3_Axis2){
        int tileCount = 8;
        int tileAxis  = 2;
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{5,20,1});
        TensorF* tensorCpu = oclTestAll->platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = oclTestAll->platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tileAxis,tileCount);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Rank3_Axis1){
        int tileCount = 8;
        int tileAxis  = 1;
        TensorF* tensorSrc1 = oclTestAll->GenerateTensor(3,{5,1,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->Tile(PLATFORMS::CPU,scheduler,tensorSrc1,tileAxis,tileCount);
        TensorF* tensorGpu = oclTestAll->platformSelector->Tile(PLATFORMS::GPU_OCL,scheduler,tensorSrc1,tileAxis,tileCount);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Reduce_Max){

    TEST (Rank4_OverAxis2){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(0,{5,2,50,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->ReduceMax(PLATFORMS::CPU,scheduler,tensorSrc,2);
        TensorF* tensorGpu = oclTestAll->platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,tensorSrc,2);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST (Rank4_OverAxis1){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(0,{5,5,1,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->ReduceMax(PLATFORMS::CPU,scheduler,tensorSrc,1);
        TensorF* tensorGpu = oclTestAll->platformSelector->ReduceMax(PLATFORMS::GPU_OCL,scheduler,tensorSrc,1);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_ReduceSum_3D){

    TEST(Rank3_OverAxis0){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(1,{50,25,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc,true,false,false);
        TensorF* tensorGpu = oclTestAll->platformSelector->ReduceSum(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,false,false);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Rank3_OverAxis1){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(1,{50,25,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc,false,true,false);
        TensorF* tensorGpu = oclTestAll->platformSelector->ReduceSum(PLATFORMS::GPU_OCL,scheduler,tensorSrc,false,true,false);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }

    TEST(Rank3_OverAxis2){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(1,{50,25,50});
        TensorF* tensorCpu = oclTestAll->platformSelector->ReduceSum(PLATFORMS::CPU,scheduler,tensorSrc,false,false,true);
        TensorF* tensorGpu = oclTestAll->platformSelector->ReduceSum(PLATFORMS::GPU_OCL,scheduler,tensorSrc,false,false,true);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_ReduceSum_4D){

    TEST(Rank4_TTTF){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(1,{5,1024,20,256});
        TensorF* tensorCpu = oclTestAll->platformSelector->ReduceSum4D(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = oclTestAll->platformSelector->ReduceSum4D(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Mean){

    TEST(Rank4_TTTF){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(1,{5,1024,20,256});
        TensorF* tensorCpu = oclTestAll->platformSelector->Mean(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = oclTestAll->platformSelector->Mean(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Variance){

    TEST(Rank4_TTTF){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(1,{5,1024,20,256});
        TensorF* tensorCpu = oclTestAll->platformSelector->Variance(PLATFORMS::CPU,scheduler,tensorSrc,true,true,true,false);
        TensorF* tensorGpu = oclTestAll->platformSelector->Variance(PLATFORMS::GPU_OCL,scheduler,tensorSrc,true,true,true,false);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Conv2D_Mlp){

    TEST(Test_1){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(3,{5,3,3,6});
        TensorF* tensorWeight = oclTestAll->GenerateTensor(3,{1,1,6,7});
        TensorF* tensorBiases = oclTestAll->GenerateTensor(-1,{7}); //THE SHAPE SHOULD BE 1D, NOT 4D LIKE {1,1,1,7}
        TensorF* tensorCpu = oclTestAll->platformSelector->Conv2D(PLATFORMS::CPU,scheduler,tensorSrc,tensorWeight,tensorBiases);
        TensorF* tensorGpu = oclTestAll->platformSelector->Conv2D(PLATFORMS::GPU_OCL,scheduler,tensorSrc,tensorWeight,tensorBiases);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_Gather){

    TEST(Test_1){
        TensorF* tensorSrc = oclTestAll->GenerateTensor(3,{5,1024,6});
        TensorI* tensorIndices = oclTestAll->GenerateTensor(0,1023,{5,1024,20});
        TensorF* tensorCpu = oclTestAll->platformSelector->Gather(PLATFORMS::CPU,scheduler,tensorSrc,tensorIndices,1);
        TensorF* tensorGpu = oclTestAll->platformSelector->Gather(PLATFORMS::GPU_OCL,scheduler,tensorSrc,tensorIndices,1);
        bool comparisonResult = oclTestAll->platformSelector->CompareTensors(PLATFORMS::CPU,scheduler,tensorCpu,tensorGpu);
                CHECK_EQUAL(comparisonResult, true);
    }
}

SUITE(Kernel_TopK){

    TEST(Test_1) {
        TensorF *tensorSrc = oclTestAll->GenerateTensor(0, {5, 1024, 1024});
        TensorI *tensorCpu = oclTestAll->platformSelector->TopK(PLATFORMS::CPU, scheduler, tensorSrc, 2, 20);
        TensorI *tensorGpu = oclTestAll->platformSelector->TopK(PLATFORMS::GPU_OCL, scheduler, tensorSrc, 2, 20);
        bool comparisonResult = true;
        if (tensorCpu->getShape() != tensorGpu->getShape()){
        //    comparisonResult = false;
        //}
        //else{
            unsigned long len = tensorCpu->getLength();
            TensorI *tensorGpuTransfered = oclTestAll->platformSelector->CrossThePlatform(tensorGpu,PLATFORMS::CPU);
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

OclTestAll::OclTestAll(int dataset_offset, int batchsize, int pointcount, int knn_k) {
    platformSelector = new PlatformSelector(PLATFORMS::CPU,{PLATFORMS::CPU,PLATFORMS::GPU_OCL});
    DB_OFFSET = dataset_offset;
    B = (unsigned int)batchsize;
    N = (unsigned int)pointcount;
    K = (unsigned int)knn_k;
}

void OclTestAll::SetModelInput_data(string npy_pcl) {
    _npy_pcl = cnpy::npy_load(npy_pcl);
    input_pcl_BxNxD = new TensorF({B,N,3}, _npy_pcl.data<float>());
    input_pcl_BxNxD += DB_OFFSET*N*3;
    assert( N == (int)(_npy_pcl.shape[1]) );
}

void OclTestAll::SetModelInput_labels(string npy_labels) {
    // dataType of npy file should be int32, NOT uchar8!
    // use dataset_B5_labels_int32.npy
    _npy_labels = cnpy::npy_load(npy_labels);
    input_labels_B = new TensorI({B},_npy_labels.data<int>());
    input_labels_B->_buff += DB_OFFSET;
}

TensorI* OclTestAll::GetLabels(){
    return input_labels_B;
}

TensorF* OclTestAll::GenerateTensor(int pattern, vector<unsigned int> shape){
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

TensorI* OclTestAll::GenerateTensor(int intMin, int intMax, vector<unsigned int> shape){
    TensorI *testTn = new TensorI(shape);
    unsigned long _len = testTn->getLength();

    for (unsigned long i = 0; i < _len; i++) {
        testTn->_buff[i] = (int)float_rand((float)intMin,(float)intMax);
    }

    return testTn;
}

float OclTestAll::float_rand( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}