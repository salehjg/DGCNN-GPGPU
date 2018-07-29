#include <iostream>
#include "../../submodules/cnpy/cnpy.h"
#include "model01.h"
#include <thread>

using namespace std;

template<typename T> void printv(vector<T> vec)
{
    for (typename std::vector<T>::const_iterator i = vec.begin(); i != vec.end(); ++i)
    {
        std::cout << *i << ' ';
    }
    std::cout<<endl;
}

void task1(int batchsize)
{
    cout << "running thread on batchsize of : " << batchsize;

    ModelArch model(0,batchsize,1024,20);
    model.SetModelInput_data("/home/saleh/00_repos/tensorflow_repo/"
                             "00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/"
                             "data/dataset/"
                             //"dataset_B5_pcl.npy");
                             "pcl_100.npy");

    model.SetModelInput_labels("/home/saleh/00_repos/tensorflow_repo/"
                               "00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/"
                               "data/dataset/"
                               //"dataset_B5_labels.npy");
                               "lbl_100.npy");

    model.LoadWeights("/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                      "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights/",

                      "/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                      "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights/filelist.txt");
    double timerStart = seconds();
    model.execute();
    cout<< "Total model execution time with "<< batchsize <<" as batchsize: " << seconds() -timerStart<<" S"<<endl;

}

int main() {
    cout << "DeepPointV1"<<endl;
    cout << "CPU Version"<<endl;
    cout << "Pure C++, Without Linear Algebra Libraries"<<endl;

    // Constructs the new thread and runs it. Does not block execution.
    thread t1(task1, 60);
    thread t2(task1, 80);
    thread t3(task1, 100);

    // Makes the main thread wait for the new thread to finish execution, therefore blocks its own execution.
    t1.join();
    t2.join();
    t3.join();

}