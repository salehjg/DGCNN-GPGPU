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

void task1(string msg)
{
    cout << "task1 says: " << msg;

    ModelArch model(5,1024,20);
    model.SetModelInput_data("/home/saleh/00_repos/tensorflow_repo/"
                             "00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/"
                             "data/dataset/dataset_B5_pcl.npy");

    model.SetModelInput_labels("/home/saleh/00_repos/tensorflow_repo/"
                               "00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/"
                               "data/dataset/dataset_B5_labels.npy");

    model.LoadWeights("/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                      "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights/",

                      "/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                      "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights/filelist.txt");

    model.execute();
}

void task2(string msg)
{
    cout << "task1 says: " << msg;

    ModelArch model(5,1024,20);
    model.SetModelInput_data("/home/saleh/00_repos/tensorflow_repo/"
                             "00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/"
                             "data/dataset/pcl_100.npy");

    model.SetModelInput_labels("/home/saleh/00_repos/tensorflow_repo/"
                               "00_Projects/deeppoint_repo/DeepPoint-V1-GPGPU/"
                               "data/dataset/lbl_100.npy");

    model.LoadWeights("/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                      "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights/",

                      "/home/saleh/00_repos/tensorflow_repo/00_Projects/"
                      "deeppoint_repo/DeepPoint-V1-GPGPU/data/weights/filelist.txt");

    model.execute();
}

int main() {
    cout << "DeepPointV1"<<endl;
    cout << "CPU Version"<<endl;
    cout << "Pure C++, Without Linear Algebra Libraries"<<endl;

    // Constructs the new thread and runs it. Does not block execution.
    thread t1(task2, "Hello1");

    // Makes the main thread wait for the new thread to finish execution, therefore blocks its own execution.
    t1.join();

}