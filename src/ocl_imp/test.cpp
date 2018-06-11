//
// Created by saleh on 5/31/18.
//

//disable debug mechanisms to have a fair comparison with ublas:
#ifndef NDEBUG
 #define NDEBUG
#endif

#ifndef VIENNACL_WITH_OPENCL
 #define VIENNACL_WITH_OPENCL
#endif

// System headers
#include <iostream>


// Must be set if you want to use ViennaCL algorithms on ublas objects
#define VIENNACL_WITH_UBLAS 0


// ViennaCL headers
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/tools/random.hpp"
#include "viennacl/tools/timer.hpp"
#include "model01.h"


/**
*  Later in this tutorial we will iterate over all available OpenCL devices.
*  To ensure that this tutorial also works if no OpenCL backend is activated, we need this dummy-struct.
**/
#ifndef VIENNACL_WITH_OPENCL
struct dummy
{
    std::size_t size() const { return 1; }
};
#endif

void test01()
{
    std::vector<float>      stl_vec(100000000);
    viennacl::vector<float> vcl_vec1(100000000);
    viennacl::vector<float> vcl_vec2(100000000);
    //fill the STL vector:
    for (size_t i=0; i<stl_vec.size(); ++i)
        stl_vec[i] = i;
    //copy content to GPU vector (recommended initialization)
    copy(stl_vec.begin(), stl_vec.end(), vcl_vec1.begin());

    std::cout<< "starting...\n";
    //manipulate GPU vector here
    for(int i=0;i<99;i++) {
        for(int r = 0 ; r< 10;r++) {
            vcl_vec1 *= 2.0;
            vcl_vec2 = vcl_vec1 - 3.0 * vcl_vec1;
            vcl_vec1 = vcl_vec2 - 3.0 * vcl_vec2;
        }
    }
    std::cout<< "Finished...\n";
    //copy content from GPU vector back to STL vector
    copy(vcl_vec1.begin(), vcl_vec1.end(), stl_vec.begin());


    std::cout << "!!!! COMPUTATION COMPLETED SUCCESSFULLY !!!!" << std::endl;
}

void test02()
{
    std::cout << "!!!! COMPUTATION IS STARTED !!!!" << std::endl;
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    std::cout << "!!!! COMPUTATION IS COMPLETED SUCCESSFULLY !!!!" << std::endl;
}

int main()
{
     test01();
     test02();


    return EXIT_SUCCESS;
}