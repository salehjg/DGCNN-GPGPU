cmake_minimum_required(VERSION 3.10)
SET(CUDA_INCLUDE_DIRS /opt/cuda/include)
SET(CUDA_LIBRARIES cudart cublas)
SET(CUDA_LIBRARY_DIRS /opt/cuda/lib64)
#SET(CUDA_TOOLKIT_ROOT_DIR /opt/cuda/)
find_package(CUDA REQUIRED)

# ----------------------------------------------------------------------------------------------------------------------
include_directories(${PROJECT_SOURCE_DIR}/inc/cuda_imp ${CUDA_INCLUDE_DIRS})
link_directories(${PROJECT_SOURCE_DIR}/submodules/cnpy ${CUDA_LIBRARY_DIRS})

# ----------------------------------------------------------------------------------------------------------------------
# Find all cuda kernels
#(https://stackoverflow.com/questions/29308787/cmake-file-for-cpp-cu-files)
file( GLOB  KERNEL_SOURCES  ${PROJECT_SOURCE_DIR}/kernels/cuda/*.cu)
message("Found these kernels: ")
message("${KERNEL_SOURCES}")

# ----------------------------------------------------------------------------------------------------------------------
# COMMENTED, BECAUSE THEY ARE ALREADY APPLIED ON MULTIPLATFORM CMAKELIST
#(https://stackoverflow.com/questions/14306642/adding-multiple-executables-in-cmake)
# Pass options to NVCC
#SET(CUDA_HOST_COMPILER /usr/bin/gcc-5)
#set(CUDA_PROPAGATE_HOST_FLAGS False)

#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options '-fPIC'")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std c++11")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

file( GLOB KERNEL_TEST_SOURCES ${PROJECT_SOURCE_DIR}/src/cuda_imp/kernel_tests/*.cpp )
foreach( testsourcefile ${KERNEL_TEST_SOURCES} )
    # I used a simple string replace, to cut off .cpp.
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    get_filename_component(barename ${testname} NAME)
    cuda_add_executable(${barename}
            ${testsourcefile}
            ${KERNEL_SOURCES}
            ${PROJECT_SOURCE_DIR}/src/TensorF.cpp
            ${PROJECT_SOURCE_DIR}/src/TensorI.cpp
            ${PROJECT_SOURCE_DIR}/src/cuda_imp/CudaTensorF.cpp
            ${PROJECT_SOURCE_DIR}/src/cuda_imp/CudaTensorI.cpp
            ${PROJECT_SOURCE_DIR}/src/cpu_imp/CpuImplementation.cpp
            ${PROJECT_SOURCE_DIR}/src/WorkScheduler.cpp
            )

    # Make sure YourLib is linked to each app
    target_link_libraries( ${barename} cnpy z pthread ${CUDA_LIBRARIES} ${KERNEL_OBJs})
endforeach( testsourcefile ${KERNEL_TEST_SOURCES})

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


















