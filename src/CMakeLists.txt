subdirs(cpu_imp)

subdirs(cuda_imp/kernel_tests)
SET(CUDA_INCLUDE_DIRS /opt/cuda/include)
SET(CUDA_LIBRARIES cudart cublas)
SET(CUDA_LIBRARY_DIRS /opt/cuda/lib64)

find_package(CUDA REQUIRED)
find_package(OpenCL REQUIRED)

SET(CUDA_HOST_COMPILER /usr/bin/gcc-5)
set(CUDA_PROPAGATE_HOST_FLAGS False)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options '-fPIC'")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

include_directories(${PROJECT_SOURCE_DIR}/inc ${PROJECT_SOURCE_DIR}/inc/cpu_imp ${PROJECT_SOURCE_DIR}/inc/cuda_imp ${CUDA_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})

set(SOURCE_FILES_TensorFTest
        #ModelArch01.cpp
        #PlatformImplementation.cpp
        #PlatformSelector.cpp
        TensorF.cpp
        #cpu_imp/CpuImplementation.cpp
        #cuda_imp/CudaImplementation.cpp
        cuda_imp/CudaTensorF.cpp
        tensorF_test.cpp)

set(SOURCE_FILES_ClassifierMultiplatform
        ClassifierMultiPlatform.cpp
        PlatformImplementation.cpp
        cpu_imp/CpuImplementation.cpp
        cuda_imp/CudaImplementation.cpp
        ocl_imp/OclImplementation.cpp
        PlatformSelector.cpp
        TensorF.cpp
        TensorI.cpp
        cuda_imp/CudaTensorF.cpp
        cuda_imp/CudaTensorI.cpp
        ocl_imp/OclTensorF.cpp
        ocl_imp/OclTensorI.cpp
        WorkScheduler.cpp
        WeightsLoader.cpp
        ModelArchTop01.cpp
        ModelArchTop02.cpp
        ModelArchTop03.cpp
        ModelArchTop04.cpp
        ModelArchTop05.cpp
        )

# Find all cuda kernels
#(https://stackoverflow.com/questions/29308787/cmake-file-for-cpp-cu-files)
file( GLOB  KERNEL_SOURCES  ${PROJECT_SOURCE_DIR}/kernels/cuda/*.cu)
message("Found these kernels: ")
message("${KERNEL_SOURCES}")

link_directories(${PROJECT_SOURCE_DIR}/submodules/cnpy ${CUDA_LIBRARY_DIRS} ${OpenCL_LIBRARY})


cuda_add_executable(ClassifierMultiplatform  ${SOURCE_FILES_ClassifierMultiplatform} ${KERNEL_SOURCES})
target_link_libraries(ClassifierMultiplatform cnpy z pthread ${CUDA_LIBRARIES} cudnn ${KERNEL_OBJs} ${OpenCL_LIBRARY})

add_executable(TensorFTest ${SOURCE_FILES_TensorFTest})
target_link_libraries(TensorFTest cnpy z pthread ${CUDA_LIBRARIES})


