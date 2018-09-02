#subdirs(cpu_imp
#        cuda_imp)

subdirs(cuda_imp/kernel_tests)
SET(CUDA_INCLUDE_DIRS /opt/cuda/include)
SET(CUDA_LIBRARIES cudart cublas)
SET(CUDA_LIBRARY_DIRS /opt/cuda/lib64)

include_directories(${PROJECT_SOURCE_DIR}/inc/cpu_imp ${PROJECT_SOURCE_DIR}/inc/cuda_imp ${CUDA_INCLUDE_DIRS})

set(SOURCE_FILES
        #ModelArch01.cpp
        #PlatformImplementation.cpp
        #PlatformSelector.cpp
        TensorF.cpp
        #cpu_imp/CpuImplementation.cpp
        #cuda_imp/CudaImplementation.cpp
        cuda_imp/CudaTensorF.cpp
        tensorF_test.cpp)

link_directories(${PROJECT_SOURCE_DIR}/submodules/cnpy ${CUDA_LIBRARY_DIRS})
add_executable(DeepPointV1_multiPlatform ${SOURCE_FILES})
target_link_libraries(DeepPointV1_multiPlatform cnpy z pthread ${CUDA_LIBRARIES})