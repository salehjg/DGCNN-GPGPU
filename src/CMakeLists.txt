subdirs(cpu_imp)

subdirs(cuda_imp/kernel_tests)
SET(CUDA_INCLUDE_DIRS /opt/cuda/include)
SET(CUDA_LIBRARIES cudart cublas)
SET(CUDA_LIBRARY_DIRS /opt/cuda/lib64)

include_directories(${PROJECT_SOURCE_DIR}/inc ${PROJECT_SOURCE_DIR}/inc/cpu_imp ${PROJECT_SOURCE_DIR}/inc/cuda_imp ${CUDA_INCLUDE_DIRS})

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
        PlatformSelector.cpp
        TensorF.cpp
        TensorI.cpp
        cuda_imp/CudaTensorF.cpp
        cuda_imp/CudaTensorI.cpp
        WorkScheduler.cpp
        WeightsLoader.cpp
        ModelArchTop.cpp)

link_directories(${PROJECT_SOURCE_DIR}/submodules/cnpy ${CUDA_LIBRARY_DIRS})

add_executable(TensorFTest ${SOURCE_FILES_TensorFTest})
add_executable(ClassifierMultiplatform ${SOURCE_FILES_ClassifierMultiplatform})


target_link_libraries(TensorFTest cnpy z pthread ${CUDA_LIBRARIES})
target_link_libraries(ClassifierMultiplatform cnpy z pthread ${CUDA_LIBRARIES})