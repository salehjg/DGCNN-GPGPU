subdirs(cpu_imp)

option(USE_CUDA "CUDA Switch" ON)
option(USE_OCL "OCL  Switch" ON)
set(USE_CUDA ON)
set(USE_OCL ON)
set(REPO_DIR_CMAKE  ${PROJECT_SOURCE_DIR})
set(BIN_DIR_CMAKE   ${CMAKE_BINARY_DIR})
configure_file(../inc/build_config.h.in ${CMAKE_BINARY_DIR}/build_config.h)

if (USE_CUDA)
    subdirs(cuda_imp/kernel_tests)
    SET(CUDA_INCLUDE_DIRS /opt/cuda/include)
    SET(CUDA_LIBRARIES cudart cublas)
    SET(CUDA_LIBRARY_DIRS /opt/cuda/lib64)

    find_package(CUDA REQUIRED)

    SET(CUDA_HOST_COMPILER /usr/bin/gcc-5)
    set(CUDA_PROPAGATE_HOST_FLAGS False)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options '-fPIC'")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")

    set(SOURCE_FILES_CUDA
            cuda_imp/CudaImplementation.cpp
            cuda_imp/CudaTensorF.cpp
            cuda_imp/CudaTensorI.cpp
            )

    # Find all cuda kernels
    #(https://stackoverflow.com/questions/29308787/cmake-file-for-cpp-cu-files)
    file( GLOB  KERNEL_SOURCES  ${PROJECT_SOURCE_DIR}/kernels/cuda/*.cu)
    message("Found these kernels: ")
    message("${KERNEL_SOURCES}")
endif (USE_CUDA)

if(USE_OCL)
    set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
    set(OPENCL_ROOT /opt/cuda/)
    find_package(OpenCL REQUIRED)
    message(${OPENCL_INCLUDE_DIRS})
    message(${OPENCL_LIBRARY})


    set(SOURCE_FILES_OCL
            ocl_imp/OclImplementation.cpp
            ocl_imp/OclTensorF.cpp
            ocl_imp/OclTensorI.cpp
            )
endif (USE_OCL)

set(SOURCE_FILES_CPU
        PlatformImplementation.cpp
        cpu_imp/CpuImplementation.cpp
        PlatformSelector.cpp
        TensorF.cpp
        TensorI.cpp
        WorkScheduler.cpp
        WeightsLoader.cpp
        ModelArchTop05.cpp
        )

set(SOURCE_FILES_ALL
        ${SOURCE_FILES_CPU}
        ${SOURCE_FILES_CUDA}
        ${SOURCE_FILES_OCL})

include_directories(
        ${CMAKE_BINARY_DIR}
        ${PROJECT_SOURCE_DIR}/inc
        ${PROJECT_SOURCE_DIR}/inc/cpu_imp
        ${PROJECT_SOURCE_DIR}/inc/cuda_imp
        ${CUDA_INCLUDE_DIRS}
        ${OPENCL_INCLUDE_DIRS}
)

link_directories(
        ${PROJECT_SOURCE_DIR}/submodules/cnpy
        ${CUDA_LIBRARY_DIRS}
        ${OPENCL_LIBRARY})

if(USE_CUDA AND USE_OCL)
    cuda_add_executable(ClassifierMultiplatform
            ClassifierMultiPlatform.cpp
            ${SOURCE_FILES_ALL}
            ${KERNEL_SOURCES})
    target_link_libraries(ClassifierMultiplatform cnpy z pthread ${CUDA_LIBRARIES} ${KERNEL_OBJs} ${OPENCL_LIBRARY})

    find_package(UnitTest++ REQUIRED)
    cuda_add_executable(oclTestAll
            ocl_imp/oclTestAll.cpp
            ${SOURCE_FILES_CPU}
            ${SOURCE_FILES_CUDA}
            ${SOURCE_FILES_OCL}
            ${KERNEL_SOURCES}
            )
    target_link_libraries(oclTestAll cnpy z pthread ${CUDA_LIBRARIES} ${KERNEL_OBJs} ${OPENCL_LIBRARY} UnitTest++)

endif(USE_CUDA AND USE_OCL)

if(USE_CUDA AND NOT(USE_OCL))
    cuda_add_executable(ClassifierMultiplatform
            ClassifierMultiPlatform.cpp
            ${SOURCE_FILES_ALL}
            ${KERNEL_SOURCES})
    target_link_libraries(ClassifierMultiplatform cnpy z pthread ${CUDA_LIBRARIES} ${KERNEL_OBJs} )
endif()

if(NOT(USE_CUDA) AND USE_OCL)
    add_executable(ClassifierMultiplatform
            ClassifierMultiPlatform.cpp
            ${SOURCE_FILES_ALL} )
    target_link_libraries(ClassifierMultiplatform cnpy z pthread ${OPENCL_LIBRARY})
endif()

if(NOT(USE_CUDA) AND NOT(USE_OCL))
    add_executable(ClassifierMultiplatform
            ClassifierMultiPlatform.cpp
            ${SOURCE_FILES_ALL} )
    target_link_libraries(ClassifierMultiplatform cnpy z pthread )
endif()