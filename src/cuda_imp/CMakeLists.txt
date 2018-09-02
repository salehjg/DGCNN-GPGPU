subdirs(kernel_tests)

SET(CUDA_INCLUDE_DIRS /opt/cuda/include)
SET(CUDA_LIBRARIES cudart cublas)
SET(CUDA_LIBRARY_DIRS /opt/cuda/lib64)

include_directories(${PROJECT_SOURCE_DIR}/inc/cuda_imp ${CUDA_INCLUDE_DIRS})
set(SOURCE_FILES cublas_tests/cublas_test.cpp)


link_directories(${PROJECT_SOURCE_DIR}/submodules/cnpy ${CUDA_LIBRARY_DIRS})

add_executable(DeepPointV1_cuda ${SOURCE_FILES})
#add_executable(DeepPointV1_batch_ex batched_sgemm_ex.cpp)
#add_executable(DeepPointV1_batch_ex2 batched_sgemm_ex2.cpp)
add_executable(DeepPointV1_LA_CUDA_test cublas_tests/LA_CUDA_Test.cpp cublas_tests/LA_CUDA.cpp cublas_tests/Tensor.cpp cublas_tests/CudaMemHelper.cpp)

target_link_libraries(DeepPointV1_cuda cnpy z pthread ${CUDA_LIBRARIES})
#target_link_libraries(DeepPointV1_batch_ex cnpy z pthread ${CUDA_LIBRARIES})
#target_link_libraries(DeepPointV1_batch_ex2 cnpy z pthread ${CUDA_LIBRARIES})
target_link_libraries(DeepPointV1_LA_CUDA_test cnpy z pthread ${CUDA_LIBRARIES})
