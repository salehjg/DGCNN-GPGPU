
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/submodules/viennacl-dev/cmake")
include(
        ${PROJECT_SOURCE_DIR}/cmake/FindOpenCL.cmake)

include_directories(
        ${PROJECT_SOURCE_DIR}/inc/ocl_imp
        ${PROJECT_SOURCE_DIR}/submodules/viennacl-dev
        ${OPENCL_INCLUDE_DIRS}
        )
# I assume you want to use LibProject as a library in MainProject.
# include_directories(${PROJECT_SOURCE_DIR}/submodules/inc)

link_directories(${PROJECT_SOURCE_DIR}/submodules/cnpy)
add_executable(DeepPointV1_ocl test.cpp model01.cpp)
target_link_libraries(DeepPointV1_ocl cnpy z ${OPENCL_LIBRARIES})