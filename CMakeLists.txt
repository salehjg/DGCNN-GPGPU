cmake_minimum_required(VERSION 3.8)
project(DeepPointV1)



set(CMAKE_BUILD_TYPE Debug)
#=====#=====#=====#=====#=====#=====#=====
macro(remove_cxx_flag flag)
    string(REPLACE "${flag}" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endmacro()
message(${CMAKE_CXX_FLAGS_DEBUG}) # print "-O3 -DNDEBUG"
remove_cxx_flag("-O3")
SET(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0" )
message(${CMAKE_CXX_FLAGS_DEBUG}) # print "-DNDEBUG"
#=====#=====#=====#=====#=====#=====#=====
set(CMAKE_CXX_STANDARD 11)



subdirs(src submodules)