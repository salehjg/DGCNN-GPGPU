include_directories(${PROJECT_SOURCE_DIR}/inc/cpu_imp)
 # I assume you want to use LibProject as a library in MainProject.
 # include_directories(${PROJECT_SOURCE_DIR}/submodules/inc)



set(SOURCE_FILES main.cpp model01.cpp)

link_directories(${PROJECT_SOURCE_DIR}/submodules/cnpy)
add_executable(DeepPointV1_cpu ${SOURCE_FILES})
target_link_libraries(DeepPointV1_cpu cnpy z pthread)