cmake_minimum_required(VERSION 3.15)
project(mobilegan_cli)

set(CMAKE_CXX_STANDARD 14)

add_library(opencl_wrapper_lib third_party/src/opencl_wrapper.cc)
target_include_directories(opencl_wrapper_lib PUBLIC third_party/inc)

add_executable(mobilegan_cli
        main.cpp
        seq/tensor.hpp
        seq/layers.hpp seq/layers.cpp
        seq/app.cpp
        parallel/layers.hpp parallel/layers.cpp
        parallel/runtime.hpp parallel/runtime.cpp
        parallel/app.cpp)
target_include_directories(mobilegan_cli PRIVATE .)
target_link_libraries(mobilegan_cli opencl_wrapper_lib)


add_executable(mobilegan_seq
        seq_main.cpp
        seq/tensor.hpp
        seq/layers.hpp seq/layers.cpp
        seq/app.cpp
        )
target_include_directories(mobilegan_seq PRIVATE .)
