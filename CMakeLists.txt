cmake_minimum_required(VERSION 3.20)
project(alice_jsp LANGUAGES CXX)

add_library(alice_jsp
    src/dataset.cpp
    src/dataset_reader.cpp
    src/problem_instance.cpp
    src/yaml_loader.cpp
)

target_include_directories(alice_jsp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(alice_jsp PUBLIC cxx_std_17)

add_executable(alice_jsp_inspect examples/inspect_dataset.cpp)

target_link_libraries(alice_jsp_inspect PRIVATE alice_jsp)



