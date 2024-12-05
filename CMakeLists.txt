cmake_minimum_required(VERSION 3.10)
project(array_utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(array_utils INTERFACE)
target_include_directories(array_utils INTERFACE include)

enable_testing()
add_subdirectory(tests)