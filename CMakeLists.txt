cmake_minimum_required(VERSION 3.3)
project(main)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11")

set(SOURCE_FILES main.cpp mnist_loader.cpp)
include_directories(.)
add_executable(main ${SOURCE_FILES})
LINK_DIRECTORIES(
/usr/lib/x86_64-linux-gnu/
/usr/include/libpng16
)
TARGET_LINK_LIBRARIES(main glfw GLEW GLU GL pthread png16)

