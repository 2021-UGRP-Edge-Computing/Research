cmake_minimum_required (VERSION 2.8)

project(vision)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(../main.o main.cpp)

target_link_libraries(../main.o ${OpenCV_LIBS})
