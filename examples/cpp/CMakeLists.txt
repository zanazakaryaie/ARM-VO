cmake_minimum_required(VERSION 3.20)
project(my_app LANGUAGES CXX)

find_package(armvo REQUIRED CONFIG)
find_package(OpenCV REQUIRED COMPONENTS imgcodecs)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE armvo::ArmVO opencv_imgcodecs)
