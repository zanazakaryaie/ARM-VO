cmake_minimum_required(VERSION 2.8)

project(ARM_VO)

SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++11 -mfpu=neon")

find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found.")
endif()

find_package(OpenMP QUIET)
if(OpenMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else()
    message(WARNING "OpenMP not found.")
endif()

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
add_executable(ARM_VO ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
