cmake_minimum_required(VERSION 3.0.2)
project(ARM_VO)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  cv_bridge
  image_transport 
  roscpp
)

find_package(OpenCV)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -mfpu=neon -mfloat-abi=hard -funroll-loops -Ofast")

include_directories(${catkin_INCLUDE_DIRS} "include" "../../include")


file(GLOB SOURCES "src/ARM_VO_Node.cpp" "src/main.cpp" "../../src/*.cpp")
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../../src/main.cpp)
add_executable(ARM_VO ${SOURCES})
target_link_libraries(ARM_VO ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})


add_executable(ImagePublisher src/image_publisher.cpp)
target_link_libraries(ImagePublisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
