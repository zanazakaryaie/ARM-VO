cmake_minimum_required(VERSION 3.20)

project(ARM_VO VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTS "Build the tests" OFF)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(BUILD_TESTS)
    enable_testing()
endif()

set(ARMVO_OPENCV_COMPONENTS core imgproc video highgui features2d calib3d imgcodecs)
find_package(OpenCV REQUIRED COMPONENTS ${ARMVO_OPENCV_COMPONENTS})

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(cli)

set(ARMVO_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/armvo")

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/armvoConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/armvoConfig.cmake"
    INSTALL_DESTINATION "${ARMVO_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/armvoConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(EXPORT armvoTargets
    NAMESPACE armvo::
    DESTINATION "${ARMVO_INSTALL_CMAKEDIR}"
)

install(FILES
    "${PROJECT_BINARY_DIR}/armvoConfig.cmake"
    "${PROJECT_BINARY_DIR}/armvoConfigVersion.cmake"
    DESTINATION "${ARMVO_INSTALL_CMAKEDIR}"
)
