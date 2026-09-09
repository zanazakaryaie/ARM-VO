cmake_minimum_required(VERSION 3.20)

project(ARM_VO VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>")

option(BUILD_TESTS "Build the tests" OFF)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(BUILD_TOOLS "Build the tools library" ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CMakeDependentOption)

cmake_dependent_option(BUILD_CLI "Build the command line tools" ON "BUILD_TOOLS" OFF)

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(lib)
if(BUILD_TOOLS)
    add_subdirectory(tools)
endif()
if(BUILD_CLI)
    add_subdirectory(cli)
endif()

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
