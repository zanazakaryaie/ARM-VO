cmake_minimum_required(VERSION 3.20)
project(ARMVO_CI_Dependencies LANGUAGES C CXX)

get_filename_component(ARMVO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
list(PREPEND CMAKE_MODULE_PATH "${ARMVO_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

set(BUILD_TESTS ON)
set(BUILD_TOOLS ON)
set(INFERENCE_BACKEND NCNN)
add_subdirectory("${ARMVO_SOURCE_DIR}/3rd-party" 3rd-party)

foreach(dependency opencv ncnn catch2)
    if(NOT TARGET armvo_dep_${dependency})
        message(FATAL_ERROR "The CI image must build ${dependency} from source.")
    endif()
endforeach()

configure_file("${ARMVO_SOURCE_DIR}/cmake/InstallDependencies.cmake.in"
    "${CMAKE_BINARY_DIR}/install-dependencies.cmake" @ONLY)
install(SCRIPT "${CMAKE_BINARY_DIR}/install-dependencies.cmake")
