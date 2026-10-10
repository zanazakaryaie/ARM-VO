cmake_minimum_required(VERSION 3.20)

project(ARM_VO VERSION 2.1.0 LANGUAGES C CXX)

if(NOT DEFINED ARMVO_BUILD_ROOT)
    set(ARMVO_BUILD_ROOT "${PROJECT_BINARY_DIR}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

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

configure_file(cmake/Uninstall.cmake.in
    "${PROJECT_BINARY_DIR}/armvo-uninstall.cmake" @ONLY)
add_custom_target(uninstall
    COMMAND "${CMAKE_COMMAND}" -DARMVO_UNINSTALL_DEPENDENCIES=OFF
        -P "${PROJECT_BINARY_DIR}/armvo-uninstall.cmake"
    COMMENT "Uninstalling ARM-VO (keeping dependencies)"
    VERBATIM
)
add_custom_target(uninstall-with-dependencies
    COMMAND "${CMAKE_COMMAND}" -DARMVO_UNINSTALL_DEPENDENCIES=ON
        -P "${PROJECT_BINARY_DIR}/armvo-uninstall.cmake"
    COMMENT "Uninstalling ARM-VO and dependencies installed by this build"
    VERBATIM
)

add_subdirectory(3rd-party)

if(NOT ARMVO_INNER_BUILD)
    include(Superbuild)
    return()
endif()

install(SCRIPT "${ARMVO_DEPENDENCY_INSTALL_SCRIPT}")

# Install rules run in declaration order, including subdirectories. Record
# third-party files before any ARM-VO install rules are added.
install(CODE [[
    set(_armvo_dependency_files "${CMAKE_INSTALL_MANIFEST_FILES}")
]])

add_compile_options("$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Werror>")

add_subdirectory(lib "${ARMVO_BUILD_ROOT}/lib")
if(BUILD_TOOLS)
    add_subdirectory(tools "${ARMVO_BUILD_ROOT}/tools")
endif()
if(BUILD_CLI)
    add_subdirectory(cli "${ARMVO_BUILD_ROOT}/cli")
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

configure_file(cmake/RecordInstall.cmake.in
    "${PROJECT_BINARY_DIR}/armvo-record-install.cmake" @ONLY)
install(SCRIPT "${PROJECT_BINARY_DIR}/armvo-record-install.cmake")
