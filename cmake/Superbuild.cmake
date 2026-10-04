# Dependencies are configured and built during make.
# Configure ARM-VO only afterward, so its find_package calls can load the real exported targets.


# This compatibility override is for older third-party projects, not ARM-VO.
list(FILTER ARMVO_EXTERNAL_CMAKE_ARGS EXCLUDE REGEX "^-DCMAKE_POLICY_VERSION_MINIMUM:")
set(inner_build "${PROJECT_BINARY_DIR}/_armvo")

configure_file("${PROJECT_SOURCE_DIR}/cmake/InstallDependencies.cmake.in"
    "${PROJECT_BINARY_DIR}/3rd-party/install-dependencies.cmake" @ONLY)

set(arguments ${ARMVO_PACKAGE_ARGS}
    -DARMVO_INNER_BUILD:BOOL=ON
    "-DARMVO_BUILD_ROOT:PATH=${PROJECT_BINARY_DIR}"
    "-DARMVO_DEPENDENCY_INSTALL_SCRIPT:FILEPATH=${PROJECT_BINARY_DIR}/3rd-party/install-dependencies.cmake"
)

foreach(variable
        BUILD_TESTS BUILD_TOOLS BUILD_CLI BUILD_PYTHON_BINDINGS
        CMAKE_INSTALL_PREFIX CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_BINDIR
        CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_DATAROOTDIR
        CMAKE_EXPORT_COMPILE_COMMANDS CMAKE_INSTALL_RPATH
        TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY TensorRT_NVINFER_PLUGIN_LIBRARY
        TensorRT_TRTEXEC_EXECUTABLE CUDAToolkit_ROOT
        Python3_EXECUTABLE Python3_ROOT_DIR pybind11_DIR)
    if(DEFINED ${variable})
        string(REPLACE ";" "|" value "${${variable}}")
        list(APPEND arguments "-D${variable}:STRING=${value}")
    endif()
endforeach()

ExternalProject_Add(armvo_build
    SOURCE_DIR "${PROJECT_SOURCE_DIR}"
    BINARY_DIR "${inner_build}"
    PREFIX "${PROJECT_BINARY_DIR}/_armvo-driver"
    DOWNLOAD_COMMAND ""
    UPDATE_COMMAND ""
    LIST_SEPARATOR "|"
    # Preserve inherited compiler/toolchain settings, including flags for other
    # configurations, without treating them as manually supplied -D options.
    CMAKE_CACHE_ARGS ${ARMVO_EXTERNAL_CMAKE_ARGS}
    CMAKE_ARGS ${arguments}
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ""
    DEPENDS ${ARMVO_DEPENDENCY_TARGETS}
)

configure_file("${PROJECT_SOURCE_DIR}/cmake/InstallArmvo.cmake.in"
    "${PROJECT_BINARY_DIR}/_armvo-driver/install-armvo.cmake" @ONLY)

install(SCRIPT "${PROJECT_BINARY_DIR}/_armvo-driver/install-armvo.cmake")

if(BUILD_TESTS)
    add_test(NAME armvo_tests
        COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${inner_build}" --output-on-failure)
endif()
