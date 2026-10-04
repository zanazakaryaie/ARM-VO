find_package(ncnn QUIET CONFIG)
if(ncnn_FOUND AND TARGET ncnn)
    set_property(TARGET ncnn PROPERTY IMPORTED_GLOBAL TRUE)
    message(STATUS "Using installed ncnn")
    if(NOT ARMVO_INNER_BUILD)
        list(APPEND ARMVO_PACKAGE_ARGS "-Dncnn_DIR:PATH=${ncnn_DIR}")
        set(ARMVO_PACKAGE_ARGS "${ARMVO_PACKAGE_ARGS}" PARENT_SCOPE)
    endif()
    return()
endif()
if(ARMVO_INNER_BUILD)
    message(FATAL_ERROR "The prepared ncnn installation is missing.")
endif()

set(ncnn_archive lib/libncnn.a)
string(TOUPPER "${CMAKE_BUILD_TYPE}" ncnn_build_type)
if(ncnn_build_type STREQUAL "DEBUG")
    set(ncnn_archive lib/libncnnd.a)
endif()
set(options ${ARMVO_EXTERNAL_CMAKE_ARGS}
    -DNCNN_VERSION:STRING=20241226
    -DNCNN_BUILD_TESTS:BOOL=OFF
    -DNCNN_BUILD_EXAMPLES:BOOL=OFF
    -DNCNN_BUILD_BENCHMARK:BOOL=OFF
    "-DNCNN_BUILD_TOOLS:BOOL=${BUILD_TOOLS}"
    -DNCNN_PYTHON:BOOL=OFF
    -DNCNN_THREADS:BOOL=ON
    -DNCNN_OPENMP:BOOL=OFF
    -DNCNN_VULKAN:BOOL=OFF
    -DNCNN_ENABLE_LTO:BOOL=ON
    -DNCNN_SHARED_LIB:BOOL=OFF
    -DNCNN_INSTALL_SDK:BOOL=ON
    -DNCNN_DISABLE_PIC:BOOL=OFF
    -DCMAKE_DISABLE_FIND_PACKAGE_OpenCV:BOOL=ON
    -DCMAKE_POLICY_DEFAULT_CMP0069:STRING=NEW
    "-DCMAKE_PROJECT_ncnn_INCLUDE:FILEPATH=${CMAKE_CURRENT_SOURCE_DIR}/ConfigureNcnn.cmake"
    ${ARMVO_DEPENDENCY_INSTALL_ARGS}
)

set(version 20241226)
set(repository https://github.com/Tencent/ncnn.git)
armvo_dependency_cache(ncnn "${version}" "${repository}" "${options}"
    "${CMAKE_CURRENT_SOURCE_DIR}/ConfigureNcnn.cmake" root)
set(prefix "${root}/install${CMAKE_INSTALL_PREFIX}")
set(ready "${root}/ready")

set(required_files "${ready}" "${root}/build/cmake_install.cmake"
    "${prefix}/lib/cmake/ncnn/ncnnConfig.cmake"
    "${prefix}/${ncnn_archive}")
set(cached TRUE)
foreach(path IN LISTS required_files)
    if(NOT EXISTS "${path}")
        set(cached FALSE)
    endif()
endforeach()

if(cached)
    message(STATUS "Using cached ncnn ${version}: ${prefix}")
else()
    message(STATUS "Will build ncnn ${version} in ${root}")
    # Retry an incomplete cache; the dependency's own make reuses object files.
    file(REMOVE "${root}/stamps/armvo_dep_ncnn-build"
                "${root}/stamps/armvo_dep_ncnn-install")
    # Filter after computing the cache key to preserve existing cached builds.
    # This compatibility setting only exists in CMake 4 and newer.
    if(CMAKE_VERSION VERSION_LESS 4.0)
        list(FILTER options EXCLUDE REGEX "^-DCMAKE_POLICY_VERSION_MINIMUM:")
    endif()
    ExternalProject_Add(armvo_dep_ncnn
        PREFIX "${root}"
        SOURCE_DIR "${root}/source"
        BINARY_DIR "${root}/build"
        STAMP_DIR "${root}/stamps"
        TMP_DIR "${root}/tmp"
        GIT_REPOSITORY "${repository}"
        GIT_TAG "${version}"
        GIT_SHALLOW TRUE
        GIT_SUBMODULES ""
        UPDATE_COMMAND ""
        LIST_SEPARATOR "|"
        # Preload settings; this project does not use every forwarded flag.
        CMAKE_CACHE_ARGS ${options}
        # make installs privately; make install later uses the system prefix.
        INSTALL_COMMAND "${CMAKE_COMMAND}" -E env "DESTDIR=${root}/install"
            "${CMAKE_COMMAND}" --install <BINARY_DIR>
        COMMAND "${CMAKE_COMMAND}" -E touch "${ready}"
    )
    list(APPEND ARMVO_DEPENDENCY_TARGETS armvo_dep_ncnn)
endif()

# Also install cached dependencies when the user runs make install.
list(APPEND ARMVO_DEPENDENCY_BUILDS "${root}/build")
list(APPEND ARMVO_PACKAGE_ARGS
    "-Dncnn_DIR:PATH=${prefix}/lib/cmake/ncnn")
set(ARMVO_DEPENDENCY_TARGETS "${ARMVO_DEPENDENCY_TARGETS}" PARENT_SCOPE)
set(ARMVO_DEPENDENCY_BUILDS "${ARMVO_DEPENDENCY_BUILDS}" PARENT_SCOPE)
set(ARMVO_PACKAGE_ARGS "${ARMVO_PACKAGE_ARGS}" PARENT_SCOPE)
