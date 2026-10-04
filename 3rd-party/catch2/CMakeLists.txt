find_package(Catch2 2.13.10 EXACT QUIET CONFIG)
set(catch2_static FALSE)
if(TARGET Catch2::Catch2WithMain)
    get_target_property(catch2_type Catch2::Catch2WithMain TYPE)
    if(catch2_type STREQUAL "STATIC_LIBRARY")
        set(catch2_static TRUE)
    endif()
endif()
if(Catch2_FOUND AND catch2_static AND EXISTS "${Catch2_DIR}/Catch.cmake")
    set_property(TARGET Catch2::Catch2 PROPERTY IMPORTED_GLOBAL TRUE)
    set_property(TARGET Catch2::Catch2WithMain PROPERTY IMPORTED_GLOBAL TRUE)
    message(STATUS "Using installed Catch2 ${Catch2_VERSION}")
    if(ARMVO_INNER_BUILD)
        include("${Catch2_DIR}/Catch.cmake")
    else()
        list(APPEND ARMVO_PACKAGE_ARGS "-DCatch2_DIR:PATH=${Catch2_DIR}")
        set(ARMVO_PACKAGE_ARGS "${ARMVO_PACKAGE_ARGS}" PARENT_SCOPE)
    endif()
    return()
endif()
if(ARMVO_INNER_BUILD)
    message(FATAL_ERROR "The prepared Catch2 2.13.10 installation lacks Catch2WithMain.")
endif()

set(options ${ARMVO_EXTERNAL_CMAKE_ARGS}
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DBUILD_TESTING:BOOL=OFF
    -DCATCH_BUILD_STATIC_LIBRARY:BOOL=ON
    -DCATCH_BUILD_TESTING:BOOL=OFF
    -DCATCH_BUILD_EXAMPLES:BOOL=OFF
    -DCATCH_BUILD_EXTRA_TESTS:BOOL=OFF
    -DCATCH_INSTALL_DOCS:BOOL=OFF
    -DCATCH_INSTALL_HELPERS:BOOL=ON
    ${ARMVO_DEPENDENCY_INSTALL_ARGS}
)

set(version v2.13.10)
set(repository https://github.com/catchorg/Catch2.git)
armvo_dependency_cache(catch2 "${version}" "${repository}" "${options}"
    "" root)
set(prefix "${root}/install${CMAKE_INSTALL_PREFIX}")
set(ready "${root}/ready")

set(required_files "${ready}" "${root}/build/cmake_install.cmake"
    "${prefix}/lib/cmake/Catch2/Catch2Config.cmake"
    "${prefix}/lib/libCatch2WithMain.a"
    "${prefix}/lib/cmake/Catch2/Catch.cmake")
set(cached TRUE)
foreach(path IN LISTS required_files)
    if(NOT EXISTS "${path}")
        set(cached FALSE)
    endif()
endforeach()

if(cached)
    message(STATUS "Using cached catch2 ${version}: ${prefix}")
else()
    message(STATUS "Will build catch2 ${version} in ${root}")
    # Retry an incomplete cache; the dependency's own make reuses object files.
    file(REMOVE "${root}/stamps/armvo_dep_catch2-build"
                "${root}/stamps/armvo_dep_catch2-install")
    # Filter after computing the cache key to preserve existing cached builds.
    # This compatibility setting only exists in CMake 4 and newer.
    if(CMAKE_VERSION VERSION_LESS 4.0)
        list(FILTER options EXCLUDE REGEX "^-DCMAKE_POLICY_VERSION_MINIMUM:")
    endif()
    ExternalProject_Add(armvo_dep_catch2
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
    list(APPEND ARMVO_DEPENDENCY_TARGETS armvo_dep_catch2)
endif()

# Also install cached dependencies when the user runs make install.
list(APPEND ARMVO_DEPENDENCY_BUILDS "${root}/build")
list(APPEND ARMVO_PACKAGE_ARGS
    "-DCatch2_DIR:PATH=${prefix}/lib/cmake/Catch2")
set(ARMVO_DEPENDENCY_TARGETS "${ARMVO_DEPENDENCY_TARGETS}" PARENT_SCOPE)
set(ARMVO_DEPENDENCY_BUILDS "${ARMVO_DEPENDENCY_BUILDS}" PARENT_SCOPE)
set(ARMVO_PACKAGE_ARGS "${ARMVO_PACKAGE_ARGS}" PARENT_SCOPE)
