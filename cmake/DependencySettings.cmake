include(ExternalProject)

# Arguments shared by the independent dependency builds and the ARM-VO build.
set(ARMVO_EXTERNAL_CMAKE_ARGS
    -DCMAKE_CXX_STANDARD:STRING=17
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=ON
    -DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5
)
set(_armvo_forward_variables
    CMAKE_BUILD_TYPE CMAKE_C_COMPILER CMAKE_CXX_COMPILER
    CMAKE_CXX_EXTENSIONS CMAKE_TOOLCHAIN_FILE CMAKE_SYSROOT
    CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER_TARGET
    CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER
    CMAKE_PREFIX_PATH CMAKE_FIND_ROOT_PATH
    CMAKE_FIND_ROOT_PATH_MODE_PACKAGE CMAKE_FIND_ROOT_PATH_MODE_LIBRARY
    CMAKE_FIND_ROOT_PATH_MODE_INCLUDE CMAKE_FIND_ROOT_PATH_MODE_PROGRAM
)
foreach(language C CXX EXE_LINKER SHARED_LINKER MODULE_LINKER)
    foreach(suffix "" _DEBUG _RELEASE _RELWITHDEBINFO _MINSIZEREL)
        list(APPEND _armvo_forward_variables "CMAKE_${language}_FLAGS${suffix}")
    endforeach()
endforeach()
foreach(variable IN LISTS _armvo_forward_variables)
    if(DEFINED ${variable})
        # ExternalProject restores list values when constructing its commands.
        string(REPLACE ";" "|" value "${${variable}}")
        list(APPEND ARMVO_EXTERNAL_CMAKE_ARGS "-D${variable}:STRING=${value}")
    endif()
endforeach()

# Every dependency uses the same installation layout inside its private stage.
set(ARMVO_DEPENDENCY_INSTALL_ARGS
    "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}"
    -DCMAKE_INSTALL_LIBDIR:PATH=lib
    -DCMAKE_INSTALL_BINDIR:PATH=bin
    -DCMAKE_INSTALL_INCLUDEDIR:PATH=include
    -DCMAKE_INSTALL_DATAROOTDIR:PATH=share
)

# Bump only when the cache layout or build/install procedure changes. Formatting
# edits to this helper should not invalidate expensive dependency builds.
set(ARMVO_DEPENDENCY_CACHE_VERSION 2)

# This helper only calculates a directory. Each dependency declares its build,
# checks its cached artifacts, and reports its package/install paths explicitly.
function(armvo_dependency_cache name version repository options setup_script output_dir)
    set(toolchain_hash "")
    if(CMAKE_TOOLCHAIN_FILE AND EXISTS "${CMAKE_TOOLCHAIN_FILE}")
        file(SHA256 "${CMAKE_TOOLCHAIN_FILE}" toolchain_hash)
    endif()
    set(setup_hash "")
    if(setup_script)
        file(SHA256 "${setup_script}" setup_hash)
    endif()
    string(SHA256 key
        "${ARMVO_DEPENDENCY_CACHE_VERSION};${repository};${version};${options};${CMAKE_GENERATOR};${CMAKE_VERSION};${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR};${CMAKE_SIZEOF_VOID_P};${CMAKE_C_COMPILER_ID};${CMAKE_C_COMPILER_VERSION};${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${toolchain_hash};${setup_hash}")
    string(SUBSTRING "${key}" 0 16 key)
    set(${output_dir} "${ARMVO_DEPENDENCY_CACHE}/${name}/${version}-${key}" PARENT_SCOPE)
endfunction()
