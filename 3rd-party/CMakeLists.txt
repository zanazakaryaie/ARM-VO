if(NOT INFERENCE_BACKEND MATCHES "^(AUTO|NCNN|TENSORRT)$")
    message(FATAL_ERROR
        "Invalid INFERENCE_BACKEND '${INFERENCE_BACKEND}'. Expected AUTO, NCNN, or TENSORRT.")
endif()

set(ARMVO_USE_TENSORRT OFF)
if(NOT INFERENCE_BACKEND STREQUAL "NCNN")
    find_package(TensorRT 8.6 QUIET)
    if(TensorRT_FOUND AND TensorRT_VERSION VERSION_LESS 9.0)
        set(ARMVO_USE_TENSORRT ON)
    elseif(INFERENCE_BACKEND STREQUAL "TENSORRT")
        message(FATAL_ERROR
            "INFERENCE_BACKEND=TENSORRT requires TensorRT 8.6.x, including "
            "nvinfer, nvinfer_plugin, trtexec, and the CUDA toolkit.")
    elseif(TensorRT_FOUND)
        message(STATUS
            "TensorRT ${TensorRT_VERSION} found, but this backend targets TensorRT 8.6.x. Falling back to ncnn.")
    else()
        message(STATUS "TensorRT 8.6 not found: falling back to ncnn")
    endif()
endif()

if(ARMVO_USE_TENSORRT)
    # The sibling lib directory needs these imported targets.
    foreach(target TensorRT::nvinfer TensorRT::nvinfer_plugin
                   TensorRT::trtexec CUDA::cudart)
        set_property(TARGET ${target} PROPERTY IMPORTED_GLOBAL TRUE)
    endforeach()
    message(STATUS "Inference backend: TENSORRT (${TensorRT_VERSION})")
else()
    message(STATUS "Inference backend: NCNN")
endif()

set(ARMVO_USE_TENSORRT "${ARMVO_USE_TENSORRT}" PARENT_SCOPE)

set(ARMVO_DEPENDENCY_CACHE "${CMAKE_SOURCE_DIR}/.cache/3rd-party" CACHE PATH
    "Persistent dependency builds and private installations")
get_filename_component(ARMVO_DEPENDENCY_CACHE "${ARMVO_DEPENDENCY_CACHE}" ABSOLUTE
    BASE_DIR "${PROJECT_SOURCE_DIR}")
if(NOT ARMVO_INNER_BUILD)
    include(DependencySettings)
endif()

set(ARMVO_PACKAGE_ARGS "")
set(ARMVO_DEPENDENCY_TARGETS "")
set(ARMVO_DEPENDENCY_BUILDS "")

add_subdirectory(opencv)

if(NOT ARMVO_USE_TENSORRT)
    add_subdirectory(ncnn)
endif()

if(BUILD_TESTS)
    add_subdirectory(catch2)
endif()

# Return the selected packages and build/install work to the outer superbuild.
foreach(variable ARMVO_EXTERNAL_CMAKE_ARGS ARMVO_PACKAGE_ARGS
                 ARMVO_DEPENDENCY_TARGETS ARMVO_DEPENDENCY_BUILDS)
    set(${variable} "${${variable}}" PARENT_SCOPE)
endforeach()
