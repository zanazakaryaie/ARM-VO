include(FindPackageHandleStandardArgs)

set(_TensorRT_HINTS
    "$ENV{TensorRT_ROOT}"
    "$ENV{TENSORRT_ROOT}"
    "$ENV{TRT_ROOT}"
    /usr
    /usr/local
    /usr/local/TensorRT
    /usr/src/tensorrt
    /workspace/tensorrt
)

find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES include
)

find_library(TensorRT_NVINFER_LIBRARY
    NAMES nvinfer
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

find_library(TensorRT_NVINFER_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

find_program(TensorRT_TRTEXEC_EXECUTABLE
    NAMES trtexec
    HINTS ${_TensorRT_HINTS}
    PATH_SUFFIXES bin
)

if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _TensorRT_MAJOR_LINE REGEX "^#define NV_TENSORRT_MAJOR [0-9]+")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _TensorRT_MINOR_LINE REGEX "^#define NV_TENSORRT_MINOR [0-9]+")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _TensorRT_PATCH_LINE REGEX "^#define NV_TENSORRT_PATCH [0-9]+")

    string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*" "\\1" TensorRT_VERSION_MAJOR "${_TensorRT_MAJOR_LINE}")
    string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*" "\\1" TensorRT_VERSION_MINOR "${_TensorRT_MINOR_LINE}")
    string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*" "\\1" TensorRT_VERSION_PATCH "${_TensorRT_PATCH_LINE}")
    set(TensorRT_VERSION "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

find_package(CUDAToolkit QUIET)

find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TensorRT_INCLUDE_DIR
        TensorRT_NVINFER_LIBRARY
        TensorRT_NVINFER_PLUGIN_LIBRARY
        TensorRT_TRTEXEC_EXECUTABLE
        CUDAToolkit_FOUND
    VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_NVINFER_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
endif()

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer_plugin)
    add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
        IMPORTED_LOCATION "${TensorRT_NVINFER_PLUGIN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )
endif()

if(TensorRT_FOUND AND NOT TARGET TensorRT::trtexec)
    add_executable(TensorRT::trtexec IMPORTED)
    set_target_properties(TensorRT::trtexec PROPERTIES
        IMPORTED_LOCATION "${TensorRT_TRTEXEC_EXECUTABLE}"
    )
endif()

mark_as_advanced(
    TensorRT_INCLUDE_DIR
    TensorRT_NVINFER_LIBRARY
    TensorRT_NVINFER_PLUGIN_LIBRARY
    TensorRT_TRTEXEC_EXECUTABLE
)
