# Run after ncnn's project() has enabled its compilers. This release only checks
# NCNN_ENABLE_LTO for shared libraries, so enable IPO explicitly for the archive.
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error LANGUAGES CXX)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION "${supported}" CACHE BOOL "ncnn LTO" FORCE)
set(NCNN_ENABLE_LTO "${supported}" CACHE BOOL "ncnn LTO" FORCE)
if(supported)
    message(STATUS "ncnn: enabling LTO for the static library")
else()
    message(WARNING "ncnn: LTO unavailable; continuing without it: ${error}")
endif()
