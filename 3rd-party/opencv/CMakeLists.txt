set(ARMVO_OPENCV_COMPONENTS core imgproc video features2d calib3d)
if(BUILD_TOOLS)
    list(APPEND ARMVO_OPENCV_COMPONENTS highgui imgcodecs)
endif()

add_subdirectory(probe)
if(ARMVO_OPENCV_FOUND)
    message(STATUS "Using installed OpenCV ${ARMVO_OPENCV_VERSION}")
    if(NOT ARMVO_INNER_BUILD)
        list(APPEND ARMVO_PACKAGE_ARGS "-DOpenCV_DIR:PATH=${OpenCV_DIR}")
        set(ARMVO_PACKAGE_ARGS "${ARMVO_PACKAGE_ARGS}" PARENT_SCOPE)
    endif()
    return()
endif()
if(ARMVO_INNER_BUILD)
    message(FATAL_ERROR "The prepared OpenCV installation is missing required modules.")
endif()

string(JOIN "," modules ${ARMVO_OPENCV_COMPONENTS})
set(options ${ARMVO_EXTERNAL_CMAKE_ARGS}
    "-DBUILD_LIST:STRING=${modules}"
    -DBUILD_SHARED_LIBS:BOOL=ON
    "-DCMAKE_INSTALL_RPATH:STRING=$ORIGIN"
    -DBUILD_TESTS:BOOL=OFF
    -DBUILD_PERF_TESTS:BOOL=OFF
    -DBUILD_DOCS:BOOL=OFF
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_opencv_python2:BOOL=OFF
    -DBUILD_opencv_python3:BOOL=OFF
    -DBUILD_opencv_apps:BOOL=OFF
    -DBUILD_JAVA:BOOL=OFF
    -DBUILD_opencv_dnn:BOOL=OFF
    -DBUILD_opencv_gapi:BOOL=OFF
    -DBUILD_opencv_videoio:BOOL=OFF
    -DWITH_PROTOBUF:BOOL=OFF
    -DWITH_FLATBUFFERS:BOOL=OFF
    -DWITH_ADE:BOOL=OFF
    -DWITH_JPEG:BOOL=ON
    -DWITH_PNG:BOOL=ON
)
foreach(codec TIFF OPENEXR OPENJPEG JASPER WEBP AVIF GDAL GDCM
              IMGCODEC_HDR IMGCODEC_SUNRASTER IMGCODEC_PXM IMGCODEC_PFM)
    list(APPEND options "-DWITH_${codec}:BOOL=OFF")
endforeach()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|ARM|aarch64|arm64|ARM64)")
    list(APPEND options -DENABLE_NEON:BOOL=ON)
endif()
if(DEFINED OPENCV_DOWNLOAD_PATH)
    list(APPEND options "-DOPENCV_DOWNLOAD_PATH:PATH=${OPENCV_DOWNLOAD_PATH}")
endif()
set(artifacts)
foreach(module ${ARMVO_OPENCV_COMPONENTS} flann)
    list(APPEND artifacts "lib/libopencv_${module}.so.4.10.0")
endforeach()
list(APPEND options ${ARMVO_DEPENDENCY_INSTALL_ARGS})

set(version 4.10.0)
set(repository https://github.com/opencv/opencv.git)
armvo_dependency_cache(opencv "${version}" "${repository}" "${options}"
    "" root)
set(prefix "${root}/install${CMAKE_INSTALL_PREFIX}")
set(ready "${root}/ready")

set(required_files "${ready}" "${root}/build/cmake_install.cmake"
    "${prefix}/lib/cmake/opencv4/OpenCVConfig.cmake")
foreach(artifact IN ITEMS ${artifacts})
    list(APPEND required_files "${prefix}/${artifact}")
endforeach()
set(cached TRUE)
foreach(path IN LISTS required_files)
    if(NOT EXISTS "${path}")
        set(cached FALSE)
    endif()
endforeach()

if(cached)
    message(STATUS "Using cached opencv ${version}: ${prefix}")
else()
    message(STATUS "Will build opencv ${version} in ${root}")
    # Retry an incomplete cache; the dependency's own make reuses object files.
    file(REMOVE "${root}/stamps/armvo_dep_opencv-build"
                "${root}/stamps/armvo_dep_opencv-install")
    # Filter after computing the cache key to preserve existing cached builds.
    # This compatibility setting only exists in CMake 4 and newer.
    if(CMAKE_VERSION VERSION_LESS 4.0)
        list(FILTER options EXCLUDE REGEX "^-DCMAKE_POLICY_VERSION_MINIMUM:")
    endif()
    ExternalProject_Add(armvo_dep_opencv
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
    list(APPEND ARMVO_DEPENDENCY_TARGETS armvo_dep_opencv)
endif()

# Also install cached dependencies when the user runs make install.
list(APPEND ARMVO_DEPENDENCY_BUILDS "${root}/build")
list(APPEND ARMVO_PACKAGE_ARGS
    "-DOpenCV_DIR:PATH=${prefix}/lib/cmake/opencv4")
set(ARMVO_DEPENDENCY_TARGETS "${ARMVO_DEPENDENCY_TARGETS}" PARENT_SCOPE)
set(ARMVO_DEPENDENCY_BUILDS "${ARMVO_DEPENDENCY_BUILDS}" PARENT_SCOPE)
set(ARMVO_PACKAGE_ARGS "${ARMVO_PACKAGE_ARGS}" PARENT_SCOPE)
