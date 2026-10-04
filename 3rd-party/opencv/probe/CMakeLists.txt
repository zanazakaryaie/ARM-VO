# Probe in a separate directory: an OpenCV package can create imported targets
# even when it is missing a requested component. Only expose a complete package.

# It was in OpenCV 4.5.0 that USAC_FAST and USAC_MAGSAC were introduced for geometric estimation.
find_package(OpenCV 4.5.0 QUIET CONFIG COMPONENTS ${ARMVO_OPENCV_COMPONENTS})

if(OpenCV_FOUND)
    foreach(module IN LISTS OpenCV_LIB_COMPONENTS)
        if(TARGET ${module})
            set_property(TARGET ${module} PROPERTY IMPORTED_GLOBAL TRUE)
        endif()
    endforeach()
endif()
set(ARMVO_OPENCV_FOUND "${OpenCV_FOUND}" PARENT_SCOPE)
set(ARMVO_OPENCV_VERSION "${OpenCV_VERSION}" PARENT_SCOPE)
