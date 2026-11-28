string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" ARCH)
if(ARCH MATCHES "^(arm|aarch64|arm64)")
    message(TRACE "CPU architecture is ARM")
    set(ARMVO_ARCH_DEFINITION ARM)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(ARMVO_ARCH_COMPILE_OPTIONS -mfpu=neon)
    endif()
elseif(ARCH MATCHES "^(x86_64|amd64|i[3-6]86)")
    message(TRACE "CPU architecture is x86")
    set(ARMVO_ARCH_DEFINITION x86)
endif()


file(GLOB LIB_SOURCES
    CONFIGURE_DEPENDS # re-run CMake if the file list changes
    "src/*.cpp"
    "src/*.hpp")

if(ARMVO_USE_TENSORRT)
    list(FILTER LIB_SOURCES EXCLUDE REGEX ".*/SemanticSegmentorNcnn\\.(cpp|hpp)$")
else()
    list(FILTER LIB_SOURCES EXCLUDE REGEX ".*/SemanticSegmentorTensorRT\\.(cpp|hpp)$")
endif()

add_library(ArmVO SHARED)
target_sources(ArmVO PRIVATE ${LIB_SOURCES})
target_include_directories(ArmVO
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/armvo>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(ArmVO PUBLIC cxx_std_17)
target_compile_definitions(ArmVO PRIVATE ${ARMVO_ARCH_DEFINITION})
target_compile_options(ArmVO PRIVATE ${ARMVO_ARCH_COMPILE_OPTIONS})
target_link_libraries(ArmVO
    PUBLIC
        opencv_core
    PRIVATE
        opencv_imgproc
        opencv_video
        opencv_features2d
        opencv_calib3d
        dl
)

if(ARMVO_USE_TENSORRT)
    target_compile_definitions(ArmVO PRIVATE ARMVO_USE_TENSORRT)
    target_link_libraries(ArmVO PRIVATE TensorRT::nvinfer TensorRT::nvinfer_plugin CUDA::cudart)
else()
    target_compile_definitions(ArmVO PRIVATE ARMVO_USE_NCNN)
    target_link_libraries(ArmVO PRIVATE ncnn)
endif()

set_target_properties(ArmVO PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME ArmVO
)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if(supported)
    set_property(TARGET ArmVO PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(python)
endif()

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

set(model_onnx "${PROJECT_SOURCE_DIR}/model/model.onnx")
set(model_dir "${CMAKE_CURRENT_BINARY_DIR}/model/${SEG_PRECISION}")

if(ARMVO_USE_TENSORRT)
    include(ConvertOnnxToTrt)
    # Include the profile size in the output path so Make also rebuilds on a resolution change.
    set(model_dir "${model_dir}/${SEG_WIDTH}x${SEG_HEIGHT}")
    convertOnnxToTrt("${model_onnx}" "${model_dir}")
    set(model_files "${model_dir}/model.engine")
else()
    include(ConvertOnnxToNcnn)
    convertOnnxToNcnn("${model_onnx}" "${model_dir}")
    file(CONFIGURE OUTPUT "${model_dir}/model.shape" CONTENT "@SEG_HEIGHT@ @SEG_WIDTH@\n" @ONLY)
    set(model_files "${model_dir}/model.param" "${model_dir}/model.bin" "${model_dir}/model.shape")
    set_property(SOURCE src/SemanticSegmentorNcnn.cpp APPEND PROPERTY
        COMPILE_DEFINITIONS "ARMVO_NCNN_FP16=$<STREQUAL:${SEG_PRECISION},fp16>")
endif()

# Stage even when only the model changed and ArmVO does not need to relink.
set(staged_model_dir "$<TARGET_FILE_DIR:ArmVO>/armvo/model")
add_custom_target(ArmVO_model
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${staged_model_dir}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${model_files} "${staged_model_dir}"
    DEPENDS ${model_files}
    VERBATIM
)
add_dependencies(ArmVO ArmVO_model)

install(TARGETS ArmVO
    EXPORT armvoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/armvo DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${model_files} DESTINATION "${CMAKE_INSTALL_LIBDIR}/armvo/model")
