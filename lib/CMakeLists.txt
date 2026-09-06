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

find_package(TensorRT 8.6 QUIET)
if(TensorRT_FOUND AND TensorRT_VERSION VERSION_LESS 9.0)
    set(ARMVO_USE_TENSORRT ON)
    message(STATUS "TensorRT ${TensorRT_VERSION} found: using TensorRT semantic segmentation backend")
else()
    set(ARMVO_USE_TENSORRT OFF)
    if(TensorRT_FOUND)
        message(STATUS "TensorRT ${TensorRT_VERSION} found, but this backend currently targets TensorRT 8.6.x. Falling back to NCNN.")
    else()
        message(STATUS "TensorRT 8.6 not found: falling back to NCNN semantic segmentation backend")
    endif()
    find_package(ncnn REQUIRED)
endif()

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
target_link_libraries(ArmVO PUBLIC ${OpenCV_LIBS} PRIVATE dl)

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

if(ARMVO_USE_TENSORRT)
    include(ConvertOnnxToTrt)

    set(ARMVO_TENSORRT_ENGINE "${CMAKE_CURRENT_BINARY_DIR}/model/model.engine")
    convertOnnxToTrt(
        TARGET ArmVO_tensorrt_engine
        OUTPUT "${ARMVO_TENSORRT_ENGINE}"
        ONNX "${PROJECT_SOURCE_DIR}/model/model.onnx"
        MODEL_SHAPE "${PROJECT_SOURCE_DIR}/model/model.shape"
        INPUT_NAME input_image
    )
    add_dependencies(ArmVO ArmVO_tensorrt_engine)
endif()

add_custom_command(TARGET ArmVO POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -rf
            "$<TARGET_FILE_DIR:ArmVO>/armvo/model"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:ArmVO>/armvo"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_SOURCE_DIR}/model"
            "$<TARGET_FILE_DIR:ArmVO>/armvo/model"
    COMMENT "Staging armvo/model/ next to libArmVO for in-tree runs")

if(ARMVO_USE_TENSORRT)
    add_custom_command(TARGET ArmVO POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ARMVO_TENSORRT_ENGINE}"
                "$<TARGET_FILE_DIR:ArmVO>/armvo/model/model.engine"
        COMMENT "Staging TensorRT engine next to libArmVO")
endif()

install(TARGETS ArmVO
    EXPORT armvoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/armvo DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ARMVO_USE_TENSORRT)
    install(FILES
        "${ARMVO_TENSORRT_ENGINE}"
        "${PROJECT_SOURCE_DIR}/model/model.shape"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/armvo/model"
    )
else()
    install(FILES
        "${PROJECT_SOURCE_DIR}/model/model.bin"
        "${PROJECT_SOURCE_DIR}/model/model.param"
        "${PROJECT_SOURCE_DIR}/model/model.shape"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/armvo/model"
    )
endif()
