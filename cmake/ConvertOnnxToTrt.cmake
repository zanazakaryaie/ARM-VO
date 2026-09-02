function(convertOnnxToTrt)
    set(oneValueArgs TARGET OUTPUT ONNX MODEL_SHAPE INPUT_NAME)
    cmake_parse_arguments(CONVERT_ONNX_TO_TRT "" "${oneValueArgs}" "" ${ARGN})

    if(NOT CONVERT_ONNX_TO_TRT_TARGET)
        message(FATAL_ERROR "convertOnnxToTrt requires TARGET")
    endif()
    if(NOT CONVERT_ONNX_TO_TRT_OUTPUT)
        message(FATAL_ERROR "convertOnnxToTrt requires OUTPUT")
    endif()
    if(NOT CONVERT_ONNX_TO_TRT_ONNX)
        message(FATAL_ERROR "convertOnnxToTrt requires ONNX")
    endif()
    if(NOT CONVERT_ONNX_TO_TRT_MODEL_SHAPE)
        message(FATAL_ERROR "convertOnnxToTrt requires MODEL_SHAPE")
    endif()
    if(NOT CONVERT_ONNX_TO_TRT_INPUT_NAME)
        set(CONVERT_ONNX_TO_TRT_INPUT_NAME "input_image")
    endif()
    if(NOT TARGET TensorRT::trtexec)
        message(FATAL_ERROR "convertOnnxToTrt requires TensorRT::trtexec")
    endif()

    file(STRINGS "${CONVERT_ONNX_TO_TRT_MODEL_SHAPE}" CONVERT_ONNX_TO_TRT_SHAPE LIMIT_COUNT 1)
    if(NOT CONVERT_ONNX_TO_TRT_SHAPE MATCHES "^[ \t]*([0-9]+)[ \t]+([0-9]+)")
        message(FATAL_ERROR "${CONVERT_ONNX_TO_TRT_MODEL_SHAPE} must contain '<height> <width>'")
    endif()
    set(CONVERT_ONNX_TO_TRT_INPUT_HEIGHT "${CMAKE_MATCH_1}")
    set(CONVERT_ONNX_TO_TRT_INPUT_WIDTH "${CMAKE_MATCH_2}")

    get_filename_component(CONVERT_ONNX_TO_TRT_OUTPUT_DIR "${CONVERT_ONNX_TO_TRT_OUTPUT}" DIRECTORY)

    add_custom_command(
        OUTPUT "${CONVERT_ONNX_TO_TRT_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CONVERT_ONNX_TO_TRT_OUTPUT_DIR}"
        COMMAND TensorRT::trtexec
                "--onnx=${CONVERT_ONNX_TO_TRT_ONNX}"
                "--saveEngine=${CONVERT_ONNX_TO_TRT_OUTPUT}"
                "--shapes=${CONVERT_ONNX_TO_TRT_INPUT_NAME}:1x3x${CONVERT_ONNX_TO_TRT_INPUT_HEIGHT}x${CONVERT_ONNX_TO_TRT_INPUT_WIDTH}"
        DEPENDS "${CONVERT_ONNX_TO_TRT_ONNX}" "${CONVERT_ONNX_TO_TRT_MODEL_SHAPE}"
        COMMENT "Converting ONNX model to TensorRT engine"
        VERBATIM
        # TODO: test fp16
    )

    add_custom_target("${CONVERT_ONNX_TO_TRT_TARGET}" DEPENDS "${CONVERT_ONNX_TO_TRT_OUTPUT}")
endfunction()
