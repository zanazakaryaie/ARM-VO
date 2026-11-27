function(convertOnnxToTrt onnx output_dir)
    set(shape "input_image:1x3x${SEG_HEIGHT}x${SEG_WIDTH}")
    if(SEG_PRECISION STREQUAL "fp16")
        set(precision_flag --fp16)
    endif()

    add_custom_command(
        OUTPUT "${output_dir}/model.engine"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${output_dir}"
        COMMAND TensorRT::trtexec "--onnx=${onnx}" "--saveEngine=${output_dir}/model.engine.tmp"
                "--minShapes=${shape}" "--optShapes=${shape}" "--maxShapes=${shape}"
                ${precision_flag} --skipInference
        COMMAND "${CMAKE_COMMAND}" -E rename "${output_dir}/model.engine.tmp" "${output_dir}/model.engine"
        DEPENDS "${onnx}" "${CMAKE_CURRENT_FUNCTION_LIST_FILE}" TensorRT::trtexec
        COMMENT "Building TensorRT engine (${SEG_WIDTH}x${SEG_HEIGHT}, ${SEG_PRECISION})"
        VERBATIM
    )
endfunction()
