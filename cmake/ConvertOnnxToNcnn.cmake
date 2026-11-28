function(convertOnnxToNcnn onnx output_dir)
    # Model files are portable. Always run converters built for the host.
    set(host_paths HINTS "${ARMVO_NCNN_HOST_BINDIR}" ENV PATH NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_program(ONNX2NCNN_EXECUTABLE onnx2ncnn ${host_paths} REQUIRED)
    find_program(NCNNOPTIMIZE_EXECUTABLE ncnnoptimize ${host_paths} REQUIRED)

    set(storage_flag 0)
    if(SEG_PRECISION STREQUAL "fp16")
        set(storage_flag 65536)
    endif()

    # This fully convolutional graph uses the input dimensions supplied by the
    # segmentor from model.shape. Its ncnn weights do not depend on those dimensions.
    set(work "${output_dir}/ncnn")
    add_custom_command(
        OUTPUT "${output_dir}/model.param" "${output_dir}/model.bin"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${work}"
        COMMAND "${ONNX2NCNN_EXECUTABLE}" "${onnx}" "${work}/raw.param" "${work}/raw.bin"
        COMMAND "${NCNNOPTIMIZE_EXECUTABLE}" "${work}/raw.param" "${work}/raw.bin"
                "${work}/model.param" "${work}/model.bin" "${storage_flag}"
        COMMAND "${CMAKE_COMMAND}" -E rename "${work}/model.bin" "${output_dir}/model.bin"
        COMMAND "${CMAKE_COMMAND}" -E rename "${work}/model.param" "${output_dir}/model.param"
        COMMAND "${CMAKE_COMMAND}" -E rm -f "${work}/raw.param" "${work}/raw.bin"
        DEPENDS "${onnx}" "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
                "${ONNX2NCNN_EXECUTABLE}" "${NCNNOPTIMIZE_EXECUTABLE}"
        COMMENT "Converting segmentation model to ncnn (${SEG_PRECISION})"
        VERBATIM
    )
endfunction()
