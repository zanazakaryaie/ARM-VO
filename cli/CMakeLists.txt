add_executable(run_armvo src/run_armvo.cpp)
target_include_directories(run_armvo PRIVATE include)
target_link_libraries(run_armvo PRIVATE ArmVO ArmVOtools)
install(TARGETS run_armvo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_executable(convert_onnx2ncnn src/convert_onnx_to_ncnn.cpp)
target_include_directories(convert_onnx2ncnn PRIVATE include)
target_link_libraries(convert_onnx2ncnn PRIVATE ArmVO ArmVOtools)
set_target_properties(run_armvo convert_onnx2ncnn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
install(TARGETS convert_onnx2ncnn
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
