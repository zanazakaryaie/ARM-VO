add_executable(run_armvo src/run_armvo.cpp)
target_include_directories(run_armvo PRIVATE include)
target_link_libraries(run_armvo PRIVATE ArmVO ArmVOtools)
install(TARGETS run_armvo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_executable(convert_onnx2ncnn src/convert_onnx_to_ncnn.cpp)
target_include_directories(convert_onnx2ncnn PRIVATE include)
target_link_libraries(convert_onnx2ncnn PRIVATE ArmVO ArmVOtools)
install(TARGETS convert_onnx2ncnn
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
