add_executable(test_file_tools test_file_tools.cpp)
target_link_libraries(test_file_tools PRIVATE Catch2::Catch2WithMain ArmVOtools opencv_imgcodecs opencv_imgproc)
catch_discover_tests(test_file_tools)

add_executable(test_error_estimator test_error_estimator.cpp)
target_link_libraries(test_error_estimator PRIVATE Catch2::Catch2WithMain ArmVOtools)
catch_discover_tests(test_error_estimator)

set_target_properties(test_file_tools test_error_estimator PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
