add_executable(run_armvo src/run_armvo.cpp)
target_include_directories(run_armvo PRIVATE include)
target_link_libraries(run_armvo PRIVATE ArmVO ArmVOtools)
install(TARGETS run_armvo
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

set_target_properties(run_armvo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
