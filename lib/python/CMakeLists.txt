find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
    pybind11_add_module(armvo src/bindings.cpp)
else()
    find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h REQUIRED)
    Python3_add_library(armvo MODULE src/bindings.cpp WITH_SOABI)
    target_include_directories(armvo PRIVATE "${PYBIND11_INCLUDE_DIR}")
    set_target_properties(armvo PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
    )
endif()

target_link_libraries(armvo PRIVATE ArmVO)
target_compile_features(armvo PRIVATE cxx_std_17)

set_target_properties(armvo PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python"
)

if(BUILD_TESTS)
    add_test(
        NAME armvo_python_bindings
        COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -v
    )
    set_tests_properties(armvo_python_bindings PROPERTIES
        ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:armvo>;PYTHONDONTWRITEBYTECODE=1"
    )
endif()

install(TARGETS armvo
    LIBRARY DESTINATION "${Python3_SITEARCH}"
)
