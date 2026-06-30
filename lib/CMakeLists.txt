find_package(ncnn REQUIRED)

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
target_link_libraries(ArmVO PUBLIC ${OpenCV_LIBS} PRIVATE ncnn dl)
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

if(BUILD_TESTS)
    add_subdirectory(tests)
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

install(TARGETS ArmVO
    EXPORT armvoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/armvo DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES
    "${PROJECT_SOURCE_DIR}/model/model.bin"
    "${PROJECT_SOURCE_DIR}/model/model.param"
    "${PROJECT_SOURCE_DIR}/model/model.shape"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/armvo/model"
)
