file(GLOB TOOLS_SOURCES
     CONFIGURE_DEPENDS
     "src/*.cpp")

add_library(ArmVOtools SHARED)
target_sources(ArmVOtools PRIVATE ${TOOLS_SOURCES})
target_include_directories(ArmVOtools
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/armvo/tools>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(ArmVOtools PUBLIC cxx_std_17)
target_link_libraries(ArmVOtools PUBLIC ArmVO ${OpenCV_LIBS})
set_target_properties(ArmVOtools PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME ArmVOtools
)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

install(TARGETS ArmVOtools
    EXPORT armvoTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/armvo DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
