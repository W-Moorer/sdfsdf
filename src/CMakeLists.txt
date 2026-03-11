add_library(vde_geometry INTERFACE)
target_include_directories(vde_geometry INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(vde_core INTERFACE
    vde_geometry
)
