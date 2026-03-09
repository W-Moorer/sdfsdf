# Source CMakeLists.txt

# 几何模块
add_library(vde_geometry INTERFACE)
target_include_directories(vde_geometry INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 主库链接所有模块
target_link_libraries(vde_core INTERFACE
    vde_geometry
)
