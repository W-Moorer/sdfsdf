# Tests CMakeLists.txt

add_executable(test_sdf test_sdf.cpp)
target_link_libraries(test_sdf PRIVATE vde_core)

add_executable(test_phase1 test_phase1.cpp)
target_link_libraries(test_phase1 PRIVATE vde_core)

add_executable(test_phase2 test_phase2.cpp)
target_link_libraries(test_phase2 PRIVATE vde_core)

# 添加测试
enable_testing()
add_test(NAME SDFTest COMMAND test_sdf)
add_test(NAME Phase1Test COMMAND test_phase1)
add_test(NAME Phase2Test COMMAND test_phase2)
