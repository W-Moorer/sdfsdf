# Tests CMakeLists.txt

add_executable(test_sdf test_sdf.cpp)
target_link_libraries(test_sdf PRIVATE vde_core)

add_executable(test_phase1 test_phase1.cpp)
target_link_libraries(test_phase1 PRIVATE vde_core)

add_executable(test_phase2 test_phase2.cpp)
target_link_libraries(test_phase2 PRIVATE vde_core)

add_executable(test_phase3 test_phase3.cpp)
target_link_libraries(test_phase3 PRIVATE vde_core)

add_executable(test_phase4 test_phase4.cpp)
target_link_libraries(test_phase4 PRIVATE vde_core)

add_executable(test_phase5 test_phase5.cpp)
target_link_libraries(test_phase5 PRIVATE vde_core)

# 测试由顶层 CTest 统一启用，这里只注册测试项。
add_test(NAME SDFTest COMMAND test_sdf)
set_tests_properties(SDFTest PROPERTIES LABELS "sdf;geometry;baseline")
add_test(NAME Phase1Test COMMAND test_phase1)
set_tests_properties(Phase1Test PROPERTIES LABELS "phase1;geometry;baseline")
add_test(NAME Phase2Test COMMAND test_phase2)
set_tests_properties(Phase2Test PROPERTIES LABELS "phase2;solver;baseline")
add_test(NAME Phase3Test COMMAND test_phase3)
set_tests_properties(Phase3Test PROPERTIES LABELS "phase3;dynamics;baseline")
add_test(NAME Phase4Test COMMAND test_phase4)
set_tests_properties(Phase4Test PROPERTIES LABELS "phase4;materials;baseline")
add_test(NAME Phase5Test COMMAND test_phase5)
set_tests_properties(Phase5Test PROPERTIES LABELS "phase5;engine;baseline")
