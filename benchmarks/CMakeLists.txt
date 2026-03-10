set(VDE_BENCHMARK_SOURCES
    benchmark_cylinder_spin.cpp
    benchmark_sharp_drop.cpp
    benchmark_mass_ratio_stack.cpp
    benchmark_resolution_convergence.cpp
    benchmark_engine_multicontact.cpp
    benchmark_engine_tripod_landing.cpp
    benchmark_engine_torsion_compare.cpp
    benchmark_sphere_drop_compare.cpp
)

foreach(source_file IN LISTS VDE_BENCHMARK_SOURCES)
    get_filename_component(target_name "${source_file}" NAME_WE)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE vde_core)
    target_compile_definitions(${target_name} PRIVATE
        VDE_RESULTS_DIR="${CMAKE_SOURCE_DIR}/results"
    )
endforeach()
