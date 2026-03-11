set(VDE_BENCHMARK_SOURCES
    benchmark_cylinder_spin.cpp
    benchmark_sharp_drop.cpp
    benchmark_mass_ratio_stack.cpp
    benchmark_resolution_convergence.cpp
    benchmark_engine_multicontact.cpp
    benchmark_engine_tripod_landing.cpp
    benchmark_engine_oblique_box_friction.cpp
    benchmark_engine_oblique_box_polyhedral_compare.cpp
    benchmark_engine_oblique_box_4d_reference.cpp
    benchmark_penalty_tripod_tuning.cpp
    benchmark_engine_torsion_compare.cpp
    benchmark_torsion_friction_reference.cpp
    benchmark_4d_coupled_friction_reference.cpp
    benchmark_engine_mesh_sdf_drop.cpp
    benchmark_engine_mesh_sdf_multicontact.cpp
    benchmark_sphere_drop_compare.cpp
    benchmark_engine_sliding_friction_reference.cpp
)

foreach(source_file IN LISTS VDE_BENCHMARK_SOURCES)
    get_filename_component(target_name "${source_file}" NAME_WE)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE vde_core)
    target_compile_definitions(${target_name} PRIVATE
        VDE_RESULTS_DIR="${CMAKE_SOURCE_DIR}/results"
    )
endforeach()
