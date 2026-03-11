#include "BenchmarkCommon.h"
#include "engine/SimulationEngine.h"
#include "geometry/MeshSDF.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

struct SceneConfig {
    double duration = 0.9;
    double sphere_radius = 0.20;
    double sphere_mass = 1.0;
    double support_width = 12.0;
    double support_height = 2.0;
    double support_depth = 12.0;
    Eigen::Vector3d initial_position = Eigen::Vector3d(0.0, 2.0, 0.0);
};

SimulationConfig makeConfig() {
    SimulationConfig config;
    config.time_step = 0.0025;
    config.enable_friction = false;
    config.enable_torsional_friction = false;
    config.contact_grid_resolution = 28;
    config.contact_p_norm = 2.0;
    config.baumgarte_gamma = 0.0;
    config.max_solver_iterations = 160;
    config.solver_tolerance = 1e-8;
    config.solver_mode = ContactSolverMode::SOCCP;
    return config;
}

int populateScene(SimulationEngine& engine,
                  const SceneConfig& scene,
                  bool use_mesh_support) {
    auto sphere = std::make_shared<RigidBody>(
        RigidBodyProperties::sphere(scene.sphere_mass, scene.sphere_radius));
    sphere->setPosition(scene.initial_position);

    auto support = std::make_shared<RigidBody>(
        RigidBodyProperties::box(1.0, 1.0, 1.0, 1.0));
    support->setStatic(true);
    support->setPosition(Eigen::Vector3d(0.0, -0.5 * scene.support_height, 0.0));

    auto sphere_sdf = std::make_shared<SphereSDF>(Eigen::Vector3d::Zero(), scene.sphere_radius);
    std::shared_ptr<SDF> support_sdf;
    const Eigen::Vector3d half_extents(
        0.5 * scene.support_width,
        0.5 * scene.support_height,
        0.5 * scene.support_depth);
    if (use_mesh_support) {
        support_sdf = std::make_shared<MeshSDF>(TriangleMesh::makeBox(-half_extents, half_extents));
    } else {
        support_sdf = std::make_shared<BoxSDF>(-half_extents, half_extents);
    }

    const AABB sphere_aabb(
        Eigen::Vector3d::Constant(-scene.sphere_radius),
        Eigen::Vector3d::Constant(scene.sphere_radius));
    const AABB support_aabb(-half_extents, half_extents);

    const int id = engine.addBody(sphere, sphere_sdf, sphere_aabb);
    engine.addBody(support, support_sdf, support_aabb);
    return id;
}

}  // namespace

int main() {
    const SceneConfig scene;

    SimulationEngine analytic_engine(makeConfig());
    SimulationEngine mesh_engine(makeConfig());

    const int analytic_id = populateScene(analytic_engine, scene, false);
    const int mesh_id = populateScene(mesh_engine, scene, true);

    const int num_steps = static_cast<int>(scene.duration / analytic_engine.config().time_step);
    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);

    double max_center_y_diff = 0.0;
    double max_vy_diff = 0.0;
    double max_penetration_diff = 0.0;
    double peak_penetration_analytic = 0.0;
    double peak_penetration_mesh = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        analytic_engine.step();
        mesh_engine.step();

        const auto analytic_body = analytic_engine.getBody(analytic_id);
        const auto mesh_body = mesh_engine.getBody(mesh_id);
        const double pen_analytic = maxPenetrationFromContacts(analytic_engine.getContacts());
        const double pen_mesh = maxPenetrationFromContacts(mesh_engine.getContacts());

        max_center_y_diff = std::max(
            max_center_y_diff,
            std::abs(analytic_body->position().y() - mesh_body->position().y()));
        max_vy_diff = std::max(
            max_vy_diff,
            std::abs(analytic_body->linearVelocity().y() - mesh_body->linearVelocity().y()));
        max_penetration_diff = std::max(max_penetration_diff, std::abs(pen_analytic - pen_mesh));
        peak_penetration_analytic = std::max(peak_penetration_analytic, pen_analytic);
        peak_penetration_mesh = std::max(peak_penetration_mesh, pen_mesh);

        rows.push_back(vectorToRow({
            analytic_engine.currentTime(),
            analytic_body->position().y(),
            mesh_body->position().y(),
            analytic_body->linearVelocity().y(),
            mesh_body->linearVelocity().y(),
            pen_analytic,
            pen_mesh
        }));
    }

    const auto analytic_body = analytic_engine.getBody(analytic_id);
    const auto mesh_body = mesh_engine.getBody(mesh_id);
    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_mesh_sdf_drop.csv",
        {
            "time",
            "center_y_analytic",
            "center_y_mesh",
            "vy_analytic",
            "vy_mesh",
            "penetration_analytic",
            "penetration_mesh"
        },
        rows);
    writeCsv(
        dir / "benchmark_engine_mesh_sdf_drop_summary.csv",
        {
            "max_center_y_diff",
            "max_vy_diff",
            "max_penetration_diff",
            "peak_penetration_analytic",
            "peak_penetration_mesh",
            "final_center_y_analytic",
            "final_center_y_mesh"
        },
        {vectorToRow({
            max_center_y_diff,
            max_vy_diff,
            max_penetration_diff,
            peak_penetration_analytic,
            peak_penetration_mesh,
            analytic_body->position().y(),
            mesh_body->position().y()
        })});

    std::cout << "benchmark_engine_mesh_sdf_drop.csv written to "
              << (dir / "benchmark_engine_mesh_sdf_drop.csv") << '\n';
    std::cout << "max_center_y_diff=" << max_center_y_diff
              << " max_vy_diff=" << max_vy_diff
              << " max_penetration_diff=" << max_penetration_diff << '\n';
    return 0;
}
