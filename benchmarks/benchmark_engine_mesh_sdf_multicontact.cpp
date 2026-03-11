#include "BenchmarkCommon.h"
#include "engine/SimulationEngine.h"
#include "geometry/MeshSDF.h"
#include <algorithm>
#include <iostream>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

RigidBodyProperties makeBoxProps(double mass, double size) {
    return RigidBodyProperties::box(mass, size, size, size);
}

std::shared_ptr<SDF> makeBoxSDF(double half, bool use_mesh) {
    if (use_mesh) {
        return std::make_shared<MeshSDF>(TriangleMesh::makeBox(
            Eigen::Vector3d::Constant(-half),
            Eigen::Vector3d::Constant(half)));
    }
    return std::make_shared<BoxSDF>(
        Eigen::Vector3d::Constant(-half),
        Eigen::Vector3d::Constant(half));
}

void populateScene(SimulationEngine& engine,
                   bool use_mesh_boxes,
                   std::shared_ptr<RigidBody>& upper_box_out) {
    constexpr double box_size = 0.5;
    constexpr double half = 0.25;

    auto lower_box = std::make_shared<RigidBody>(makeBoxProps(10.0, box_size));
    lower_box->setPosition(Eigen::Vector3d(0.0, 0.23, 0.0));
    upper_box_out = std::make_shared<RigidBody>(makeBoxProps(1.0, box_size));
    upper_box_out->setPosition(Eigen::Vector3d(0.0, 0.71, 0.0));
    auto ground = std::make_shared<RigidBody>();
    ground->setStatic(true);

    const std::shared_ptr<SDF> box_sdf = makeBoxSDF(half, use_mesh_boxes);
    const auto ground_sdf = std::make_shared<HalfSpaceSDF>(
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d::Zero());
    const AABB box_aabb(
        Eigen::Vector3d::Constant(-half),
        Eigen::Vector3d::Constant(half));
    const AABB ground_aabb =
        VolumetricIntegrator::halfSpaceAABB(*ground_sdf, Eigen::Vector3d::Zero(), 3.0);

    engine.addBody(lower_box, box_sdf, box_aabb);
    engine.addBody(upper_box_out, box_sdf, box_aabb);
    engine.addBody(ground, ground_sdf, ground_aabb);
}

}  // namespace

int main() {
    SimulationConfig config;
    config.time_step = 0.01;
    config.enable_friction = false;
    config.enable_torsional_friction = false;
    config.contact_grid_resolution = 28;
    config.contact_p_norm = 2.0;
    config.baumgarte_gamma = 0.10;
    config.max_solver_iterations = 160;
    config.solver_tolerance = 1e-8;

    SimulationEngine analytic_engine(config);
    SimulationEngine mesh_engine(config);
    std::shared_ptr<RigidBody> analytic_upper;
    std::shared_ptr<RigidBody> mesh_upper;
    populateScene(analytic_engine, false, analytic_upper);
    populateScene(mesh_engine, true, mesh_upper);

    constexpr double duration = 0.5;
    const int num_steps = static_cast<int>(duration / config.time_step);
    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);

    double max_upper_y_diff = 0.0;
    double max_upper_vy_diff = 0.0;
    double max_penetration_diff = 0.0;
    double max_contact_diff = 0.0;
    double peak_penetration_analytic = 0.0;
    double peak_penetration_mesh = 0.0;
    double peak_scaled_residual_analytic = 0.0;
    double peak_scaled_residual_mesh = 0.0;
    double peak_complementarity_analytic = 0.0;
    double peak_complementarity_mesh = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        const SimulationStats analytic_stats = analytic_engine.step();
        const SimulationStats mesh_stats = mesh_engine.step();
        const double analytic_penetration =
            maxPenetrationFromContacts(analytic_engine.getContacts());
        const double mesh_penetration =
            maxPenetrationFromContacts(mesh_engine.getContacts());

        max_upper_y_diff = std::max(
            max_upper_y_diff,
            std::abs(analytic_upper->position().y() - mesh_upper->position().y()));
        max_upper_vy_diff = std::max(
            max_upper_vy_diff,
            std::abs(analytic_upper->linearVelocity().y() - mesh_upper->linearVelocity().y()));
        max_penetration_diff = std::max(
            max_penetration_diff,
            std::abs(analytic_penetration - mesh_penetration));
        max_contact_diff = std::max(
            max_contact_diff,
            std::abs(
                static_cast<double>(analytic_engine.getContacts().size()) -
                static_cast<double>(mesh_engine.getContacts().size())));
        peak_penetration_analytic = std::max(peak_penetration_analytic, analytic_penetration);
        peak_penetration_mesh = std::max(peak_penetration_mesh, mesh_penetration);
        peak_scaled_residual_analytic = std::max(
            peak_scaled_residual_analytic,
            analytic_stats.solver_scaled_residual);
        peak_scaled_residual_mesh = std::max(
            peak_scaled_residual_mesh,
            mesh_stats.solver_scaled_residual);
        peak_complementarity_analytic = std::max(
            peak_complementarity_analytic,
            analytic_stats.solver_complementarity_violation);
        peak_complementarity_mesh = std::max(
            peak_complementarity_mesh,
            mesh_stats.solver_complementarity_violation);

        rows.push_back(vectorToRow({
            analytic_engine.currentTime(),
            analytic_upper->position().y(),
            mesh_upper->position().y(),
            analytic_upper->linearVelocity().y(),
            mesh_upper->linearVelocity().y(),
            analytic_penetration,
            mesh_penetration,
            static_cast<double>(analytic_engine.getContacts().size()),
            static_cast<double>(mesh_engine.getContacts().size()),
            analytic_stats.solver_scaled_residual,
            mesh_stats.solver_scaled_residual,
            analytic_stats.solver_complementarity_violation,
            mesh_stats.solver_complementarity_violation
        }));
    }

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_mesh_sdf_multicontact.csv",
        {
            "time",
            "upper_y_analytic",
            "upper_y_mesh",
            "upper_vy_analytic",
            "upper_vy_mesh",
            "penetration_analytic",
            "penetration_mesh",
            "contacts_analytic",
            "contacts_mesh",
            "scaled_residual_analytic",
            "scaled_residual_mesh",
            "complementarity_analytic",
            "complementarity_mesh"
        },
        rows);
    writeCsv(
        dir / "benchmark_engine_mesh_sdf_multicontact_summary.csv",
        {
            "max_upper_y_diff",
            "max_upper_vy_diff",
            "max_penetration_diff",
            "max_contact_diff",
            "peak_penetration_analytic",
            "peak_penetration_mesh",
            "peak_scaled_residual_analytic",
            "peak_scaled_residual_mesh",
            "peak_complementarity_analytic",
            "peak_complementarity_mesh"
        },
        {vectorToRow({
            max_upper_y_diff,
            max_upper_vy_diff,
            max_penetration_diff,
            max_contact_diff,
            peak_penetration_analytic,
            peak_penetration_mesh,
            peak_scaled_residual_analytic,
            peak_scaled_residual_mesh,
            peak_complementarity_analytic,
            peak_complementarity_mesh
        })});

    std::cout << "benchmark_engine_mesh_sdf_multicontact.csv written to "
              << (dir / "benchmark_engine_mesh_sdf_multicontact.csv") << '\n';
    std::cout << "max_upper_y_diff=" << max_upper_y_diff
              << " max_upper_vy_diff=" << max_upper_vy_diff
              << " max_penetration_diff=" << max_penetration_diff << '\n';
    return 0;
}
