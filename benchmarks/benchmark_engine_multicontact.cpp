#include "BenchmarkCommon.h"
#include "engine/SimulationEngine.h"
#include "solver/PenaltySolver.h"
#include <array>
#include <iostream>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

struct BodyShape {
    std::shared_ptr<RigidBody> body;
    std::shared_ptr<SDF> local_sdf;
    AABB local_aabb;
};

RigidBodyProperties makeBoxProps(double mass, double size) {
    return RigidBodyProperties::box(mass, size, size, size);
}

AABB transformLocalAABB(const AABB& local_aabb, const RigidBodyState& state) {
    const Eigen::Vector3d& min = local_aabb.min;
    const Eigen::Vector3d& max = local_aabb.max;
    const std::array<Eigen::Vector3d, 8> local_corners = {{
        {min.x(), min.y(), min.z()},
        {min.x(), min.y(), max.z()},
        {min.x(), max.y(), min.z()},
        {min.x(), max.y(), max.z()},
        {max.x(), min.y(), min.z()},
        {max.x(), min.y(), max.z()},
        {max.x(), max.y(), min.z()},
        {max.x(), max.y(), max.z()}
    }};

    AABB world_aabb;
    for (const auto& corner : local_corners) {
        world_aabb.expand(state.localToWorld(corner));
    }
    return world_aabb;
}

double runPenaltyStep(std::vector<BodyShape>& bodies,
                      VolumetricIntegrator& integrator,
                      PenaltySolver& penalty_solver,
                      double dt,
                      const Eigen::Vector3d& gravity) {
    double max_penetration = 0.0;

    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            auto& body_a = bodies[i];
            auto& body_b = bodies[j];
            if (body_a.body->isStatic() && body_b.body->isStatic()) {
                continue;
            }

            const AABB aabb_a = transformLocalAABB(body_a.local_aabb, body_a.body->state());
            const AABB aabb_b = transformLocalAABB(body_b.local_aabb, body_b.body->state());
            if (!aabb_a.intersects(aabb_b)) {
                continue;
            }

            TransformedSDF sdf_a(body_a.local_sdf, body_a.body->state().position, body_a.body->state().orientation);
            TransformedSDF sdf_b(body_b.local_sdf, body_b.body->state().position, body_b.body->state().orientation);
            const ContactGeometry geometry = integrator.computeContactGeometry(sdf_a, sdf_b, aabb_a, aabb_b);
            if (geometry.g_eq >= 0.0 || geometry.volume <= 0.0) {
                continue;
            }

            ContactConstraint constraint;
            constraint.computeJacobiansFromGeometry(
                geometry,
                body_a.body->centerOfMassWorld(),
                body_b.body->centerOfMassWorld(),
                1e-4);
            penalty_solver.solveContact(*body_a.body, *body_b.body, constraint, dt);
            max_penetration = std::max(max_penetration, -geometry.g_eq);
        }
    }

    for (auto& item : bodies) {
        item.body->integrate(dt, gravity);
    }

    return max_penetration;
}

double kineticEnergy(const std::vector<BodyShape>& bodies) {
    double total = 0.0;
    for (const auto& item : bodies) {
        total += item.body->kineticEnergy();
    }
    return total;
}

}  // namespace

int main() {
    constexpr double box_size = 0.5;
    constexpr double half = 0.25;
    constexpr double dt = 0.01;
    constexpr double duration = 0.5;

    SimulationConfig config;
    config.time_step = dt;
    config.enable_friction = false;
    config.contact_grid_resolution = 28;
    config.contact_p_norm = 2.0;
    config.baumgarte_gamma = 0.10;
    config.max_solver_iterations = 160;
    config.solver_tolerance = 1e-8;

    SimulationEngine soccp_engine(config);

    auto lower_box = std::make_shared<RigidBody>(makeBoxProps(10.0, box_size));
    lower_box->setPosition(Eigen::Vector3d(0.0, 0.23, 0.0));
    auto upper_box = std::make_shared<RigidBody>(makeBoxProps(1.0, box_size));
    upper_box->setPosition(Eigen::Vector3d(0.0, 0.71, 0.0));
    auto ground = std::make_shared<RigidBody>();
    ground->setStatic(true);

    auto box_sdf = std::make_shared<BoxSDF>(
        Eigen::Vector3d::Constant(-half),
        Eigen::Vector3d::Constant(half));
    auto ground_sdf = std::make_shared<HalfSpaceSDF>(
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d::Zero());
    const AABB box_aabb(
        Eigen::Vector3d::Constant(-half),
        Eigen::Vector3d::Constant(half));
    const AABB ground_aabb =
        VolumetricIntegrator::halfSpaceAABB(*ground_sdf, Eigen::Vector3d::Zero(), 3.0);

    soccp_engine.addBody(lower_box, box_sdf, box_aabb);
    soccp_engine.addBody(upper_box, box_sdf, box_aabb);
    soccp_engine.addBody(ground, ground_sdf, ground_aabb);

    std::vector<BodyShape> penalty_bodies;
    penalty_bodies.push_back({std::make_shared<RigidBody>(makeBoxProps(10.0, box_size)), box_sdf, box_aabb});
    penalty_bodies.push_back({std::make_shared<RigidBody>(makeBoxProps(1.0, box_size)), box_sdf, box_aabb});
    penalty_bodies.push_back({std::make_shared<RigidBody>(), ground_sdf, ground_aabb});
    penalty_bodies[0].body->setPosition(Eigen::Vector3d(0.0, 0.23, 0.0));
    penalty_bodies[1].body->setPosition(Eigen::Vector3d(0.0, 0.71, 0.0));
    penalty_bodies[2].body->setStatic(true);

    VolumetricIntegrator penalty_integrator(config.contact_grid_resolution, config.contact_p_norm);
    PenaltyParameters penalty_params;
    penalty_params.stiffness = 50000.0;
    penalty_params.damping = 120.0;
    penalty_params.friction_coefficient = 0.0;
    PenaltySolver penalty_solver(penalty_params);

    std::vector<std::vector<std::string>> rows;
    rows.reserve(static_cast<size_t>(duration / dt) + 1);

    double peak_soccp_penetration = 0.0;
    double peak_penalty_penetration = 0.0;
    double peak_soccp_residual = 0.0;
    double max_soccp_iterations = 0.0;
    double residual_exceed_steps = 0.0;

    const int num_steps = static_cast<int>(duration / dt);
    for (int step = 0; step < num_steps; ++step) {
        const SimulationStats soccp_stats = soccp_engine.step();
        const double soccp_penetration = maxPenetrationFromContacts(soccp_engine.getContacts());
        const double penalty_penetration = runPenaltyStep(
            penalty_bodies,
            penalty_integrator,
            penalty_solver,
            dt,
            config.gravity);

        peak_soccp_penetration = std::max(peak_soccp_penetration, soccp_penetration);
        peak_penalty_penetration = std::max(peak_penalty_penetration, penalty_penetration);
        peak_soccp_residual = std::max(peak_soccp_residual, soccp_stats.solver_residual);
        max_soccp_iterations = std::max(max_soccp_iterations, static_cast<double>(soccp_stats.solver_iterations));
        if (soccp_stats.solver_residual > 1e-4) {
            residual_exceed_steps += 1.0;
        }

        rows.push_back(vectorToRow({
            soccp_engine.currentTime(),
            soccp_penetration,
            penalty_penetration,
            soccp_stats.solver_residual,
            static_cast<double>(soccp_stats.solver_iterations),
            soccp_stats.total_energy,
            kineticEnergy(penalty_bodies)
        }));
    }

    const double penetration_reduction =
        (peak_penalty_penetration - peak_soccp_penetration) /
        std::max(peak_penalty_penetration, 1e-12);

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_multicontact.csv",
        {"time", "soccp_penetration", "penalty_penetration", "soccp_residual", "soccp_iterations", "soccp_energy", "penalty_energy"},
        rows);
    writeCsv(
        dir / "benchmark_engine_multicontact_summary.csv",
        {"peak_soccp_penetration", "peak_penalty_penetration", "penetration_reduction", "peak_soccp_residual", "max_soccp_iterations", "residual_exceed_steps"},
        {vectorToRow({
            peak_soccp_penetration,
            peak_penalty_penetration,
            penetration_reduction,
            peak_soccp_residual,
            max_soccp_iterations,
            residual_exceed_steps
        })});

    std::cout << "benchmark_engine_multicontact.csv written to "
              << (dir / "benchmark_engine_multicontact.csv") << '\n';
    std::cout << "peak_soccp_penetration=" << peak_soccp_penetration
              << " peak_penalty_penetration=" << peak_penalty_penetration
              << " penetration_reduction=" << penetration_reduction << '\n';
    return 0;
}
