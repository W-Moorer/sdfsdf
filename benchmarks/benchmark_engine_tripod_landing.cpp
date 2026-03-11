#include "BenchmarkCommon.h"
#include "engine/SimulationEngine.h"
#include "solver/PenaltySolver.h"
#include <array>
#include <cmath>
#include <iostream>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

constexpr double kPi = 3.14159265358979323846;

struct BodyShape {
    std::shared_ptr<RigidBody> body;
    std::shared_ptr<SDF> local_sdf;
    AABB local_aabb;
};

struct PenaltyStepResult {
    double max_penetration = 0.0;
    int active_contacts = 0;
};

RigidBodyProperties makeBoxProps(double mass,
                                 double width,
                                 double height,
                                 double depth) {
    return RigidBodyProperties::box(mass, width, height, depth);
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

double tiltAngleDegrees(const RigidBody& body) {
    const Eigen::Vector3d world_up =
        body.state().orientation * Eigen::Vector3d::UnitY();
    const double cosine =
        std::clamp(world_up.dot(Eigen::Vector3d::UnitY()), -1.0, 1.0);
    return std::acos(cosine) * 180.0 / kPi;
}

PenaltyStepResult runPenaltyStep(std::vector<BodyShape>& bodies,
                                 VolumetricIntegrator& integrator,
                                 PenaltySolver& penalty_solver,
                                 double dt,
                                 const Eigen::Vector3d& gravity) {
    PenaltyStepResult result;

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

            TransformedSDF sdf_a(
                body_a.local_sdf,
                body_a.body->state().position,
                body_a.body->state().orientation);
            TransformedSDF sdf_b(
                body_b.local_sdf,
                body_b.body->state().position,
                body_b.body->state().orientation);
            const ContactGeometry geometry =
                integrator.computeContactGeometry(sdf_a, sdf_b, aabb_a, aabb_b);
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
            result.max_penetration = std::max(result.max_penetration, -geometry.g_eq);
            result.active_contacts += 1;
        }
    }

    for (auto& item : bodies) {
        item.body->integrate(dt, gravity);
    }

    return result;
}

int populateEngineScene(SimulationEngine& engine,
                        const std::shared_ptr<SDF>& top_sdf,
                        const AABB& top_aabb,
                        const std::shared_ptr<SDF>& support_sdf,
                        const AABB& support_aabb,
                        double support_half,
                        double top_width,
                        double top_height,
                        double top_depth) {
    auto top_box = std::make_shared<RigidBody>(
        makeBoxProps(3.0, top_width, top_height, top_depth));
    top_box->setPosition(Eigen::Vector3d(0.025, 0.48, -0.015));
    top_box->setLinearVelocity(Eigen::Vector3d(0.0, -0.30, 0.0));
    top_box->state().orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(13.0 * kPi / 180.0, Eigen::Vector3d::UnitX())) *
        Eigen::Quaterniond(Eigen::AngleAxisd(-9.0 * kPi / 180.0, Eigen::Vector3d::UnitZ()));

    auto support_a =
        std::make_shared<RigidBody>(makeBoxProps(5.0, 2.0 * support_half, 2.0 * support_half, 2.0 * support_half));
    auto support_b =
        std::make_shared<RigidBody>(makeBoxProps(5.0, 2.0 * support_half, 2.0 * support_half, 2.0 * support_half));
    auto support_c =
        std::make_shared<RigidBody>(makeBoxProps(5.0, 2.0 * support_half, 2.0 * support_half, 2.0 * support_half));
    support_a->setStatic(true);
    support_b->setStatic(true);
    support_c->setStatic(true);
    support_a->setPosition(Eigen::Vector3d(-0.22, support_half, -0.16));
    support_b->setPosition(Eigen::Vector3d(0.22, support_half, -0.16));
    support_c->setPosition(Eigen::Vector3d(0.00, support_half, 0.24));

    const int top_id = engine.addBody(top_box, top_sdf, top_aabb);
    engine.addBody(support_a, support_sdf, support_aabb);
    engine.addBody(support_b, support_sdf, support_aabb);
    engine.addBody(support_c, support_sdf, support_aabb);
    return top_id;
}

}  // namespace

int main() {
    constexpr double dt = 0.01;
    constexpr double duration = 0.8;
    constexpr double support_size = 0.18;
    constexpr double support_half = 0.5 * support_size;
    constexpr double top_width = 0.72;
    constexpr double top_height = 0.16;
    constexpr double top_depth = 0.56;
    const Eigen::Vector3d top_half_extents(
        0.5 * top_width,
        0.5 * top_height,
        0.5 * top_depth);

    SimulationConfig config;
    config.time_step = dt;
    config.enable_friction = false;
    config.contact_grid_resolution = 32;
    config.contact_p_norm = 2.0;
    config.baumgarte_gamma = 0.10;
    config.max_solver_iterations = 180;
    config.solver_tolerance = 1e-8;

    SimulationEngine soccp_engine(config);
    SimulationConfig normal_lcp_config = config;
    normal_lcp_config.solver_mode = ContactSolverMode::NormalLCP;
    SimulationEngine normal_lcp_engine(normal_lcp_config);

    auto top_box = std::make_shared<RigidBody>(
        makeBoxProps(3.0, top_width, top_height, top_depth));
    top_box->setPosition(Eigen::Vector3d(0.025, 0.48, -0.015));
    top_box->setLinearVelocity(Eigen::Vector3d(0.0, -0.30, 0.0));
    top_box->state().orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(13.0 * kPi / 180.0, Eigen::Vector3d::UnitX())) *
        Eigen::Quaterniond(Eigen::AngleAxisd(-9.0 * kPi / 180.0, Eigen::Vector3d::UnitZ()));

    auto support_a = std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size));
    auto support_b = std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size));
    auto support_c = std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size));
    support_a->setStatic(true);
    support_b->setStatic(true);
    support_c->setStatic(true);
    support_a->setPosition(Eigen::Vector3d(-0.22, support_half, -0.16));
    support_b->setPosition(Eigen::Vector3d(0.22, support_half, -0.16));
    support_c->setPosition(Eigen::Vector3d(0.00, support_half, 0.24));

    auto top_sdf = std::make_shared<BoxSDF>(-top_half_extents, top_half_extents);
    auto support_sdf = std::make_shared<BoxSDF>(
        Eigen::Vector3d::Constant(-support_half),
        Eigen::Vector3d::Constant(support_half));
    const AABB top_aabb(-top_half_extents, top_half_extents);
    const AABB support_aabb(
        Eigen::Vector3d::Constant(-support_half),
        Eigen::Vector3d::Constant(support_half));

    const int top_id = soccp_engine.addBody(top_box, top_sdf, top_aabb);
    soccp_engine.addBody(support_a, support_sdf, support_aabb);
    soccp_engine.addBody(support_b, support_sdf, support_aabb);
    soccp_engine.addBody(support_c, support_sdf, support_aabb);

    const int normal_lcp_top_id = populateEngineScene(
        normal_lcp_engine,
        top_sdf,
        top_aabb,
        support_sdf,
        support_aabb,
        support_half,
        top_width,
        top_height,
        top_depth);

    std::vector<BodyShape> penalty_bodies;
    penalty_bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(3.0, top_width, top_height, top_depth)),
        top_sdf,
        top_aabb
    });
    penalty_bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size)),
        support_sdf,
        support_aabb
    });
    penalty_bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size)),
        support_sdf,
        support_aabb
    });
    penalty_bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size)),
        support_sdf,
        support_aabb
    });
    penalty_bodies[0].body->setPosition(Eigen::Vector3d(0.025, 0.48, -0.015));
    penalty_bodies[0].body->setLinearVelocity(Eigen::Vector3d(0.0, -0.30, 0.0));
    penalty_bodies[0].body->state().orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(13.0 * kPi / 180.0, Eigen::Vector3d::UnitX())) *
        Eigen::Quaterniond(Eigen::AngleAxisd(-9.0 * kPi / 180.0, Eigen::Vector3d::UnitZ()));
    penalty_bodies[1].body->setPosition(Eigen::Vector3d(-0.22, support_half, -0.16));
    penalty_bodies[2].body->setPosition(Eigen::Vector3d(0.22, support_half, -0.16));
    penalty_bodies[3].body->setPosition(Eigen::Vector3d(0.00, support_half, 0.24));
    penalty_bodies[1].body->setStatic(true);
    penalty_bodies[2].body->setStatic(true);
    penalty_bodies[3].body->setStatic(true);

    VolumetricIntegrator penalty_integrator(config.contact_grid_resolution, config.contact_p_norm);
    PenaltyParameters penalty_params;
    penalty_params.stiffness = 100000.0;
    penalty_params.damping = 150.0;
    penalty_params.friction_coefficient = 0.0;
    PenaltySolver penalty_solver(penalty_params);

    std::vector<std::vector<std::string>> rows;
    rows.reserve(static_cast<size_t>(duration / dt) + 1);

    double peak_soccp_penetration = 0.0;
    double peak_normal_lcp_penetration = 0.0;
    double peak_penalty_penetration = 0.0;
    double peak_soccp_residual = 0.0;
    double peak_normal_lcp_residual = 0.0;
    double peak_soccp_scaled_residual = 0.0;
    double peak_normal_lcp_scaled_residual = 0.0;
    double peak_soccp_complementarity = 0.0;
    double peak_normal_lcp_complementarity = 0.0;
    double max_soccp_iterations = 0.0;
    double max_normal_lcp_iterations = 0.0;
    double max_soccp_contacts = 0.0;
    double max_normal_lcp_contacts = 0.0;
    double max_penalty_contacts = 0.0;
    double soccp_three_contact_steps = 0.0;
    double normal_lcp_three_contact_steps = 0.0;
    double penalty_three_contact_steps = 0.0;

    const int num_steps = static_cast<int>(duration / dt);
    for (int step = 0; step < num_steps; ++step) {
        const SimulationStats soccp_stats = soccp_engine.step();
        const SimulationStats normal_lcp_stats = normal_lcp_engine.step();
        const double soccp_penetration =
            maxPenetrationFromContacts(soccp_engine.getContacts());
        const double normal_lcp_penetration =
            maxPenetrationFromContacts(normal_lcp_engine.getContacts());
        const PenaltyStepResult penalty_step = runPenaltyStep(
            penalty_bodies,
            penalty_integrator,
            penalty_solver,
            dt,
            config.gravity);

        const auto soccp_top = soccp_engine.getBody(top_id);
        const auto normal_lcp_top = normal_lcp_engine.getBody(normal_lcp_top_id);
        const double soccp_top_y = soccp_top->position().y();
        const double normal_lcp_top_y = normal_lcp_top->position().y();
        const double penalty_top_y = penalty_bodies[0].body->position().y();
        const double soccp_tilt_deg = tiltAngleDegrees(*soccp_top);
        const double normal_lcp_tilt_deg = tiltAngleDegrees(*normal_lcp_top);
        const double penalty_tilt_deg = tiltAngleDegrees(*penalty_bodies[0].body);
        const double soccp_contact_count =
            static_cast<double>(soccp_engine.getContacts().size());
        const double normal_lcp_contact_count =
            static_cast<double>(normal_lcp_engine.getContacts().size());
        const double penalty_contact_count =
            static_cast<double>(penalty_step.active_contacts);

        peak_soccp_penetration =
            std::max(peak_soccp_penetration, soccp_penetration);
        peak_normal_lcp_penetration =
            std::max(peak_normal_lcp_penetration, normal_lcp_penetration);
        peak_penalty_penetration =
            std::max(peak_penalty_penetration, penalty_step.max_penetration);
        peak_soccp_residual =
            std::max(peak_soccp_residual, soccp_stats.solver_residual);
        peak_normal_lcp_residual =
            std::max(peak_normal_lcp_residual, normal_lcp_stats.solver_residual);
        peak_soccp_scaled_residual = std::max(
            peak_soccp_scaled_residual, soccp_stats.solver_scaled_residual);
        peak_normal_lcp_scaled_residual = std::max(
            peak_normal_lcp_scaled_residual, normal_lcp_stats.solver_scaled_residual);
        peak_soccp_complementarity = std::max(
            peak_soccp_complementarity, soccp_stats.solver_complementarity_violation);
        peak_normal_lcp_complementarity = std::max(
            peak_normal_lcp_complementarity,
            normal_lcp_stats.solver_complementarity_violation);
        max_soccp_iterations = std::max(
            max_soccp_iterations,
            static_cast<double>(soccp_stats.solver_iterations));
        max_normal_lcp_iterations = std::max(
            max_normal_lcp_iterations,
            static_cast<double>(normal_lcp_stats.solver_iterations));
        max_soccp_contacts =
            std::max(max_soccp_contacts, soccp_contact_count);
        max_normal_lcp_contacts =
            std::max(max_normal_lcp_contacts, normal_lcp_contact_count);
        max_penalty_contacts =
            std::max(max_penalty_contacts, penalty_contact_count);
        if (soccp_contact_count >= 3.0) {
            soccp_three_contact_steps += 1.0;
        }
        if (normal_lcp_contact_count >= 3.0) {
            normal_lcp_three_contact_steps += 1.0;
        }
        if (penalty_contact_count >= 3.0) {
            penalty_three_contact_steps += 1.0;
        }

        rows.push_back(vectorToRow({
            soccp_engine.currentTime(),
            soccp_penetration,
            normal_lcp_penetration,
            penalty_step.max_penetration,
            soccp_stats.solver_residual,
            normal_lcp_stats.solver_residual,
            soccp_stats.solver_scaled_residual,
            normal_lcp_stats.solver_scaled_residual,
            soccp_stats.solver_complementarity_violation,
            normal_lcp_stats.solver_complementarity_violation,
            static_cast<double>(soccp_stats.solver_iterations),
            static_cast<double>(normal_lcp_stats.solver_iterations),
            soccp_top_y,
            normal_lcp_top_y,
            penalty_top_y,
            soccp_tilt_deg,
            normal_lcp_tilt_deg,
            penalty_tilt_deg,
            soccp_contact_count,
            normal_lcp_contact_count,
            penalty_contact_count
        }));
    }

    const double penetration_reduction =
        (peak_penalty_penetration - peak_soccp_penetration) /
        std::max(peak_penalty_penetration, 1e-12);
    const double normal_lcp_penetration_reduction =
        (peak_penalty_penetration - peak_normal_lcp_penetration) /
        std::max(peak_penalty_penetration, 1e-12);
    const double final_soccp_tilt_deg =
        tiltAngleDegrees(*soccp_engine.getBody(top_id));
    const double final_normal_lcp_tilt_deg =
        tiltAngleDegrees(*normal_lcp_engine.getBody(normal_lcp_top_id));
    const double final_penalty_tilt_deg =
        tiltAngleDegrees(*penalty_bodies[0].body);

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_tripod_landing.csv",
        {
            "time",
            "soccp_penetration",
            "normal_lcp_penetration",
            "penalty_penetration",
            "soccp_residual",
            "normal_lcp_residual",
            "soccp_scaled_residual",
            "normal_lcp_scaled_residual",
            "soccp_complementarity",
            "normal_lcp_complementarity",
            "soccp_iterations",
            "normal_lcp_iterations",
            "soccp_top_y",
            "normal_lcp_top_y",
            "penalty_top_y",
            "soccp_tilt_deg",
            "normal_lcp_tilt_deg",
            "penalty_tilt_deg",
            "soccp_contacts",
            "normal_lcp_contacts",
            "penalty_contacts"
        },
        rows);
    writeCsv(
        dir / "benchmark_engine_tripod_landing_summary.csv",
        {
            "peak_soccp_penetration",
            "peak_normal_lcp_penetration",
            "peak_penalty_penetration",
            "penetration_reduction",
            "normal_lcp_penetration_reduction",
            "peak_soccp_residual",
            "peak_normal_lcp_residual",
            "peak_soccp_scaled_residual",
            "peak_normal_lcp_scaled_residual",
            "peak_soccp_complementarity",
            "peak_normal_lcp_complementarity",
            "max_soccp_iterations",
            "max_normal_lcp_iterations",
            "max_soccp_contacts",
            "max_normal_lcp_contacts",
            "max_penalty_contacts",
            "soccp_three_contact_steps",
            "normal_lcp_three_contact_steps",
            "penalty_three_contact_steps",
            "final_soccp_tilt_deg",
            "final_normal_lcp_tilt_deg",
            "final_penalty_tilt_deg"
        },
        {vectorToRow({
            peak_soccp_penetration,
            peak_normal_lcp_penetration,
            peak_penalty_penetration,
            penetration_reduction,
            normal_lcp_penetration_reduction,
            peak_soccp_residual,
            peak_normal_lcp_residual,
            peak_soccp_scaled_residual,
            peak_normal_lcp_scaled_residual,
            peak_soccp_complementarity,
            peak_normal_lcp_complementarity,
            max_soccp_iterations,
            max_normal_lcp_iterations,
            max_soccp_contacts,
            max_normal_lcp_contacts,
            max_penalty_contacts,
            soccp_three_contact_steps,
            normal_lcp_three_contact_steps,
            penalty_three_contact_steps,
            final_soccp_tilt_deg,
            final_normal_lcp_tilt_deg,
            final_penalty_tilt_deg
        })});

    std::cout << "benchmark_engine_tripod_landing.csv written to "
              << (dir / "benchmark_engine_tripod_landing.csv") << '\n';
    std::cout << "peak_soccp_penetration=" << peak_soccp_penetration
              << " peak_penalty_penetration=" << peak_penalty_penetration
              << " penetration_reduction=" << penetration_reduction << '\n';
    std::cout << "max_soccp_contacts=" << max_soccp_contacts
              << " max_penalty_contacts=" << max_penalty_contacts << '\n';
    return 0;
}
