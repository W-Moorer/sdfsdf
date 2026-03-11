#include "BenchmarkCommon.h"
#include "engine/SimulationEngine.h"
#include "solver/PenaltySolver.h"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
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

struct PenaltySweepRecord {
    double stiffness = 0.0;
    double damping = 0.0;
    double peak_penetration = 0.0;
    double final_spin = 0.0;
    double final_slip = 0.0;
    double final_tilt = 0.0;
    double score = std::numeric_limits<double>::infinity();
};

struct SceneConfig {
    double dt = 0.01;
    double duration = 2.0;
    double box_width = 0.80;
    double box_height = 0.18;
    double box_depth = 0.62;
    double ground_width = 8.0;
    double ground_height = 1.5;
    double ground_depth = 8.0;
    double friction_coefficient = 0.42;
    double preload_penetration = 0.035;
    Eigen::Vector3d initial_position = Eigen::Vector3d(0.0, 0.5 * box_height - preload_penetration, 0.0);
    Eigen::Vector3d initial_linear_velocity = Eigen::Vector3d(0.30, 0.0, -0.14);
    Eigen::Vector3d initial_angular_velocity = Eigen::Vector3d(0.0, 8.0, 0.0);
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

double spinRateY(const RigidBody& body) {
    return std::abs(body.angularVelocity().y());
}

double horizontalSlip(const RigidBody& body, const Eigen::Vector2d& initial_xz) {
    const Eigen::Vector2d current_xz(body.position().x(), body.position().z());
    return (current_xz - initial_xz).norm();
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

void initializeDynamicBody(RigidBody& body, const SceneConfig& scene) {
    body.setPosition(scene.initial_position);
    body.setLinearVelocity(scene.initial_linear_velocity);
    body.setAngularVelocity(scene.initial_angular_velocity);
    body.state().orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(17.0 * kPi / 180.0, Eigen::Vector3d::UnitY()));
}

int populateEngineScene(SimulationEngine& engine,
                        const std::shared_ptr<SDF>& box_sdf,
                        const AABB& box_aabb,
                        const std::shared_ptr<SDF>& ground_sdf,
                        const AABB& ground_aabb,
                        const SceneConfig& scene) {
    auto box = std::make_shared<RigidBody>(
        makeBoxProps(2.4, scene.box_width, scene.box_height, scene.box_depth));
    initializeDynamicBody(*box, scene);
    auto ground = std::make_shared<RigidBody>(
        makeBoxProps(10.0, scene.ground_width, scene.ground_height, scene.ground_depth));
    ground->setStatic(true);
    ground->setPosition(Eigen::Vector3d(0.0, -0.5 * scene.ground_height, 0.0));

    const int box_id = engine.addBody(box, box_sdf, box_aabb);
    engine.addBody(ground, ground_sdf, ground_aabb);
    return box_id;
}

std::vector<BodyShape> makePenaltyBodies(const std::shared_ptr<SDF>& box_sdf,
                                         const AABB& box_aabb,
                                         const std::shared_ptr<SDF>& ground_sdf,
                                         const AABB& ground_aabb,
                                         const SceneConfig& scene) {
    std::vector<BodyShape> bodies;
    bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(2.4, scene.box_width, scene.box_height, scene.box_depth)),
        box_sdf,
        box_aabb
    });
    bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(10.0, scene.ground_width, scene.ground_height, scene.ground_depth)),
        ground_sdf,
        ground_aabb
    });
    initializeDynamicBody(*bodies[0].body, scene);
    bodies[1].body->setStatic(true);
    bodies[1].body->setPosition(Eigen::Vector3d(0.0, -0.5 * scene.ground_height, 0.0));
    return bodies;
}

PenaltySweepRecord evaluatePenaltyCandidate(double stiffness,
                                            double damping,
                                            const SceneConfig& scene,
                                            const std::shared_ptr<SDF>& box_sdf,
                                            const AABB& box_aabb,
                                            const std::shared_ptr<SDF>& ground_sdf,
                                            const AABB& ground_aabb,
                                            int contact_grid_resolution,
                                            double contact_p_norm,
                                            const Eigen::Vector3d& gravity) {
    PenaltySweepRecord record;
    record.stiffness = stiffness;
    record.damping = damping;

    auto bodies = makePenaltyBodies(box_sdf, box_aabb, ground_sdf, ground_aabb, scene);
    VolumetricIntegrator integrator(contact_grid_resolution, contact_p_norm);
    PenaltyParameters penalty_params;
    penalty_params.stiffness = stiffness;
    penalty_params.damping = damping;
    penalty_params.friction_coefficient = scene.friction_coefficient;
    PenaltySolver penalty_solver(penalty_params);

    const Eigen::Vector2d initial_xz(scene.initial_position.x(), scene.initial_position.z());
    const int num_steps = static_cast<int>(scene.duration / scene.dt);
    bool stable = true;
    for (int step = 0; step < num_steps; ++step) {
        const PenaltyStepResult step_result =
            runPenaltyStep(bodies, integrator, penalty_solver, scene.dt, gravity);
        record.peak_penetration = std::max(record.peak_penetration, step_result.max_penetration);
        const auto& body = *bodies[0].body;
        record.final_spin = spinRateY(body);
        record.final_slip = horizontalSlip(body, initial_xz);
        record.final_tilt = tiltAngleDegrees(body);
        if (!body.position().allFinite() ||
            !body.linearVelocity().allFinite() ||
            !body.angularVelocity().allFinite() ||
            record.final_tilt > 170.0) {
            stable = false;
            break;
        }
    }

    record.score =
        record.peak_penetration +
        2e-4 * record.final_spin +
        2e-4 * record.final_slip +
        5e-4 * record.final_tilt +
        (stable ? 0.0 : 100.0);
    return record;
}

}  // namespace

int main() {
    const SceneConfig scene;
    const Eigen::Vector3d box_half(
        0.5 * scene.box_width,
        0.5 * scene.box_height,
        0.5 * scene.box_depth);
    const Eigen::Vector3d ground_half(
        0.5 * scene.ground_width,
        0.5 * scene.ground_height,
        0.5 * scene.ground_depth);

    SimulationConfig config;
    config.time_step = scene.dt;
    config.enable_friction = true;
    config.enable_torsional_friction = true;
    config.friction_coefficient = scene.friction_coefficient;
    config.contact_grid_resolution = 40;
    config.contact_p_norm = 2.0;
    config.baumgarte_gamma = 0.0;
    config.max_solver_iterations = 180;
    config.solver_tolerance = 1e-8;

    auto box_sdf = std::make_shared<BoxSDF>(-box_half, box_half);
    auto ground_sdf = std::make_shared<BoxSDF>(-ground_half, ground_half);
    const AABB box_aabb(-box_half, box_half);
    const AABB ground_aabb(-ground_half, ground_half);

    const std::vector<double> stiffness_candidates = {20000.0, 30000.0, 45000.0, 60000.0, 80000.0};
    const std::vector<double> damping_candidates = {60.0, 100.0, 140.0, 180.0, 240.0};
    std::vector<PenaltySweepRecord> sweep_records;
    PenaltySweepRecord best_record;
    for (double stiffness : stiffness_candidates) {
        for (double damping : damping_candidates) {
            const PenaltySweepRecord record = evaluatePenaltyCandidate(
                stiffness,
                damping,
                scene,
                box_sdf,
                box_aabb,
                ground_sdf,
                ground_aabb,
                config.contact_grid_resolution,
                config.contact_p_norm,
                config.gravity);
            if (record.score < best_record.score) {
                best_record = record;
            }
            sweep_records.push_back(record);
        }
    }

    SimulationEngine engine_4d(config);
    SimulationConfig config_3d = config;
    config_3d.enable_torsional_friction = false;
    SimulationEngine engine_3d(config_3d);

    const int box_id_4d = populateEngineScene(engine_4d, box_sdf, box_aabb, ground_sdf, ground_aabb, scene);
    const int box_id_3d = populateEngineScene(engine_3d, box_sdf, box_aabb, ground_sdf, ground_aabb, scene);

    std::vector<BodyShape> penalty_bodies = makePenaltyBodies(
        box_sdf, box_aabb, ground_sdf, ground_aabb, scene);
    VolumetricIntegrator penalty_integrator(config.contact_grid_resolution, config.contact_p_norm);
    PenaltyParameters penalty_params;
    penalty_params.stiffness = best_record.stiffness;
    penalty_params.damping = best_record.damping;
    penalty_params.friction_coefficient = scene.friction_coefficient;
    PenaltySolver penalty_solver(penalty_params);

    const int num_steps = static_cast<int>(scene.duration / scene.dt);
    const Eigen::Vector2d initial_xz(scene.initial_position.x(), scene.initial_position.z());
    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);

    double peak_penetration_4d = 0.0;
    double peak_penetration_3d = 0.0;
    double peak_penetration_penalty = 0.0;
    double peak_residual_4d = 0.0;
    double peak_residual_3d = 0.0;
    double peak_scaled_residual_4d = 0.0;
    double peak_scaled_residual_3d = 0.0;
    double peak_complementarity_4d = 0.0;
    double peak_complementarity_3d = 0.0;
    double max_iterations_4d = 0.0;
    double max_iterations_3d = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        const SimulationStats stats_4d = engine_4d.step();
        const SimulationStats stats_3d = engine_3d.step();
        const PenaltyStepResult penalty_step = runPenaltyStep(
            penalty_bodies, penalty_integrator, penalty_solver, scene.dt, config.gravity);

        const auto body_4d = engine_4d.getBody(box_id_4d);
        const auto body_3d = engine_3d.getBody(box_id_3d);
        const auto& body_penalty = *penalty_bodies[0].body;

        const double penetration_4d = maxPenetrationFromContacts(engine_4d.getContacts());
        const double penetration_3d = maxPenetrationFromContacts(engine_3d.getContacts());
        peak_penetration_4d = std::max(peak_penetration_4d, penetration_4d);
        peak_penetration_3d = std::max(peak_penetration_3d, penetration_3d);
        peak_penetration_penalty = std::max(peak_penetration_penalty, penalty_step.max_penetration);
        peak_residual_4d = std::max(peak_residual_4d, stats_4d.solver_residual);
        peak_residual_3d = std::max(peak_residual_3d, stats_3d.solver_residual);
        peak_scaled_residual_4d = std::max(
            peak_scaled_residual_4d, stats_4d.solver_scaled_residual);
        peak_scaled_residual_3d = std::max(
            peak_scaled_residual_3d, stats_3d.solver_scaled_residual);
        peak_complementarity_4d = std::max(
            peak_complementarity_4d, stats_4d.solver_complementarity_violation);
        peak_complementarity_3d = std::max(
            peak_complementarity_3d, stats_3d.solver_complementarity_violation);
        max_iterations_4d = std::max(max_iterations_4d, static_cast<double>(stats_4d.solver_iterations));
        max_iterations_3d = std::max(max_iterations_3d, static_cast<double>(stats_3d.solver_iterations));

        rows.push_back(vectorToRow({
            engine_4d.currentTime(),
            penetration_4d,
            penetration_3d,
            penalty_step.max_penetration,
            stats_4d.solver_residual,
            stats_3d.solver_residual,
            stats_4d.solver_scaled_residual,
            stats_3d.solver_scaled_residual,
            stats_4d.solver_complementarity_violation,
            stats_3d.solver_complementarity_violation,
            static_cast<double>(stats_4d.solver_iterations),
            static_cast<double>(stats_3d.solver_iterations),
            spinRateY(*body_4d),
            spinRateY(*body_3d),
            spinRateY(body_penalty),
            horizontalSlip(*body_4d, initial_xz),
            horizontalSlip(*body_3d, initial_xz),
            horizontalSlip(body_penalty, initial_xz),
            tiltAngleDegrees(*body_4d),
            tiltAngleDegrees(*body_3d),
            tiltAngleDegrees(body_penalty),
            static_cast<double>(engine_4d.getContacts().size()),
            static_cast<double>(engine_3d.getContacts().size()),
            static_cast<double>(penalty_step.active_contacts)
        }));
    }

    std::vector<std::vector<std::string>> tuning_rows;
    tuning_rows.reserve(sweep_records.size());
    for (const auto& record : sweep_records) {
        tuning_rows.push_back(vectorToRow({
            record.stiffness,
            record.damping,
            record.peak_penetration,
            record.final_spin,
            record.final_slip,
            record.final_tilt,
            record.score
        }));
    }

    const auto body_4d_final = engine_4d.getBody(box_id_4d);
    const auto body_3d_final = engine_3d.getBody(box_id_3d);
    const auto& body_penalty_final = *penalty_bodies[0].body;
    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_oblique_box_friction.csv",
        {
            "time",
            "penetration_4d",
            "penetration_3d",
            "penetration_penalty",
            "residual_4d",
            "residual_3d",
            "scaled_residual_4d",
            "scaled_residual_3d",
            "complementarity_4d",
            "complementarity_3d",
            "iterations_4d",
            "iterations_3d",
            "spin_4d",
            "spin_3d",
            "spin_penalty",
            "slip_4d",
            "slip_3d",
            "slip_penalty",
            "tilt_4d",
            "tilt_3d",
            "tilt_penalty",
            "contacts_4d",
            "contacts_3d",
            "contacts_penalty"
        },
        rows);
    writeCsv(
        dir / "benchmark_engine_oblique_box_friction_summary.csv",
        {
            "peak_penetration_4d",
            "peak_penetration_3d",
            "peak_penetration_penalty",
            "penetration_reduction_4d",
            "penetration_reduction_3d",
            "peak_residual_4d",
            "peak_residual_3d",
            "peak_scaled_residual_4d",
            "peak_scaled_residual_3d",
            "peak_complementarity_4d",
            "peak_complementarity_3d",
            "max_iterations_4d",
            "max_iterations_3d",
            "final_spin_4d",
            "final_spin_3d",
            "final_spin_penalty",
            "final_slip_4d",
            "final_slip_3d",
            "final_slip_penalty",
            "final_tilt_4d",
            "final_tilt_3d",
            "final_tilt_penalty",
            "best_penalty_stiffness",
            "best_penalty_damping",
            "best_penalty_score"
        },
        {vectorToRow({
            peak_penetration_4d,
            peak_penetration_3d,
            peak_penetration_penalty,
            (peak_penetration_penalty - peak_penetration_4d) / std::max(peak_penetration_penalty, 1e-12),
            (peak_penetration_penalty - peak_penetration_3d) / std::max(peak_penetration_penalty, 1e-12),
            peak_residual_4d,
            peak_residual_3d,
            peak_scaled_residual_4d,
            peak_scaled_residual_3d,
            peak_complementarity_4d,
            peak_complementarity_3d,
            max_iterations_4d,
            max_iterations_3d,
            spinRateY(*body_4d_final),
            spinRateY(*body_3d_final),
            spinRateY(body_penalty_final),
            horizontalSlip(*body_4d_final, initial_xz),
            horizontalSlip(*body_3d_final, initial_xz),
            horizontalSlip(body_penalty_final, initial_xz),
            tiltAngleDegrees(*body_4d_final),
            tiltAngleDegrees(*body_3d_final),
            tiltAngleDegrees(body_penalty_final),
            best_record.stiffness,
            best_record.damping,
            best_record.score
        })});
    writeCsv(
        dir / "benchmark_engine_oblique_box_friction_penalty_tuning.csv",
        {
            "stiffness",
            "damping",
            "peak_penetration",
            "final_spin",
            "final_slip",
            "final_tilt",
            "score"
        },
        tuning_rows);

    std::cout << "benchmark_engine_oblique_box_friction.csv written to "
              << (dir / "benchmark_engine_oblique_box_friction.csv") << '\n';
    std::cout << "best_penalty_stiffness=" << best_record.stiffness
              << " best_penalty_damping=" << best_record.damping << '\n';
    std::cout << "final_spin_4d=" << spinRateY(*body_4d_final)
              << " final_spin_3d=" << spinRateY(*body_3d_final)
              << " final_spin_penalty=" << spinRateY(body_penalty_final) << '\n';
    return 0;
}
