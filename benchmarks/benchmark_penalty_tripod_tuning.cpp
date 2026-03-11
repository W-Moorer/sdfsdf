#include "BenchmarkCommon.h"
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

struct SweepRecord {
    double stiffness = 0.0;
    double damping = 0.0;
    double peak_penetration = 0.0;
    double final_tilt = 0.0;
    double max_contacts = 0.0;
    double score = std::numeric_limits<double>::infinity();
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

std::vector<BodyShape> makeScene(const std::shared_ptr<SDF>& top_sdf,
                                 const AABB& top_aabb,
                                 const std::shared_ptr<SDF>& support_sdf,
                                 const AABB& support_aabb,
                                 double support_size,
                                 double support_half,
                                 double top_width,
                                 double top_height,
                                 double top_depth) {
    std::vector<BodyShape> bodies;
    bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(3.0, top_width, top_height, top_depth)),
        top_sdf,
        top_aabb
    });
    bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size)),
        support_sdf,
        support_aabb
    });
    bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size)),
        support_sdf,
        support_aabb
    });
    bodies.push_back({
        std::make_shared<RigidBody>(makeBoxProps(5.0, support_size, support_size, support_size)),
        support_sdf,
        support_aabb
    });
    bodies[0].body->setPosition(Eigen::Vector3d(0.025, 0.48, -0.015));
    bodies[0].body->setLinearVelocity(Eigen::Vector3d(0.0, -0.30, 0.0));
    bodies[0].body->state().orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(13.0 * kPi / 180.0, Eigen::Vector3d::UnitX())) *
        Eigen::Quaterniond(Eigen::AngleAxisd(-9.0 * kPi / 180.0, Eigen::Vector3d::UnitZ()));
    bodies[1].body->setPosition(Eigen::Vector3d(-0.22, support_half, -0.16));
    bodies[2].body->setPosition(Eigen::Vector3d(0.22, support_half, -0.16));
    bodies[3].body->setPosition(Eigen::Vector3d(0.00, support_half, 0.24));
    bodies[1].body->setStatic(true);
    bodies[2].body->setStatic(true);
    bodies[3].body->setStatic(true);
    return bodies;
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

    auto top_sdf = std::make_shared<BoxSDF>(-top_half_extents, top_half_extents);
    auto support_sdf = std::make_shared<BoxSDF>(
        Eigen::Vector3d::Constant(-support_half),
        Eigen::Vector3d::Constant(support_half));
    const AABB top_aabb(-top_half_extents, top_half_extents);
    const AABB support_aabb(
        Eigen::Vector3d::Constant(-support_half),
        Eigen::Vector3d::Constant(support_half));
    const Eigen::Vector3d gravity(0.0, -9.81, 0.0);

    const std::vector<double> stiffness_candidates = {20000.0, 30000.0, 45000.0, 60000.0, 80000.0, 100000.0};
    const std::vector<double> damping_candidates = {60.0, 90.0, 105.0, 120.0, 150.0, 180.0};

    std::vector<std::vector<std::string>> rows;
    rows.reserve(stiffness_candidates.size() * damping_candidates.size());
    SweepRecord best_record;

    for (double stiffness : stiffness_candidates) {
        for (double damping : damping_candidates) {
            auto bodies = makeScene(
                top_sdf, top_aabb, support_sdf, support_aabb,
                support_size, support_half, top_width, top_height, top_depth);
            VolumetricIntegrator integrator(32, 2.0);
            PenaltyParameters params;
            params.stiffness = stiffness;
            params.damping = damping;
            params.friction_coefficient = 0.0;
            PenaltySolver solver(params);

            SweepRecord record;
            record.stiffness = stiffness;
            record.damping = damping;

            const int num_steps = static_cast<int>(duration / dt);
            bool finite = true;
            for (int step = 0; step < num_steps; ++step) {
                const PenaltyStepResult result =
                    runPenaltyStep(bodies, integrator, solver, dt, gravity);
                record.peak_penetration = std::max(record.peak_penetration, result.max_penetration);
                record.max_contacts = std::max(record.max_contacts, static_cast<double>(result.active_contacts));
                record.final_tilt = tiltAngleDegrees(*bodies[0].body);
                if (!bodies[0].body->position().allFinite()) {
                    finite = false;
                    break;
                }
            }

            record.score =
                record.peak_penetration +
                5e-4 * record.final_tilt +
                (finite ? 0.0 : 100.0);
            if (record.score < best_record.score) {
                best_record = record;
            }

            rows.push_back(vectorToRow({
                record.stiffness,
                record.damping,
                record.peak_penetration,
                record.final_tilt,
                record.max_contacts,
                record.score
            }));
        }
    }

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_penalty_tripod_tuning.csv",
        {"stiffness", "damping", "peak_penetration", "final_tilt_deg", "max_contacts", "score"},
        rows);
    writeCsv(
        dir / "benchmark_penalty_tripod_tuning_summary.csv",
        {"best_stiffness", "best_damping", "best_peak_penetration", "best_final_tilt_deg", "best_score"},
        {vectorToRow({
            best_record.stiffness,
            best_record.damping,
            best_record.peak_penetration,
            best_record.final_tilt,
            best_record.score
        })});

    std::cout << "benchmark_penalty_tripod_tuning.csv written to "
              << (dir / "benchmark_penalty_tripod_tuning.csv") << '\n';
    std::cout << "best_stiffness=" << best_record.stiffness
              << " best_damping=" << best_record.damping
              << " best_peak_penetration=" << best_record.peak_penetration << '\n';
    return 0;
}
