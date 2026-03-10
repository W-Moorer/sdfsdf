#include "BenchmarkCommon.h"
#include "dynamics/ContactDynamics.h"
#include "geometry/AnalyticalSDF.h"
#include "geometry/VolumetricIntegrator.h"
#include "solver/NormalLCPSolver.h"
#include "solver/PenaltySolver.h"
#include "solver/SOCCPSolver.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

struct SimpleScene {
    std::shared_ptr<RigidBody> sphere;
    std::shared_ptr<RigidBody> support;
    std::shared_ptr<SDF> sphere_sdf;
    std::shared_ptr<SDF> support_sdf;
    AABB sphere_local_aabb;
    AABB support_local_aabb;
};

struct StepMetrics {
    double height = 0.0;
    double vy = 0.0;
    double orientation_angle_deg = 0.0;
    double angular_speed = 0.0;
    double penetration = 0.0;
    double equivalent_gap = 0.0;
    double normal_response = 0.0;
    double residual = 0.0;
    double contact_active = 0.0;
};

struct ContactSample {
    bool active = false;
    ContactGeometry geometry;
    ContactConstraint constraint;
    Eigen::Matrix<double, 1, 6> reduced_normal = Eigen::Matrix<double, 1, 6>::Zero();
    SpatialVector free_velocity = SpatialVector::Zero();
};

struct DropConfig {
    double radius = 0.20;
    double sphere_mass = 1.0;
    double support_width = 12.0;
    double support_height = 2.0;
    double support_depth = 12.0;
    double initial_height = 2.0;
    double duration = 1.20;
    double dt = 0.0025;
    double baumgarte_gamma = 0.0;
    double torsion_regularization = 1e-4;
    int contact_grid_resolution = 56;
    double contact_p_norm = 2.0;
    int max_solver_iterations = 160;
    double solver_tolerance = 1e-8;
    Eigen::Vector3d gravity = Eigen::Vector3d(0.0, -9.81, 0.0);
};

RigidBodyProperties makeSupportProperties() {
    return RigidBodyProperties::box(1.0, 1.0, 1.0, 1.0);
}

AABB transformLocalAABB(const AABB& local_aabb, const RigidBodyState& state) {
    const Eigen::Vector3d& min = local_aabb.min;
    const Eigen::Vector3d& max = local_aabb.max;

    const std::array<Eigen::Vector3d, 8> local_corners = {{
        Eigen::Vector3d(min.x(), min.y(), min.z()),
        Eigen::Vector3d(min.x(), min.y(), max.z()),
        Eigen::Vector3d(min.x(), max.y(), min.z()),
        Eigen::Vector3d(min.x(), max.y(), max.z()),
        Eigen::Vector3d(max.x(), min.y(), min.z()),
        Eigen::Vector3d(max.x(), min.y(), max.z()),
        Eigen::Vector3d(max.x(), max.y(), min.z()),
        Eigen::Vector3d(max.x(), max.y(), max.z())
    }};

    AABB world_aabb;
    for (const auto& corner : local_corners) {
        world_aabb.expand(state.localToWorld(corner));
    }
    return world_aabb;
}

SimpleScene makeScene(const DropConfig& config) {
    SimpleScene scene;
    scene.sphere = std::make_shared<RigidBody>(
        RigidBodyProperties::sphere(config.sphere_mass, config.radius));
    scene.sphere->setPosition(Eigen::Vector3d(0.0, config.initial_height, 0.0));

    scene.support = std::make_shared<RigidBody>(makeSupportProperties());
    scene.support->setStatic(true);
    scene.support->setPosition(Eigen::Vector3d(0.0, -0.5 * config.support_height, 0.0));

    scene.sphere_sdf = std::make_shared<SphereSDF>(Eigen::Vector3d::Zero(), config.radius);
    scene.support_sdf = std::make_shared<BoxSDF>(
        Eigen::Vector3d(-0.5 * config.support_width, -0.5 * config.support_height, -0.5 * config.support_depth),
        Eigen::Vector3d(0.5 * config.support_width, 0.5 * config.support_height, 0.5 * config.support_depth));

    scene.sphere_local_aabb = AABB(
        Eigen::Vector3d::Constant(-config.radius),
        Eigen::Vector3d::Constant(config.radius));
    scene.support_local_aabb = AABB(
        Eigen::Vector3d(-0.5 * config.support_width, -0.5 * config.support_height, -0.5 * config.support_depth),
        Eigen::Vector3d(0.5 * config.support_width, 0.5 * config.support_height, 0.5 * config.support_depth));
    return scene;
}

double orientationAngleDeg(const Eigen::Quaterniond& q) {
    const double clamped_w = std::clamp(std::abs(q.normalized().w()), 0.0, 1.0);
    return 2.0 * std::acos(clamped_w) * 180.0 / 3.14159265358979323846;
}

StepMetrics captureState(const SimpleScene& scene) {
    StepMetrics metrics;
    metrics.height = scene.sphere->position().y();
    metrics.vy = scene.sphere->linearVelocity().y();
    metrics.orientation_angle_deg = orientationAngleDeg(scene.sphere->state().orientation);
    metrics.angular_speed = scene.sphere->angularVelocity().norm();
    metrics.equivalent_gap = 0.0;
    metrics.penetration = 0.0;
    metrics.normal_response = 0.0;
    metrics.residual = 0.0;
    metrics.contact_active = 0.0;
    return metrics;
}

ContactSample sampleContact(const SimpleScene& scene,
                            const DropConfig& config,
                            const VolumetricIntegrator& integrator) {
    ContactSample sample;
    sample.free_velocity = scene.sphere->spatialVelocity();
    sample.free_velocity.head<3>() += config.gravity * config.dt;

    const AABB sphere_world_aabb = transformLocalAABB(scene.sphere_local_aabb, scene.sphere->state());
    const AABB support_world_aabb = transformLocalAABB(scene.support_local_aabb, scene.support->state());
    if (!sphere_world_aabb.intersects(support_world_aabb)) {
        return sample;
    }

    const TransformedSDF sphere_world(
        scene.sphere_sdf, scene.sphere->state().position, scene.sphere->state().orientation);
    const TransformedSDF support_world(
        scene.support_sdf, scene.support->state().position, scene.support->state().orientation);

    sample.geometry = integrator.computeContactGeometry(
        sphere_world,
        support_world,
        sphere_world_aabb,
        support_world_aabb);

    if (sample.geometry.g_eq >= 0.0 || sample.geometry.volume <= 0.0) {
        return sample;
    }

    sample.active = true;
    sample.constraint.computeJacobiansFromGeometry(
        sample.geometry,
        scene.sphere->centerOfMassWorld(),
        scene.support->centerOfMassWorld(),
        config.torsion_regularization);
    sample.reduced_normal = sample.constraint.J_n.block<1, 6>(0, 0);
    return sample;
}

StepMetrics finalizeImpulseStep(SimpleScene& scene,
                                const SpatialVector& solved_velocity,
                                const ContactSample& sample,
                                double normal_response,
                                double residual,
                                double dt) {
    scene.sphere->setSpatialVelocity(solved_velocity);
    scene.sphere->clearForces();
    scene.support->clearForces();
    scene.sphere->integrate(dt, Eigen::Vector3d::Zero());
    scene.support->integrate(dt, Eigen::Vector3d::Zero());

    StepMetrics metrics = captureState(scene);
    metrics.contact_active = sample.active ? 1.0 : 0.0;
    metrics.equivalent_gap = sample.geometry.g_eq;
    metrics.penetration = sample.active ? std::max(0.0, -sample.geometry.g_eq) : 0.0;
    metrics.normal_response = normal_response;
    metrics.residual = residual;
    return metrics;
}

StepMetrics runPenaltyStep(SimpleScene& scene,
                           const DropConfig& config,
                           VolumetricIntegrator& integrator,
                           PenaltySolver& penalty_solver) {
    const ContactSample sample = sampleContact(scene, config, integrator);
    PenaltyContactResult penalty_result;
    if (sample.active) {
        penalty_result = penalty_solver.solveContactDetailed(
            *scene.sphere,
            *scene.support,
            sample.constraint,
            config.dt);
    }

    scene.sphere->integrate(config.dt, config.gravity);
    scene.support->integrate(config.dt, Eigen::Vector3d::Zero());

    StepMetrics metrics = captureState(scene);
    metrics.contact_active = sample.active ? 1.0 : 0.0;
    metrics.equivalent_gap = sample.geometry.g_eq;
    metrics.penetration = sample.active ? std::max(0.0, -sample.geometry.g_eq) : 0.0;
    metrics.normal_response = penalty_result.normal_force;
    return metrics;
}

StepMetrics runNormalLcpStep(SimpleScene& scene,
                             const DropConfig& config,
                             VolumetricIntegrator& integrator,
                             NormalLCPSolver& solver,
                             Eigen::VectorXd& warm_start) {
    const ContactSample sample = sampleContact(scene, config, integrator);
    SpatialVector solved_velocity = sample.free_velocity;
    double lambda = 0.0;
    double residual = 0.0;

    if (sample.active) {
        NormalLCPProblem problem;
        problem.mass_matrix = scene.sphere->massMatrix();
        problem.free_velocity = sample.free_velocity;
        problem.normal_jacobian = sample.reduced_normal;
        problem.g_eq = Eigen::VectorXd::Constant(1, sample.geometry.g_eq);
        problem.baumgarte_gamma = Eigen::VectorXd::Constant(1, config.baumgarte_gamma);
        problem.time_step = config.dt;

        NormalLCPSolution solution;
        const Eigen::VectorXd* initial_lambda =
            (warm_start.size() == problem.numContacts()) ? &warm_start : nullptr;
        solver.solve(problem, initial_lambda, solution);
        if (solution.lambda_n.size() == problem.numContacts() && solution.lambda_n.allFinite()) {
            warm_start = solution.lambda_n;
        } else {
            warm_start.resize(0);
        }
        if (solution.velocity.size() == problem.velocityDofs() && solution.velocity.allFinite()) {
            solved_velocity = solution.velocity;
        }
        if (solution.lambda_n.size() == 1) {
            lambda = solution.lambda_n(0);
        }
        if (solution.residual.size() > 0) {
            residual = solution.residual.lpNorm<Eigen::Infinity>();
        }
    } else {
        warm_start.resize(0);
    }

    return finalizeImpulseStep(
        scene,
        solved_velocity,
        sample,
        lambda,
        residual,
        config.dt);
}

StepMetrics runSoccpStep(SimpleScene& scene,
                         const DropConfig& config,
                         VolumetricIntegrator& integrator,
                         SOCCPSolver& solver,
                         Eigen::VectorXd& warm_start) {
    const ContactSample sample = sampleContact(scene, config, integrator);
    SpatialVector solved_velocity = sample.free_velocity;
    double lambda = 0.0;
    double residual = 0.0;

    if (sample.active) {
        SOCCPProblem problem;
        problem.mass_matrix = scene.sphere->massMatrix();
        problem.free_velocity = sample.free_velocity;
        problem.normal_jacobian = sample.reduced_normal;
        problem.tangential_jacobian = Eigen::MatrixXd::Zero(3, 6);
        problem.g_eq = Eigen::VectorXd::Constant(1, sample.geometry.g_eq);
        problem.friction_coefficients = Eigen::VectorXd::Zero(1);
        problem.baumgarte_gamma = Eigen::VectorXd::Constant(1, config.baumgarte_gamma);
        problem.torsion_enabled = Eigen::ArrayXi::Zero(1);
        problem.time_step = config.dt;
        problem.epsilon = 1e-8;

        SOCCPSolution solution;
        const Eigen::VectorXd* initial_state =
            (warm_start.size() == problem.stateSize()) ? &warm_start : nullptr;
        solver.solve(problem, initial_state, solution);
        if (solution.state.size() == problem.stateSize() && solution.state.allFinite()) {
            warm_start = solution.state;
        } else {
            warm_start.resize(0);
        }
        if (solution.velocity.size() == problem.velocityDofs() && solution.velocity.allFinite()) {
            solved_velocity = solution.velocity;
        }
        if (solution.lambda_n.size() == 1) {
            lambda = solution.lambda_n(0);
        }
        if (solution.residual.size() > 0) {
            residual = solution.residual.lpNorm<Eigen::Infinity>();
        }
    } else {
        warm_start.resize(0);
    }

    return finalizeImpulseStep(
        scene,
        solved_velocity,
        sample,
        lambda,
        residual,
        config.dt);
}

double firstContactTime(const std::vector<StepMetrics>& metrics,
                        double dt,
                        double duration) {
    for (size_t i = 0; i < metrics.size(); ++i) {
        if (metrics[i].contact_active > 0.5) {
            return (static_cast<double>(i) + 1.0) * dt;
        }
    }
    return duration;
}

double maxAbsDifference(const std::vector<double>& a,
                        const std::vector<double>& b,
                        size_t last_index) {
    const size_t count = std::min({a.size(), b.size(), last_index});
    double max_diff = 0.0;
    for (size_t i = 0; i < count; ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

double maxValue(const std::vector<StepMetrics>& metrics,
                double StepMetrics::* member) {
    double value = 0.0;
    for (const auto& item : metrics) {
        value = std::max(value, item.*member);
    }
    return value;
}

}  // namespace

int main() {
    const DropConfig config;
    VolumetricIntegrator integrator(config.contact_grid_resolution, config.contact_p_norm);

    PenaltyParameters penalty_params;
    penalty_params.stiffness = 60000.0;
    penalty_params.damping = 220.0;
    penalty_params.friction_coefficient = 0.0;
    PenaltySolver penalty_solver(penalty_params);

    NormalLCPSolver normal_lcp_solver(
        config.max_solver_iterations,
        config.solver_tolerance,
        1.0);
    SOCCPSolver soccp_solver(
        config.max_solver_iterations,
        config.solver_tolerance,
        1e-8);

    SimpleScene soccp_scene = makeScene(config);
    SimpleScene normal_lcp_scene = makeScene(config);
    SimpleScene penalty_scene = makeScene(config);
    Eigen::VectorXd normal_lcp_warm_start;
    Eigen::VectorXd soccp_warm_start;

    const int num_steps = static_cast<int>(std::round(config.duration / config.dt));
    std::vector<StepMetrics> soccp_history;
    std::vector<StepMetrics> normal_lcp_history;
    std::vector<StepMetrics> penalty_history;
    soccp_history.reserve(num_steps);
    normal_lcp_history.reserve(num_steps);
    penalty_history.reserve(num_steps);

    std::vector<double> reference_height;
    std::vector<double> reference_vy;
    reference_height.reserve(num_steps);
    reference_vy.reserve(num_steps);

    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);

    double y_ref = config.initial_height;
    double vy_ref = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        const StepMetrics soccp = runSoccpStep(
            soccp_scene, config, integrator, soccp_solver, soccp_warm_start);
        const StepMetrics normal_lcp = runNormalLcpStep(
            normal_lcp_scene, config, integrator, normal_lcp_solver, normal_lcp_warm_start);
        const StepMetrics penalty = runPenaltyStep(
            penalty_scene, config, integrator, penalty_solver);

        vy_ref += config.gravity.y() * config.dt;
        y_ref += vy_ref * config.dt;

        soccp_history.push_back(soccp);
        normal_lcp_history.push_back(normal_lcp);
        penalty_history.push_back(penalty);
        reference_height.push_back(y_ref);
        reference_vy.push_back(vy_ref);

        const double time = (static_cast<double>(step) + 1.0) * config.dt;
        rows.push_back(vectorToRow({
            time,
            y_ref,
            vy_ref,
            soccp.height,
            normal_lcp.height,
            penalty.height,
            soccp.vy,
            normal_lcp.vy,
            penalty.vy,
            soccp.orientation_angle_deg,
            normal_lcp.orientation_angle_deg,
            penalty.orientation_angle_deg,
            soccp.angular_speed,
            normal_lcp.angular_speed,
            penalty.angular_speed,
            soccp.penetration,
            normal_lcp.penetration,
            penalty.penetration,
            soccp.normal_response,
            normal_lcp.normal_response,
            penalty.normal_response,
            soccp.residual,
            normal_lcp.residual,
            soccp.contact_active,
            normal_lcp.contact_active,
            penalty.contact_active
        }));
    }

    const double soccp_contact_time = firstContactTime(soccp_history, config.dt, config.duration);
    const double normal_lcp_contact_time =
        firstContactTime(normal_lcp_history, config.dt, config.duration);
    const double penalty_contact_time = firstContactTime(penalty_history, config.dt, config.duration);
    const double precontact_end_time =
        std::min({soccp_contact_time, normal_lcp_contact_time, penalty_contact_time});
    const size_t precontact_steps =
        std::min(static_cast<size_t>(num_steps), static_cast<size_t>(std::floor(precontact_end_time / config.dt)));

    std::vector<double> soccp_height;
    std::vector<double> normal_lcp_height;
    std::vector<double> penalty_height;
    std::vector<double> soccp_vy;
    std::vector<double> normal_lcp_vy;
    std::vector<double> penalty_vy;
    soccp_height.reserve(num_steps);
    normal_lcp_height.reserve(num_steps);
    penalty_height.reserve(num_steps);
    soccp_vy.reserve(num_steps);
    normal_lcp_vy.reserve(num_steps);
    penalty_vy.reserve(num_steps);

    for (int i = 0; i < num_steps; ++i) {
        soccp_height.push_back(soccp_history[i].height);
        normal_lcp_height.push_back(normal_lcp_history[i].height);
        penalty_height.push_back(penalty_history[i].height);
        soccp_vy.push_back(soccp_history[i].vy);
        normal_lcp_vy.push_back(normal_lcp_history[i].vy);
        penalty_vy.push_back(penalty_history[i].vy);
    }

    const double precontact_height_error_soccp =
        maxAbsDifference(reference_height, soccp_height, precontact_steps);
    const double precontact_height_error_normal_lcp =
        maxAbsDifference(reference_height, normal_lcp_height, precontact_steps);
    const double precontact_height_error_penalty =
        maxAbsDifference(reference_height, penalty_height, precontact_steps);
    const double precontact_velocity_error_soccp =
        maxAbsDifference(reference_vy, soccp_vy, precontact_steps);
    const double precontact_velocity_error_normal_lcp =
        maxAbsDifference(reference_vy, normal_lcp_vy, precontact_steps);
    const double precontact_velocity_error_penalty =
        maxAbsDifference(reference_vy, penalty_vy, precontact_steps);

    const double max_height_diff_soccp_normal_lcp =
        maxAbsDifference(soccp_height, normal_lcp_height, soccp_height.size());
    const double max_velocity_diff_soccp_normal_lcp =
        maxAbsDifference(soccp_vy, normal_lcp_vy, soccp_vy.size());

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_sphere_drop_compare.csv",
        {
            "time",
            "reference_height",
            "reference_vy",
            "soccp_height",
            "normal_lcp_height",
            "penalty_height",
            "soccp_vy",
            "normal_lcp_vy",
            "penalty_vy",
            "soccp_orientation_angle_deg",
            "normal_lcp_orientation_angle_deg",
            "penalty_orientation_angle_deg",
            "soccp_angular_speed",
            "normal_lcp_angular_speed",
            "penalty_angular_speed",
            "soccp_penetration",
            "normal_lcp_penetration",
            "penalty_penetration",
            "soccp_lambda_n",
            "normal_lcp_lambda_n",
            "penalty_normal_force",
            "soccp_residual",
            "normal_lcp_residual",
            "soccp_contact_active",
            "normal_lcp_contact_active",
            "penalty_contact_active"
        },
        rows);
    writeCsv(
        dir / "benchmark_sphere_drop_compare_summary.csv",
        {
            "soccp_contact_time",
            "normal_lcp_contact_time",
            "penalty_contact_time",
            "precontact_height_error_soccp",
            "precontact_height_error_normal_lcp",
            "precontact_height_error_penalty",
            "precontact_velocity_error_soccp",
            "precontact_velocity_error_normal_lcp",
            "precontact_velocity_error_penalty",
            "max_height_diff_soccp_normal_lcp",
            "max_velocity_diff_soccp_normal_lcp",
            "peak_penetration_soccp",
            "peak_penetration_normal_lcp",
            "peak_penetration_penalty",
            "max_orientation_angle_soccp",
            "max_orientation_angle_normal_lcp",
            "max_orientation_angle_penalty",
            "max_angular_speed_soccp",
            "max_angular_speed_normal_lcp",
            "max_angular_speed_penalty",
            "max_residual_soccp",
            "max_residual_normal_lcp",
            "peak_penalty_force"
        },
        {vectorToRow({
            soccp_contact_time,
            normal_lcp_contact_time,
            penalty_contact_time,
            precontact_height_error_soccp,
            precontact_height_error_normal_lcp,
            precontact_height_error_penalty,
            precontact_velocity_error_soccp,
            precontact_velocity_error_normal_lcp,
            precontact_velocity_error_penalty,
            max_height_diff_soccp_normal_lcp,
            max_velocity_diff_soccp_normal_lcp,
            maxValue(soccp_history, &StepMetrics::penetration),
            maxValue(normal_lcp_history, &StepMetrics::penetration),
            maxValue(penalty_history, &StepMetrics::penetration),
            maxValue(soccp_history, &StepMetrics::orientation_angle_deg),
            maxValue(normal_lcp_history, &StepMetrics::orientation_angle_deg),
            maxValue(penalty_history, &StepMetrics::orientation_angle_deg),
            maxValue(soccp_history, &StepMetrics::angular_speed),
            maxValue(normal_lcp_history, &StepMetrics::angular_speed),
            maxValue(penalty_history, &StepMetrics::angular_speed),
            maxValue(soccp_history, &StepMetrics::residual),
            maxValue(normal_lcp_history, &StepMetrics::residual),
            maxValue(penalty_history, &StepMetrics::normal_response)
        })});

    std::cout << "benchmark_sphere_drop_compare.csv written to "
              << (dir / "benchmark_sphere_drop_compare.csv") << '\n';
    std::cout << "max_height_diff_soccp_normal_lcp=" << max_height_diff_soccp_normal_lcp
              << " peak_penetration_penalty=" << maxValue(penalty_history, &StepMetrics::penetration)
              << " max_residual_soccp=" << maxValue(soccp_history, &StepMetrics::residual)
              << '\n';
    return 0;
}
