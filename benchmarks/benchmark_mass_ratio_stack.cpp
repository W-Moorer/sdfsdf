#include "BenchmarkCommon.h"
#include "dynamics/ContactDynamics.h"
#include "solver/SOCCPSolver.h"
#include <iostream>

namespace {

using namespace vde;
using namespace vde::benchmarks;

ContactConstraint makeTwoBodyConstraint(const Eigen::Vector3d& contact_point,
                                        const Eigen::Vector3d& normal,
                                        const Eigen::Vector3d& center_a,
                                        const Eigen::Vector3d& center_b,
                                        double gap) {
    ContactConstraint constraint;
    constraint.contact.position = contact_point;
    constraint.contact.normal = normal;
    constraint.contact.computeTangentDirections();
    constraint.g_eq = gap;
    constraint.r_tau = 0.0;
    constraint.r_tau_hat = 1e-4;
    constraint.computeJacobians(center_a, center_b);
    return constraint;
}

}  // namespace

int main() {
    constexpr double dt = 0.05;
    constexpr double gap = -0.01;
    constexpr double gravity = 9.81;
    constexpr double light_radius = 0.25;
    constexpr double heavy_radius = 0.35;

    const Eigen::Vector3d center_heavy(0.0, light_radius + heavy_radius + gap, 0.0);
    const Eigen::Vector3d center_light(0.0, light_radius + gap, 0.0);
    const Eigen::Vector3d normal = Eigen::Vector3d::UnitY();

    const ContactConstraint upper_lower = makeTwoBodyConstraint(
        Eigen::Vector3d(0.0, center_light.y() + light_radius + 0.5 * gap, 0.0),
        normal,
        center_heavy,
        center_light,
        gap);

    Eigen::RowVectorXd lower_ground_normal = Eigen::RowVectorXd::Zero(12);
    lower_ground_normal.segment<6>(6) =
        ContactJacobian::computeNormal(
            Eigen::Vector3d(0.0, 0.0, 0.0),
            center_light,
            normal);

    Eigen::MatrixXd lower_ground_tangent = Eigen::MatrixXd::Zero(3, 12);
    const Eigen::Matrix<double, 2, 6> light_tangent =
        ContactJacobian::computeTangential(
            Eigen::Vector3d(0.0, 0.0, 0.0),
            center_light,
            Eigen::Vector3d::UnitX(),
            Eigen::Vector3d::UnitZ());
    lower_ground_tangent.block<2, 6>(0, 6) = light_tangent;

    std::vector<std::vector<std::string>> rows;
    rows.reserve(4);

    for (double heavy_mass : {1.0, 10.0, 100.0, 1000.0}) {
        const RigidBodyProperties heavy_props = RigidBodyProperties::sphere(heavy_mass, heavy_radius);
        const RigidBodyProperties light_props = RigidBodyProperties::sphere(1.0, light_radius);

        Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Zero(12, 12);
        mass_matrix.block<6, 6>(0, 0) = RigidBody(heavy_props).massMatrix();
        mass_matrix.block<6, 6>(6, 6) = RigidBody(light_props).massMatrix();

        Eigen::VectorXd free_velocity = Eigen::VectorXd::Zero(12);
        free_velocity(1) = -gravity * dt;
        free_velocity(7) = -gravity * dt;

        SOCCPProblem problem;
        problem.mass_matrix = mass_matrix;
        problem.free_velocity = free_velocity;
        problem.normal_jacobian = Eigen::MatrixXd::Zero(2, 12);
        problem.tangential_jacobian = Eigen::MatrixXd::Zero(6, 12);
        problem.g_eq = Eigen::Vector2d::Constant(gap);
        problem.friction_coefficients = Eigen::Vector2d::Zero();
        problem.baumgarte_gamma = Eigen::Vector2d::Constant(0.8);
        problem.time_step = dt;
        problem.epsilon = 1e-8;

        problem.normal_jacobian.row(0) = upper_lower.J_n;
        problem.tangential_jacobian.block(0, 0, 3, 12) = upper_lower.tangentialJacobian();
        problem.normal_jacobian.row(1) = lower_ground_normal;
        problem.tangential_jacobian.block(3, 0, 3, 12) = lower_ground_tangent;

        SOCCPSolver solver(120, 1e-10, problem.epsilon);
        SOCCPSolution solution;
        const bool converged = solver.solve(problem, solution);

        const double heavy_speed = solution.velocity.segment<3>(0).norm();
        const double light_speed = solution.velocity.segment<3>(6).norm();
        const double residual =
            solution.residual.size() > 0
                ? solution.residual.lpNorm<Eigen::Infinity>()
                : 0.0;

        rows.push_back(vectorToRow({
            heavy_mass,
            heavy_mass / 1.0,
            converged ? 1.0 : 0.0,
            static_cast<double>(solution.stats.iterations),
            residual,
            solution.scaled_residual,
            solution.complementarity_violation,
            heavy_speed,
            light_speed
        }));
    }

    double max_iterations = 0.0;
    double max_residual = 0.0;
    double max_scaled_residual = 0.0;
    double max_complementarity = 0.0;
    double min_converged = 1.0;
    for (const auto& row : rows) {
        max_iterations = std::max(max_iterations, std::stod(row[3]));
        max_residual = std::max(max_residual, std::stod(row[4]));
        max_scaled_residual = std::max(max_scaled_residual, std::stod(row[5]));
        max_complementarity = std::max(max_complementarity, std::stod(row[6]));
        min_converged = std::min(min_converged, std::stod(row[2]));
    }

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_mass_ratio_stack.csv",
        {
            "heavy_mass",
            "mass_ratio",
            "converged",
            "iterations",
            "residual_inf",
            "scaled_residual",
            "complementarity_violation",
            "heavy_speed",
            "light_speed"
        },
        rows);
    writeCsv(
        dir / "benchmark_mass_ratio_stack_summary.csv",
        {
            "max_mass_ratio",
            "min_converged",
            "max_iterations",
            "max_residual_inf",
            "max_scaled_residual",
            "max_complementarity_violation"
        },
        {vectorToRow({
            1000.0,
            min_converged,
            max_iterations,
            max_residual,
            max_scaled_residual,
            max_complementarity
        })});

    std::cout << "benchmark_mass_ratio_stack.csv written to "
              << (dir / "benchmark_mass_ratio_stack.csv") << '\n';
    for (const auto& row : rows) {
        std::cout << "mass_ratio=" << row[1]
                  << " converged=" << row[2]
                  << " iterations=" << row[3]
                  << " residual=" << row[4] << '\n';
    }
    return 0;
}
