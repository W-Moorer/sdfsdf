#include "BenchmarkCommon.h"
#include "solver/SOCCPSolver.h"
#include <cmath>
#include <iostream>

namespace {

using namespace vde;
using namespace vde::benchmarks;

RigidBodyProperties cylinderProperties(double mass, double radius, double height) {
    RigidBodyProperties props;
    props.mass = mass;
    const double i_axis = 0.5 * mass * radius * radius;
    const double i_radial = (mass / 12.0) * (3.0 * radius * radius + height * height);
    props.inertia_local.setZero();
    props.inertia_local(0, 0) = i_radial;
    props.inertia_local(1, 1) = i_axis;
    props.inertia_local(2, 2) = i_radial;
    return props;
}

double firstBelowThreshold(const std::vector<double>& time,
                           const std::vector<double>& values,
                           double threshold) {
    for (size_t i = 0; i < time.size(); ++i) {
        if (values[i] <= threshold) {
            return time[i];
        }
    }
    return time.empty() ? 0.0 : time.back();
}

double stoppedSpinAngle(double omega0, double alpha, double time) {
    const double stop_time = omega0 / std::max(alpha, 1e-12);
    if (time <= stop_time) {
        return omega0 * time - 0.5 * alpha * time * time;
    }
    return omega0 * stop_time - 0.5 * alpha * stop_time * stop_time;
}

SOCCPProblem makeProblem(const Eigen::Matrix<double, 6, 6>& mass_matrix,
                         const Eigen::Matrix<double, 6, 1>& free_velocity,
                         double friction_coefficient,
                         double dt,
                         double torsion_radius,
                         bool torsion_enabled) {
    SOCCPProblem problem;
    problem.mass_matrix = mass_matrix;
    problem.free_velocity = free_velocity;
    problem.normal_jacobian = Eigen::MatrixXd::Zero(1, 6);
    problem.normal_jacobian(0, 1) = 1.0;
    problem.tangential_jacobian = Eigen::MatrixXd::Zero(3, 6);
    problem.tangential_jacobian(2, 4) = torsion_radius;
    problem.g_eq = Eigen::VectorXd::Zero(1);
    problem.friction_coefficients = Eigen::VectorXd::Constant(1, friction_coefficient);
    problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    problem.torsion_enabled = Eigen::ArrayXi::Constant(1, torsion_enabled ? 1 : 0);
    problem.time_step = dt;
    problem.epsilon = 1e-8;
    return problem;
}

}  // namespace

int main() {
    constexpr double radius = 0.45;
    constexpr double height = 0.24;
    constexpr double mass = 3.0;
    constexpr double friction = 0.35;
    constexpr double gravity = 9.81;
    constexpr double omega0 = 10.0;
    constexpr double omega_threshold = 0.5;
    constexpr double torsion_radius = 0.30;
    constexpr double dt = 0.01;
    constexpr double duration = 1.2;

    const RigidBodyProperties props = cylinderProperties(mass, radius, height);
    Eigen::Matrix<double, 6, 6> mass_matrix = Eigen::Matrix<double, 6, 6>::Zero();
    mass_matrix.diagonal() << mass, mass, mass,
                               props.inertia_local(0, 0),
                               props.inertia_local(1, 1),
                               props.inertia_local(2, 2);
    Eigen::Matrix<double, 6, 1> velocity_4d = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> velocity_3d = Eigen::Matrix<double, 6, 1>::Zero();
    velocity_4d(4) = omega0;
    velocity_3d(4) = omega0;

    SOCCPSolver solver(60, 1e-10, 1e-8);
    Eigen::VectorXd warm_start_4d;
    Eigen::VectorXd warm_start_3d;

    const double alpha_ref = friction * mass * gravity * torsion_radius / props.inertia_local(1, 1);
    const double stop_time_ref = omega0 / std::max(alpha_ref, 1e-12);
    const double lambda_n_ref = mass * gravity;
    const double lambda_tau_ref = friction * lambda_n_ref;
    const int num_steps = static_cast<int>(duration / dt);

    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);
    std::vector<double> time_history;
    std::vector<double> omega_ref_history;
    std::vector<double> omega_4d_history;
    std::vector<double> omega_3d_history;

    double theta_4d = 0.0;
    double theta_3d = 0.0;
    double max_omega_error_4d = 0.0;
    double max_theta_error_4d = 0.0;
    double max_lambda_n_error = 0.0;
    double max_lambda_tau_error_active = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        const double time = (step + 1) * dt;
        Eigen::Matrix<double, 6, 1> free_velocity_4d = velocity_4d;
        Eigen::Matrix<double, 6, 1> free_velocity_3d = velocity_3d;
        free_velocity_4d(1) -= gravity * dt;
        free_velocity_3d(1) -= gravity * dt;

        SOCCPSolution solution_4d;
        SOCCPSolution solution_3d;
        const SOCCPProblem problem_4d = makeProblem(
            mass_matrix, free_velocity_4d, friction, dt, torsion_radius, true);
        const SOCCPProblem problem_3d = makeProblem(
            mass_matrix, free_velocity_3d, friction, dt, torsion_radius, false);

        const Eigen::VectorXd* initial_4d =
            (warm_start_4d.size() == problem_4d.stateSize()) ? &warm_start_4d : nullptr;
        const Eigen::VectorXd* initial_3d =
            (warm_start_3d.size() == problem_3d.stateSize()) ? &warm_start_3d : nullptr;

        solver.solve(problem_4d, initial_4d, solution_4d);
        solver.solve(problem_3d, initial_3d, solution_3d);

        warm_start_4d = solution_4d.state;
        warm_start_3d = solution_3d.state;
        velocity_4d = solution_4d.velocity;
        velocity_3d = solution_3d.velocity;

        const double omega_ref = std::max(0.0, omega0 - alpha_ref * time);
        const double theta_ref = stoppedSpinAngle(omega0, alpha_ref, time);
        const double omega_4d = std::max(0.0, velocity_4d(4));
        const double omega_3d = std::max(0.0, velocity_3d(4));
        theta_4d += omega_4d * dt;
        theta_3d += omega_3d * dt;

        const double lambda_n_4d = solution_4d.lambda_n(0);
        const double lambda_tau_4d = std::abs(solution_4d.lambda_t(2));

        time_history.push_back(time);
        omega_ref_history.push_back(omega_ref);
        omega_4d_history.push_back(omega_4d);
        omega_3d_history.push_back(omega_3d);

        max_omega_error_4d = std::max(max_omega_error_4d, std::abs(omega_4d - omega_ref));
        max_theta_error_4d = std::max(max_theta_error_4d, std::abs(theta_4d - theta_ref));
        max_lambda_n_error = std::max(max_lambda_n_error, std::abs(lambda_n_4d - lambda_n_ref));
        if (omega_ref > 1e-8) {
            max_lambda_tau_error_active = std::max(
                max_lambda_tau_error_active,
                std::abs(lambda_tau_4d - lambda_tau_ref));
        }

        rows.push_back(vectorToRow({
            time,
            omega_ref,
            omega_4d,
            omega_3d,
            theta_ref,
            theta_4d,
            theta_3d,
            lambda_n_ref,
            lambda_n_4d,
            lambda_tau_ref,
            lambda_tau_4d
        }));
    }

    const double stop_time_4d = firstBelowThreshold(time_history, omega_4d_history, omega_threshold);
    const double stop_time_3d = firstBelowThreshold(time_history, omega_3d_history, omega_threshold);

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_torsion_friction_reference.csv",
        {
            "time",
            "omega_ref",
            "omega_4d",
            "omega_3d",
            "theta_ref",
            "theta_4d",
            "theta_3d",
            "lambda_n_ref",
            "lambda_n_4d",
            "lambda_tau_ref",
            "lambda_tau_4d"
        },
        rows);
    writeCsv(
        dir / "benchmark_torsion_friction_reference_summary.csv",
        {
            "torsion_radius",
            "alpha_ref",
            "stop_time_ref",
            "stop_time_4d",
            "stop_time_3d",
            "relative_stop_error_4d",
            "max_omega_error_4d",
            "max_theta_error_4d",
            "max_lambda_n_error",
            "max_lambda_tau_error_active",
            "final_omega_4d",
            "final_omega_3d"
        },
        {vectorToRow({
            torsion_radius,
            alpha_ref,
            stop_time_ref,
            stop_time_4d,
            stop_time_3d,
            std::abs(stop_time_4d - stop_time_ref) / std::max(stop_time_ref, 1e-12),
            max_omega_error_4d,
            max_theta_error_4d,
            max_lambda_n_error,
            max_lambda_tau_error_active,
            omega_4d_history.back(),
            omega_3d_history.back()
        })});

    std::cout << "benchmark_torsion_friction_reference.csv written to "
              << (dir / "benchmark_torsion_friction_reference.csv") << '\n';
    std::cout << "stop_time_ref=" << stop_time_ref
              << " stop_time_4d=" << stop_time_4d
              << " max_omega_error_4d=" << max_omega_error_4d
              << " max_lambda_tau_error_active=" << max_lambda_tau_error_active << '\n';
    return 0;
}
