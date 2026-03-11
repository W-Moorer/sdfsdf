#include "BenchmarkCommon.h"
#include "solver/PolyhedralFrictionSolver.h"
#include "solver/FrictionPGSSolver.h"
#include "solver/SOCCPSolver.h"
#include <cmath>
#include <iostream>

namespace {

using namespace vde;
using namespace vde::benchmarks;

struct SlidingConfig {
    double dt = 0.001;
    double duration = 0.35;
    double mass = 2.0;
    double friction_coefficient = 0.40;
    double gravity = 9.81;
    double initial_speed = 1.00;
    double stop_threshold = 0.02;
};

struct TranslationalState {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
};

double referenceVelocity(double time, const SlidingConfig& scene) {
    return std::max(scene.initial_speed - scene.friction_coefficient * scene.gravity * time, 0.0);
}

double referencePosition(double time, const SlidingConfig& scene) {
    const double stop_time = scene.initial_speed / (scene.friction_coefficient * scene.gravity);
    if (time <= stop_time) {
        return scene.initial_speed * time -
               0.5 * scene.friction_coefficient * scene.gravity * time * time;
    }
    return 0.5 * scene.initial_speed * stop_time;
}

double firstBelowThreshold(const std::vector<double>& time_history,
                           const std::vector<double>& speed_history,
                           double threshold) {
    for (size_t i = 0; i < time_history.size(); ++i) {
        if (speed_history[i] <= threshold) {
            return time_history[i];
        }
    }
    return time_history.empty() ? 0.0 : time_history.back();
}

SOCCPProblem makeSOCCPProblem(const SlidingConfig& scene,
                              const Eigen::Vector3d& current_velocity) {
    SOCCPProblem problem;
    problem.mass_matrix = scene.mass * Eigen::Matrix3d::Identity();
    problem.free_velocity = current_velocity + scene.dt * Eigen::Vector3d(0.0, -scene.gravity, 0.0);
    problem.normal_jacobian = Eigen::MatrixXd::Zero(1, 3);
    problem.normal_jacobian << 0.0, 1.0, 0.0;
    problem.tangential_jacobian = Eigen::MatrixXd::Zero(3, 3);
    problem.tangential_jacobian << 1.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0,
                                   0.0, 0.0, 0.0;
    problem.g_eq = Eigen::VectorXd::Zero(1);
    problem.friction_coefficients = Eigen::VectorXd::Constant(1, scene.friction_coefficient);
    problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    problem.torsion_enabled = Eigen::ArrayXi::Zero(1);
    problem.time_step = scene.dt;
    problem.epsilon = 1e-8;
    return problem;
}

FrictionPGSProblem makeFrictionPGSProblem(const SlidingConfig& scene,
                                          const Eigen::Vector3d& current_velocity) {
    FrictionPGSProblem problem;
    problem.mass_matrix = scene.mass * Eigen::Matrix3d::Identity();
    problem.free_velocity = current_velocity + scene.dt * Eigen::Vector3d(0.0, -scene.gravity, 0.0);
    problem.normal_jacobian = Eigen::MatrixXd::Zero(1, 3);
    problem.normal_jacobian << 0.0, 1.0, 0.0;
    problem.tangential_jacobian = Eigen::MatrixXd::Zero(2, 3);
    problem.tangential_jacobian << 1.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0;
    problem.g_eq = Eigen::VectorXd::Zero(1);
    problem.friction_coefficients = Eigen::VectorXd::Constant(1, scene.friction_coefficient);
    problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    problem.time_step = scene.dt;
    return problem;
}

PolyhedralFrictionProblem makePolyhedralProblem(const SlidingConfig& scene,
                                                const Eigen::Vector3d& current_velocity) {
    PolyhedralFrictionProblem problem;
    problem.mass_matrix = scene.mass * Eigen::Matrix3d::Identity();
    problem.free_velocity = current_velocity + scene.dt * Eigen::Vector3d(0.0, -scene.gravity, 0.0);
    problem.normal_jacobian = Eigen::MatrixXd::Zero(1, 3);
    problem.normal_jacobian << 0.0, 1.0, 0.0;
    problem.tangential_jacobian = Eigen::MatrixXd::Zero(2, 3);
    problem.tangential_jacobian << 1.0, 0.0, 0.0,
                                   0.0, 0.0, 1.0;
    problem.g_eq = Eigen::VectorXd::Zero(1);
    problem.friction_coefficients = Eigen::VectorXd::Constant(1, scene.friction_coefficient);
    problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    problem.time_step = scene.dt;
    problem.num_directions = 8;
    return problem;
}

}  // namespace

int main() {
    const SlidingConfig scene;
    const int num_steps = static_cast<int>(scene.duration / scene.dt);

    SOCCPSolver soccp_solver(80, 1e-10, 1e-8);
    FrictionPGSSolver friction_pgs_solver(80, 1e-10);
    PolyhedralFrictionSolver polyhedral_solver(80, 1e-10);

    TranslationalState soccp_state;
    TranslationalState friction_pgs_state;
    TranslationalState polyhedral_state;
    soccp_state.velocity.x() = scene.initial_speed;
    friction_pgs_state.velocity.x() = scene.initial_speed;
    polyhedral_state.velocity.x() = scene.initial_speed;

    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);
    std::vector<double> time_history;
    std::vector<double> speed_soccp_history;
    std::vector<double> speed_friction_pgs_history;
    std::vector<double> speed_polyhedral_history;

    double max_velocity_error_soccp = 0.0;
    double max_velocity_error_friction_pgs = 0.0;
    double max_velocity_error_polyhedral = 0.0;
    double max_position_error_soccp = 0.0;
    double max_position_error_friction_pgs = 0.0;
    double max_position_error_polyhedral = 0.0;
    double peak_residual_soccp = 0.0;
    double peak_residual_friction_pgs = 0.0;
    double peak_residual_polyhedral = 0.0;
    double max_soccp_lambda_n = 0.0;
    double max_friction_pgs_lambda_n = 0.0;
    double max_polyhedral_lambda_n = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        SOCCPSolution soccp_solution;
        const SOCCPProblem soccp_problem = makeSOCCPProblem(scene, soccp_state.velocity);
        const bool soccp_converged = soccp_solver.solve(soccp_problem, soccp_solution);
        if (!soccp_converged || soccp_solution.velocity.size() != 3) {
            std::cerr << "SOCCP sliding benchmark failed at step " << step << '\n';
            return 1;
        }
        soccp_state.velocity = soccp_solution.velocity;
        soccp_state.position += scene.dt * soccp_state.velocity;

        FrictionPGSSolution friction_pgs_solution;
        const FrictionPGSProblem friction_pgs_problem =
            makeFrictionPGSProblem(scene, friction_pgs_state.velocity);
        const bool friction_pgs_converged =
            friction_pgs_solver.solve(friction_pgs_problem, friction_pgs_solution);
        if (!friction_pgs_converged || friction_pgs_solution.velocity.size() != 3) {
            std::cerr << "Friction PGS sliding benchmark failed at step " << step << '\n';
            return 1;
        }
        friction_pgs_state.velocity = friction_pgs_solution.velocity;
        friction_pgs_state.position += scene.dt * friction_pgs_state.velocity;

        PolyhedralFrictionSolution polyhedral_solution;
        const PolyhedralFrictionProblem polyhedral_problem =
            makePolyhedralProblem(scene, polyhedral_state.velocity);
        const bool polyhedral_converged =
            polyhedral_solver.solve(polyhedral_problem, polyhedral_solution);
        if (!polyhedral_converged || polyhedral_solution.velocity.size() != 3) {
            std::cerr << "Polyhedral sliding benchmark failed at step " << step << '\n';
            return 1;
        }
        polyhedral_state.velocity = polyhedral_solution.velocity;
        polyhedral_state.position += scene.dt * polyhedral_state.velocity;

        const double time = (static_cast<double>(step) + 1.0) * scene.dt;
        const double x_ref = referencePosition(time, scene);
        const double v_ref = referenceVelocity(time, scene);
        const double x_soccp = soccp_state.position.x();
        const double x_friction_pgs = friction_pgs_state.position.x();
        const double x_polyhedral = polyhedral_state.position.x();
        const double v_soccp = std::abs(soccp_state.velocity.x());
        const double v_friction_pgs = std::abs(friction_pgs_state.velocity.x());
        const double v_polyhedral = std::abs(polyhedral_state.velocity.x());

        time_history.push_back(time);
        speed_soccp_history.push_back(v_soccp);
        speed_friction_pgs_history.push_back(v_friction_pgs);
        speed_polyhedral_history.push_back(v_polyhedral);

        max_velocity_error_soccp = std::max(max_velocity_error_soccp, std::abs(v_soccp - v_ref));
        max_velocity_error_friction_pgs =
            std::max(max_velocity_error_friction_pgs, std::abs(v_friction_pgs - v_ref));
        max_velocity_error_polyhedral =
            std::max(max_velocity_error_polyhedral, std::abs(v_polyhedral - v_ref));
        max_position_error_soccp = std::max(max_position_error_soccp, std::abs(x_soccp - x_ref));
        max_position_error_friction_pgs =
            std::max(max_position_error_friction_pgs, std::abs(x_friction_pgs - x_ref));
        max_position_error_polyhedral =
            std::max(max_position_error_polyhedral, std::abs(x_polyhedral - x_ref));
        peak_residual_soccp =
            std::max(peak_residual_soccp, soccp_solution.residual.lpNorm<Eigen::Infinity>());
        peak_residual_friction_pgs =
            std::max(peak_residual_friction_pgs, friction_pgs_solution.residual.lpNorm<Eigen::Infinity>());
        peak_residual_polyhedral =
            std::max(peak_residual_polyhedral, polyhedral_solution.residual.lpNorm<Eigen::Infinity>());
        max_soccp_lambda_n = std::max(max_soccp_lambda_n, soccp_solution.lambda_n(0));
        max_friction_pgs_lambda_n =
            std::max(max_friction_pgs_lambda_n, friction_pgs_solution.lambda_n(0));
        max_polyhedral_lambda_n =
            std::max(max_polyhedral_lambda_n, polyhedral_solution.lambda_n(0));

        rows.push_back(vectorToRow({
            time,
            x_ref,
            v_ref,
            x_soccp,
            v_soccp,
            x_friction_pgs,
            v_friction_pgs,
            x_polyhedral,
            v_polyhedral,
            soccp_solution.lambda_n(0),
            friction_pgs_solution.lambda_n(0),
            polyhedral_solution.lambda_n(0),
            soccp_solution.lambda_t(0),
            friction_pgs_solution.lambda_t(0),
            polyhedral_solution.lambda_t(0),
            soccp_solution.residual.lpNorm<Eigen::Infinity>(),
            friction_pgs_solution.residual.lpNorm<Eigen::Infinity>(),
            polyhedral_solution.residual.lpNorm<Eigen::Infinity>(),
            soccp_solution.scaled_residual,
            friction_pgs_solution.scaled_residual,
            polyhedral_solution.scaled_residual,
            static_cast<double>(soccp_solution.stats.iterations),
            static_cast<double>(friction_pgs_solution.stats.iterations),
            static_cast<double>(polyhedral_solution.stats.iterations)
        }));
    }

    const double stop_time_ref =
        scene.initial_speed / (scene.friction_coefficient * scene.gravity);
    const double stop_distance_ref =
        0.5 * scene.initial_speed * stop_time_ref;
    const double stop_time_soccp =
        firstBelowThreshold(time_history, speed_soccp_history, scene.stop_threshold);
    const double stop_time_friction_pgs =
        firstBelowThreshold(time_history, speed_friction_pgs_history, scene.stop_threshold);
    const double stop_time_polyhedral =
        firstBelowThreshold(time_history, speed_polyhedral_history, scene.stop_threshold);

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_sliding_friction_reference.csv",
        {
            "time",
            "x_ref",
            "v_ref",
            "x_soccp",
            "v_soccp",
            "x_friction_pgs",
            "v_friction_pgs",
            "x_polyhedral",
            "v_polyhedral",
            "lambda_n_soccp",
            "lambda_n_friction_pgs",
            "lambda_n_polyhedral",
            "lambda_t_soccp",
            "lambda_t_friction_pgs",
            "lambda_t_polyhedral",
            "residual_soccp",
            "residual_friction_pgs",
            "residual_polyhedral",
            "scaled_residual_soccp",
            "scaled_residual_friction_pgs",
            "scaled_residual_polyhedral",
            "iterations_soccp",
            "iterations_friction_pgs",
            "iterations_polyhedral"
        },
        rows);
    writeCsv(
        dir / "benchmark_engine_sliding_friction_reference_summary.csv",
        {
            "stop_time_ref",
            "stop_time_soccp",
            "stop_time_friction_pgs",
            "stop_time_polyhedral",
            "stop_distance_ref",
            "stop_distance_soccp",
            "stop_distance_friction_pgs",
            "stop_distance_polyhedral",
            "max_velocity_error_soccp",
            "max_velocity_error_friction_pgs",
            "max_velocity_error_polyhedral",
            "max_position_error_soccp",
            "max_position_error_friction_pgs",
            "max_position_error_polyhedral",
            "peak_residual_soccp",
            "peak_residual_friction_pgs",
            "peak_residual_polyhedral",
            "max_lambda_n_soccp",
            "max_lambda_n_friction_pgs",
            "max_lambda_n_polyhedral"
        },
        {vectorToRow({
            stop_time_ref,
            stop_time_soccp,
            stop_time_friction_pgs,
            stop_time_polyhedral,
            stop_distance_ref,
            soccp_state.position.x(),
            friction_pgs_state.position.x(),
            polyhedral_state.position.x(),
            max_velocity_error_soccp,
            max_velocity_error_friction_pgs,
            max_velocity_error_polyhedral,
            max_position_error_soccp,
            max_position_error_friction_pgs,
            max_position_error_polyhedral,
            peak_residual_soccp,
            peak_residual_friction_pgs,
            peak_residual_polyhedral,
            max_soccp_lambda_n,
            max_friction_pgs_lambda_n,
            max_polyhedral_lambda_n
        })});

    std::cout << "benchmark_engine_sliding_friction_reference.csv written to "
              << (dir / "benchmark_engine_sliding_friction_reference.csv") << '\n';
    std::cout << "max_velocity_error_soccp=" << max_velocity_error_soccp
              << " max_velocity_error_friction_pgs=" << max_velocity_error_friction_pgs
              << " max_velocity_error_polyhedral=" << max_velocity_error_polyhedral << '\n';
    return 0;
}
