#include "BenchmarkCommon.h"
#include "solver/FourDPGSSolver.h"
#include "solver/SOCCPSolver.h"
#include <cmath>
#include <iostream>

namespace {

using namespace vde;
using namespace vde::benchmarks;

RigidBodyProperties makeBoxProps(double mass,
                                 double width,
                                 double height,
                                 double depth) {
    return RigidBodyProperties::box(mass, width, height, depth);
}

struct ReferenceCase {
    double x = 0.0;
    double z = 0.0;
    double yaw = 0.0;
    double vx = 0.65;
    double vz = -0.35;
    double omega = 7.5;
};

Eigen::MatrixXd makeMassMatrix(const RigidBodyProperties& props) {
    Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Zero(6, 6);
    mass_matrix.diagonal() << props.mass,
                               props.mass,
                               props.mass,
                               props.inertia_local(0, 0),
                               props.inertia_local(1, 1),
                               props.inertia_local(2, 2);
    return mass_matrix;
}

Eigen::VectorXd makeFreeVelocity(const ReferenceCase& state, double gravity, double dt) {
    Eigen::VectorXd free_velocity = Eigen::VectorXd::Zero(6);
    free_velocity(0) = state.vx;
    free_velocity(1) = -gravity * dt;
    free_velocity(2) = state.vz;
    free_velocity(4) = state.omega;
    return free_velocity;
}

void integrateState(ReferenceCase& state, const Eigen::VectorXd& solved_velocity, double dt) {
    state.vx = solved_velocity(0);
    state.vz = solved_velocity(2);
    state.omega = solved_velocity(4);
    state.x += dt * state.vx;
    state.z += dt * state.vz;
    state.yaw += dt * state.omega;
}

}  // namespace

int main() {
    constexpr double dt = 0.005;
    constexpr double duration = 1.0;
    constexpr double gravity = 9.81;
    constexpr double friction = 0.42;
    constexpr double torsion_radius = 0.27;

    const RigidBodyProperties props = makeBoxProps(2.4, 0.80, 0.18, 0.62);
    const Eigen::MatrixXd mass_matrix = makeMassMatrix(props);
    Eigen::MatrixXd normal_jacobian = Eigen::MatrixXd::Zero(1, 6);
    normal_jacobian(0, 1) = 1.0;
    Eigen::MatrixXd tangential_jacobian = Eigen::MatrixXd::Zero(3, 6);
    tangential_jacobian(0, 0) = 1.0;
    tangential_jacobian(1, 2) = 1.0;
    tangential_jacobian(2, 4) = torsion_radius;

    ReferenceCase soccp_state;
    ReferenceCase pgs_state = soccp_state;

    const int num_steps = static_cast<int>(duration / dt);
    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);

    double max_vx_diff = 0.0;
    double max_vz_diff = 0.0;
    double max_omega_diff = 0.0;
    double max_slip_diff = 0.0;
    double max_yaw_diff = 0.0;
    double max_lambda_n_diff = 0.0;
    double max_lambda_t_diff = 0.0;
    double peak_raw_residual_soccp = 0.0;
    double peak_raw_residual_pgs = 0.0;
    double peak_scaled_residual_soccp = 0.0;
    double peak_scaled_residual_pgs = 0.0;
    double peak_complementarity_soccp = 0.0;
    double peak_complementarity_pgs = 0.0;
    double max_iterations_soccp = 0.0;
    double max_iterations_pgs = 0.0;

    SOCCPSolver soccp_solver(160, 1e-12, 1e-8);
    FourDPGSSolver pgs_solver(400, 1e-12);
    Eigen::VectorXd soccp_warm_start;
    Eigen::VectorXd pgs_warm_start;

    for (int step = 0; step < num_steps; ++step) {
        SOCCPProblem soccp_problem;
        soccp_problem.mass_matrix = mass_matrix;
        soccp_problem.free_velocity = makeFreeVelocity(soccp_state, gravity, dt);
        soccp_problem.normal_jacobian = normal_jacobian;
        soccp_problem.tangential_jacobian = tangential_jacobian;
        soccp_problem.g_eq = Eigen::VectorXd::Zero(1);
        soccp_problem.friction_coefficients = Eigen::VectorXd::Constant(1, friction);
        soccp_problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
        soccp_problem.torsion_enabled = Eigen::ArrayXi::Ones(1);
        soccp_problem.time_step = dt;
        soccp_problem.epsilon = 1e-8;

        FourDPGSProblem pgs_problem;
        pgs_problem.mass_matrix = mass_matrix;
        pgs_problem.free_velocity = makeFreeVelocity(pgs_state, gravity, dt);
        pgs_problem.normal_jacobian = normal_jacobian;
        pgs_problem.tangential_jacobian = tangential_jacobian;
        pgs_problem.g_eq = Eigen::VectorXd::Zero(1);
        pgs_problem.friction_coefficients = Eigen::VectorXd::Constant(1, friction);
        pgs_problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
        pgs_problem.time_step = dt;

        SOCCPSolution soccp_solution;
        FourDPGSSolution pgs_solution;
        const Eigen::VectorXd* soccp_initial =
            (soccp_warm_start.size() == soccp_problem.stateSize()) ? &soccp_warm_start : nullptr;
        const Eigen::VectorXd* pgs_initial =
            (pgs_warm_start.size() == 4) ? &pgs_warm_start : nullptr;
        const bool soccp_converged =
            soccp_solver.solve(soccp_problem, soccp_initial, soccp_solution);
        const bool pgs_converged =
            pgs_solver.solve(pgs_problem, pgs_initial, pgs_solution);
        if (!soccp_converged || !pgs_converged) {
            std::cerr << "reference solvers failed to converge at step " << step << '\n';
            return 1;
        }

        soccp_warm_start = soccp_solution.state;
        pgs_warm_start = FourDPGSSolver::packWarmStart(
            pgs_solution.lambda_n,
            pgs_solution.lambda_t);

        integrateState(soccp_state, soccp_solution.velocity, dt);
        integrateState(pgs_state, pgs_solution.velocity, dt);

        const double soccp_slip = std::sqrt(soccp_state.x * soccp_state.x + soccp_state.z * soccp_state.z);
        const double pgs_slip = std::sqrt(pgs_state.x * pgs_state.x + pgs_state.z * pgs_state.z);
        max_vx_diff = std::max(max_vx_diff, std::abs(soccp_state.vx - pgs_state.vx));
        max_vz_diff = std::max(max_vz_diff, std::abs(soccp_state.vz - pgs_state.vz));
        max_omega_diff = std::max(max_omega_diff, std::abs(soccp_state.omega - pgs_state.omega));
        max_slip_diff = std::max(max_slip_diff, std::abs(soccp_slip - pgs_slip));
        max_yaw_diff = std::max(max_yaw_diff, std::abs(soccp_state.yaw - pgs_state.yaw));
        max_lambda_n_diff = std::max(
            max_lambda_n_diff,
            std::abs(soccp_solution.lambda_n(0) - pgs_solution.lambda_n(0)));
        max_lambda_t_diff = std::max(
            max_lambda_t_diff,
            (soccp_solution.lambda_t - pgs_solution.lambda_t).lpNorm<Eigen::Infinity>());
        peak_raw_residual_soccp = std::max(
            peak_raw_residual_soccp,
            soccp_solution.residual.lpNorm<Eigen::Infinity>());
        peak_raw_residual_pgs = std::max(
            peak_raw_residual_pgs,
            pgs_solution.residual.lpNorm<Eigen::Infinity>());
        peak_scaled_residual_soccp = std::max(
            peak_scaled_residual_soccp,
            soccp_solution.scaled_residual);
        peak_scaled_residual_pgs = std::max(
            peak_scaled_residual_pgs,
            pgs_solution.scaled_residual);
        peak_complementarity_soccp = std::max(
            peak_complementarity_soccp,
            soccp_solution.complementarity_violation);
        peak_complementarity_pgs = std::max(
            peak_complementarity_pgs,
            pgs_solution.complementarity_violation);
        max_iterations_soccp = std::max(
            max_iterations_soccp,
            static_cast<double>(soccp_solution.stats.iterations));
        max_iterations_pgs = std::max(
            max_iterations_pgs,
            static_cast<double>(pgs_solution.stats.iterations));

        rows.push_back(vectorToRow({
            (step + 1) * dt,
            soccp_state.vx,
            pgs_state.vx,
            soccp_state.vz,
            pgs_state.vz,
            soccp_state.omega,
            pgs_state.omega,
            soccp_slip,
            pgs_slip,
            soccp_state.yaw,
            pgs_state.yaw,
            soccp_solution.lambda_n(0),
            pgs_solution.lambda_n(0),
            soccp_solution.lambda_t(0),
            pgs_solution.lambda_t(0),
            soccp_solution.lambda_t(1),
            pgs_solution.lambda_t(1),
            soccp_solution.lambda_t(2),
            pgs_solution.lambda_t(2),
            soccp_solution.residual.lpNorm<Eigen::Infinity>(),
            pgs_solution.residual.lpNorm<Eigen::Infinity>(),
            soccp_solution.scaled_residual,
            pgs_solution.scaled_residual,
            soccp_solution.complementarity_violation,
            pgs_solution.complementarity_violation,
            static_cast<double>(soccp_solution.stats.iterations),
            static_cast<double>(pgs_solution.stats.iterations)
        }));
    }

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_4d_coupled_friction_reference.csv",
        {
            "time",
            "vx_soccp",
            "vx_4dpgs",
            "vz_soccp",
            "vz_4dpgs",
            "omega_soccp",
            "omega_4dpgs",
            "slip_soccp",
            "slip_4dpgs",
            "yaw_soccp",
            "yaw_4dpgs",
            "lambda_n_soccp",
            "lambda_n_4dpgs",
            "lambda_tx_soccp",
            "lambda_tx_4dpgs",
            "lambda_tz_soccp",
            "lambda_tz_4dpgs",
            "lambda_tau_soccp",
            "lambda_tau_4dpgs",
            "raw_residual_soccp",
            "raw_residual_4dpgs",
            "scaled_residual_soccp",
            "scaled_residual_4dpgs",
            "complementarity_soccp",
            "complementarity_4dpgs",
            "iterations_soccp",
            "iterations_4dpgs"
        },
        rows);
    writeCsv(
        dir / "benchmark_4d_coupled_friction_reference_summary.csv",
        {
            "max_vx_diff",
            "max_vz_diff",
            "max_omega_diff",
            "max_slip_diff",
            "max_yaw_diff",
            "max_lambda_n_diff",
            "max_lambda_t_diff",
            "peak_raw_residual_soccp",
            "peak_raw_residual_4dpgs",
            "peak_scaled_residual_soccp",
            "peak_scaled_residual_4dpgs",
            "peak_complementarity_soccp",
            "peak_complementarity_4dpgs",
            "max_iterations_soccp",
            "max_iterations_4dpgs",
            "final_slip_soccp",
            "final_slip_4dpgs",
            "final_omega_soccp",
            "final_omega_4dpgs"
        },
        {vectorToRow({
            max_vx_diff,
            max_vz_diff,
            max_omega_diff,
            max_slip_diff,
            max_yaw_diff,
            max_lambda_n_diff,
            max_lambda_t_diff,
            peak_raw_residual_soccp,
            peak_raw_residual_pgs,
            peak_scaled_residual_soccp,
            peak_scaled_residual_pgs,
            peak_complementarity_soccp,
            peak_complementarity_pgs,
            max_iterations_soccp,
            max_iterations_pgs,
            std::sqrt(soccp_state.x * soccp_state.x + soccp_state.z * soccp_state.z),
            std::sqrt(pgs_state.x * pgs_state.x + pgs_state.z * pgs_state.z),
            soccp_state.omega,
            pgs_state.omega
        })});

    std::cout << "benchmark_4d_coupled_friction_reference.csv written to "
              << (dir / "benchmark_4d_coupled_friction_reference.csv") << '\n';
    std::cout << "max_omega_diff=" << max_omega_diff
              << " max_slip_diff=" << max_slip_diff
              << " max_lambda_t_diff=" << max_lambda_t_diff << '\n';
    return 0;
}
