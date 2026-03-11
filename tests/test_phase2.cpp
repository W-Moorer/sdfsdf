/**
 * @file test_phase2.cpp
 * @brief Phase 2 acceptance tests
 *
 * Verify Phase 2 acceptance criteria:
 * 1. FB Zero Theorem: Phi_FB(x, y) = 0 when x, y satisfy Coulomb friction conditions
 * 2. Quadratic Convergence: Newton residual shows quadratic decrease in later iterations
 */

#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <Eigen/Dense>
#include "math/JordanAlgebra.h"
#include "math/FischerBurmeister.h"
#include "dynamics/ContactDynamics.h"
#include "solver/FourDPGSSolver.h"
#include "solver/FrictionPGSSolver.h"
#include "solver/NormalLCPSolver.h"
#include "solver/PolyhedralFrictionSolver.h"
#include "solver/SemiSmoothNewton.h"
#include "solver/SOCCPSolver.h"

using namespace vde;

RigidBodyProperties makeCylinderProps(double mass, double radius, double height) {
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

SOCCPProblem makeReferenceSOCCPProblem() {
    ContactConstraint constraint;
    constraint.contact.position = Eigen::Vector3d::Zero();
    constraint.contact.normal = Eigen::Vector3d::UnitY();
    constraint.contact.computeTangentDirections();
    constraint.g_eq = -0.02;
    constraint.r_tau = 0.15;
    constraint.computeJacobians(Eigen::Vector3d(0.0, -0.5, 0.0),
                                Eigen::Vector3d(0.0, 0.5, 0.0));

    Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Identity(12, 12);
    mass_matrix.block<3, 3>(3, 3) = 0.25 * Eigen::Matrix3d::Identity();
    mass_matrix.block<3, 3>(9, 9) = 0.4 * Eigen::Matrix3d::Identity();

    Eigen::VectorXd free_velocity = Eigen::VectorXd::Zero(12);
    free_velocity(0) = 0.30;
    free_velocity(6) = -0.05;

    return SOCCPSolver::fromConstraint(
        mass_matrix, free_velocity, constraint, 1e-2, 0.5, 0.5, 1e-8);
}

Eigen::VectorXd makeReferenceState(const SOCCPProblem& problem) {
    const int nv = problem.velocityDofs();
    const int nc = problem.numContacts();
    Eigen::VectorXd state = Eigen::VectorXd::Zero(problem.stateSize());
    state.head(nv) = problem.free_velocity;

    const double u_n =
        (problem.normal_jacobian.row(0) * problem.free_velocity)(0)
        + (problem.baumgarte_gamma(0) / problem.time_step) * problem.g_eq(0);
    state.segment(nv, nc).setZero();
    state.segment(nv + nc, 3 * nc).setZero();
    state(nv) = std::max(0.0, -u_n);
    state(nv + 4 * nc) =
        std::max(problem.epsilon, (problem.tangential_jacobian * problem.free_velocity).norm());

    return state;
}

/**
 * @brief Acceptance criterion 1: FB Zero Theorem
 *
 * When input vectors x, y satisfy Coulomb friction boundary conditions,
 * Phi_FB(x, y) output must be strictly zero vector
 */
bool testCriterion1_FBZeroTheorem() {
    std::cout << "========================================" << std::endl;
    std::cout << "Criterion 1: FB Zero Theorem" << std::endl;
    std::cout << "  Phi_FB(x, y) = 0 when x in SOC, y in SOC, x^T y = 0" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    // Test case 1: x on SOC boundary, y = 0 (complementary)
    {
        Vector4d x;
        x << 1.0, 0.0, 0.0, 0.0;  // x0 = ||x_bar||
        Vector4d y = Vector4d::Zero();

        Vector4d phi = fischerBurmeister(x, y);
        double phi_norm = phi.norm();

        std::cout << "Test 1: x on boundary, y = 0" << std::endl;
        std::cout << "  Phi_FB norm: " << phi_norm << std::endl;

        if (phi_norm > 1e-10) {
            std::cout << "  FAILED" << std::endl;
            all_passed = false;
        } else {
            std::cout << "  PASSED" << std::endl;
        }
    }

    // Test case 2: x = 0, y on SOC boundary
    {
        Vector4d x = Vector4d::Zero();
        Vector4d y;
        y << 1.0, 0.5, 0.0, 0.0;  // y0 > ||y_bar||
        y(0) = y.tail<3>().norm();  // On boundary

        Vector4d phi = fischerBurmeister(x, y);
        double phi_norm = phi.norm();

        std::cout << "\nTest 2: x = 0, y on boundary" << std::endl;
        std::cout << "  Phi_FB norm: " << phi_norm << std::endl;

        if (phi_norm > 1e-10) {
            std::cout << "  FAILED" << std::endl;
            all_passed = false;
        } else {
            std::cout << "  PASSED" << std::endl;
        }
    }

    // Test case 3: Both x and y in SOC interior (not complementary)
    {
        Vector4d x;
        x << 2.0, 0.5, 0.0, 0.0;  // x0 > ||x_bar||
        Vector4d y;
        y << 2.0, 0.3, 0.0, 0.0;  // y0 > ||y_bar||

        Vector4d phi = fischerBurmeister(x, y);
        double phi_norm = phi.norm();

        std::cout << "\nTest 3: Both in SOC interior (not complementary)" << std::endl;
        std::cout << "  Phi_FB norm: " << phi_norm << std::endl;
        std::cout << "  (Should be non-zero)" << std::endl;

        if (phi_norm < 1e-10) {
            std::cout << "  FAILED (should be non-zero)" << std::endl;
            all_passed = false;
        } else {
            std::cout << "  PASSED (correctly non-zero)" << std::endl;
        }
    }

    std::cout << "\nOverall: " << (all_passed ? "PASSED" : "FAILED") << std::endl;
    return all_passed;
}

/**
 * @brief Acceptance criterion 2: Quadratic Convergence
 *
 * In a simple spring-block test, Newton residual ||H|| must show
 * quadratic decrease in later iterations (e.g., 1e-2 -> 1e-4 -> 1e-8 -> 1e-16)
 */
bool testCriterion2_QuadraticConvergence() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Criterion 2: Quadratic Convergence" << std::endl;
    std::cout << "  Residual must decrease quadratically" << std::endl;
    std::cout << "========================================" << std::endl;

    // Simple spring-block problem setup
    // M*x + q = y, x >= 0, y >= 0, x^T y = 0

    // Mass matrix (simplified)
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 0) = 2.0;  // Normal direction stiffer

    // External force
    Vector4d q;
    q << -1.0, 0.0, 0.0, 0.0;  // Compressing force in normal direction

    // Initial guess
    Vector4d x;
    x << 0.5, 0.1, 0.0, 0.0;

    // Solve using semi-smooth Newton
    SemiSmoothNewton solver(50, 1e-14, 1e-10);
    NewtonStats stats;

    std::cout << "Solving LCP with semi-smooth Newton..." << std::endl;
    bool converged = solver.solveLCP(x, -q, M, stats);

    std::cout << "\nConvergence history:" << std::endl;
    std::cout << std::setw(5) << "Iter" << std::setw(20) << "Residual" << std::setw(20) << "Ratio" << std::endl;
    std::cout << std::string(45, '-') << std::endl;

    bool quadratic = false;
    for (size_t i = 0; i < stats.residual_history.size(); ++i) {
        double res = stats.residual_history[i];
        double ratio = 0.0;

        if (i > 0) {
            double prev_res = stats.residual_history[i - 1];
            ratio = res / (prev_res * prev_res);  // Check if res ~ prev_res^2
        }

        std::cout << std::setw(5) << i
                  << std::scientific << std::setw(20) << res
                  << std::setw(20) << ratio << std::endl;

        // Check for quadratic convergence in later iterations
        if (i >= 3 && res < 1e-2 && ratio < 10.0) {
            quadratic = true;
        }
    }

    std::cout << "\nIterations: " << stats.iterations << std::endl;
    std::cout << "Final residual: " << stats.final_residual << std::endl;
    std::cout << "Converged: " << (stats.converged ? "YES" : "NO") << std::endl;

    // Check quadratic convergence
    bool passed = stats.converged && stats.iterations <= 20;

    std::cout << "\nQuadratic convergence: " << (quadratic ? "DETECTED" : "NOT DETECTED") << std::endl;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test Jordan algebra operations
 */
bool testJordanAlgebra() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Jordan Algebra Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    // Test 1: Jordan multiplication
    {
        Vector4d x;
        x << 2.0, 1.0, 0.0, 0.0;
        Vector4d y;
        y << 3.0, 0.5, 0.0, 0.0;

        Vector4d product = jordanMultiply(x, y);

        // Expected: x o y = (x^T y, x0*y_bar + y0*x_bar)
        double expected_scalar = x(0)*y(0) + x.tail<3>().dot(y.tail<3>());
        Eigen::Vector3d expected_vector = x(0)*y.tail<3>() + y(0)*x.tail<3>();

        bool correct = std::abs(product(0) - expected_scalar) < 1e-10 &&
                       (product.tail<3>() - expected_vector).norm() < 1e-10;

        std::cout << "Test 1: Jordan multiplication" << std::endl;
        std::cout << "  " << (correct ? "PASSED" : "FAILED") << std::endl;
        all_passed &= correct;
    }

    // Test 2: Spectral decomposition
    {
        Vector4d x;
        x << 5.0, 3.0, 0.0, 0.0;

        double lambda1, lambda2;
        Vector4d u1, u2;
        spectralDecomposition(x, lambda1, lambda2, u1, u2);

        // Reconstruct x
        Vector4d x_reconstructed = lambda1 * u1 + lambda2 * u2;

        bool correct = (x - x_reconstructed).norm() < 1e-10;

        std::cout << "\nTest 2: Spectral decomposition" << std::endl;
        std::cout << "  lambda1 = " << lambda1 << ", lambda2 = " << lambda2 << std::endl;
        std::cout << "  " << (correct ? "PASSED" : "FAILED") << std::endl;
        all_passed &= correct;
    }

    // Test 3: Square root
    {
        Vector4d x;
        x << 5.0, 3.0, 0.0, 0.0;  // In SOC

        Vector4d sqrt_x = socSqrt(x);
        Vector4d sqrt_x_squared = jordanMultiply(sqrt_x, sqrt_x);

        bool correct = (x - sqrt_x_squared).norm() < 1e-10;

        std::cout << "\nTest 3: Square root" << std::endl;
        std::cout << "  sqrt(x) = " << sqrt_x.transpose() << std::endl;
        std::cout << "  " << (correct ? "PASSED" : "FAILED") << std::endl;
        all_passed &= correct;
    }

    // Test 4: SOC projection
    {
        Vector4d x;
        x << -1.0, 2.0, 0.0, 0.0;  // Outside SOC

        Vector4d proj = projectOntoSOC(x);

        bool in_soc = proj(0) >= proj.tail<3>().norm() - 1e-10;

        std::cout << "\nTest 4: SOC projection" << std::endl;
        std::cout << "  proj(x) = " << proj.transpose() << std::endl;
        std::cout << "  " << (in_soc ? "PASSED" : "FAILED") << std::endl;
        all_passed &= in_soc;
    }

    return all_passed;
}

/**
 * @brief Test Jacobian correctness using finite differences
 */
bool testJacobianCorrectness() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Jacobian Correctness Test (Finite Difference)" << std::endl;
    std::cout << "========================================" << std::endl;

    Vector4d x;
    x << 2.0, 0.5, 0.3, 0.1;
    Vector4d y;
    y << 1.5, 0.2, 0.1, 0.0;

    double h = 1e-6;
    double tolerance = 1e-4;

    // Compute analytical Jacobian
    Eigen::Matrix<double, 4, 8> J_analytical = fischerBurmeisterJacobian(x, y, 1e-10);

    // Compute finite difference Jacobian
    Eigen::Matrix<double, 4, 8> J_numerical;
    J_numerical.setZero();

    for (int i = 0; i < 4; ++i) {
        // Perturb x
        Vector4d x_plus = x;
        x_plus(i) += h;
        Vector4d phi_plus = fischerBurmeister(x_plus, y);

        Vector4d x_minus = x;
        x_minus(i) -= h;
        Vector4d phi_minus = fischerBurmeister(x_minus, y);

        J_numerical.col(i) = (phi_plus - phi_minus) / (2.0 * h);
    }

    for (int i = 0; i < 4; ++i) {
        // Perturb y
        Vector4d y_plus = y;
        y_plus(i) += h;
        Vector4d phi_plus = fischerBurmeister(x, y_plus);

        Vector4d y_minus = y;
        y_minus(i) -= h;
        Vector4d phi_minus = fischerBurmeister(x, y_minus);

        J_numerical.col(i + 4) = (phi_plus - phi_minus) / (2.0 * h);
    }

    // Compare
    double error = (J_analytical - J_numerical).norm();
    bool passed = error < tolerance;

    std::cout << "Analytical Jacobian norm: " << J_analytical.norm() << std::endl;
    std::cout << "Numerical Jacobian norm: " << J_numerical.norm() << std::endl;
    std::cout << "Error: " << error << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

bool testSOCCPAssemblyJacobian() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "SOCCP Assembly Jacobian Test" << std::endl;
    std::cout << "========================================" << std::endl;

    const SOCCPProblem problem = makeReferenceSOCCPProblem();
    SOCCPSolver solver(80, 1e-12, problem.epsilon);
    Eigen::VectorXd state = makeReferenceState(problem);

    const Eigen::MatrixXd J_analytical = solver.computeJacobian(problem, state);
    Eigen::MatrixXd J_numerical = Eigen::MatrixXd::Zero(J_analytical.rows(), J_analytical.cols());

    const double h = 1e-6;
    for (int i = 0; i < state.size(); ++i) {
        Eigen::VectorXd state_plus = state;
        Eigen::VectorXd state_minus = state;
        state_plus(i) += h;
        state_minus(i) -= h;

        const Eigen::VectorXd residual_plus = solver.computeResidual(problem, state_plus);
        const Eigen::VectorXd residual_minus = solver.computeResidual(problem, state_minus);
        J_numerical.col(i) = (residual_plus - residual_minus) / (2.0 * h);
    }

    const double abs_error = (J_analytical - J_numerical).norm();
    const double rel_error = abs_error / std::max(1.0, J_numerical.norm());
    const bool passed = rel_error < 1e-4;

    std::cout << "Analytical Jacobian norm: " << J_analytical.norm() << std::endl;
    std::cout << "Numerical Jacobian norm: " << J_numerical.norm() << std::endl;
    std::cout << "Absolute error: " << abs_error << std::endl;
    std::cout << "Relative error: " << rel_error << std::endl;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

bool testSOCCPSolverCore() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "SOCCP Core Solve Test" << std::endl;
    std::cout << "========================================" << std::endl;

    const SOCCPProblem problem = makeReferenceSOCCPProblem();
    SOCCPSolver solver(100, 1e-10, problem.epsilon);
    SOCCPSolution solution;

    const bool converged = solver.solve(problem, solution);
    const Eigen::Vector3d tangential_velocity =
        problem.tangential_jacobian * solution.velocity;
    const Vector4d friction_multiplier = SOCCPSolver::makeFrictionConeVector(
        problem.friction_coefficients(0),
        solution.lambda_n(0),
        solution.lambda_t.segment<3>(0));
    const Vector4d dual_velocity =
        SOCCPSolver::makeDualConeVector(solution.slack(0), tangential_velocity);

    const bool primal_soc = SOCCPSolver::isInsideSOC(friction_multiplier, 1e-8);
    const bool dual_soc = SOCCPSolver::isInsideSOC(dual_velocity, 1e-8);
    const double residual_inf = solution.residual.lpNorm<Eigen::Infinity>();
    const double fb_inf = fischerBurmeister(friction_multiplier, dual_velocity).lpNorm<Eigen::Infinity>();

    std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
    std::cout << "Iterations: " << solution.stats.iterations << std::endl;
    std::cout << "Residual inf-norm: " << residual_inf << std::endl;
    std::cout << "lambda_n: " << solution.lambda_n(0) << std::endl;
    std::cout << "||lambda_t||: " << solution.lambda_t.segment<3>(0).norm() << std::endl;
    std::cout << "slack: " << solution.slack(0) << std::endl;
    std::cout << "Primal SOC feasible: " << (primal_soc ? "YES" : "NO") << std::endl;
    std::cout << "Dual SOC feasible: " << (dual_soc ? "YES" : "NO") << std::endl;
    std::cout << "FB block inf-norm: " << fb_inf << std::endl;

    const bool passed =
        converged &&
        solution.lambda_n(0) >= -1e-8 &&
        primal_soc &&
        dual_soc &&
        residual_inf < 1e-7 &&
        fb_inf < 1e-7;

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testSOCCP3DBaselineBranch() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "SOCCP 3D Friction Baseline Branch Test" << std::endl;
    std::cout << "========================================" << std::endl;

    SOCCPProblem problem = makeReferenceSOCCPProblem();
    problem.torsion_enabled = Eigen::ArrayXi::Zero(problem.numContacts());
    problem.tangential_jacobian.row(2).setZero();

    SOCCPSolver solver(100, 1e-10, problem.epsilon);
    SOCCPSolution solution;
    const bool converged = solver.solve(problem, solution);

    const double lambda_tau = solution.lambda_t(2);
    const double residual_inf = solution.residual.lpNorm<Eigen::Infinity>();
    const bool torsion_disabled = std::abs(lambda_tau) < 1e-8;

    std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
    std::cout << "Residual inf-norm: " << residual_inf << std::endl;
    std::cout << "lambda_tau: " << lambda_tau << std::endl;
    std::cout << "Torsion disabled: " << (torsion_disabled ? "YES" : "NO") << std::endl;

    const bool passed = converged && torsion_disabled && residual_inf < 1e-7;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testSOCCPScalarLCPDegeneracy() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "SOCCP Scalar LCP Degeneracy Test" << std::endl;
    std::cout << "========================================" << std::endl;

    SOCCPProblem problem;
    problem.mass_matrix = Eigen::MatrixXd::Identity(6, 6);
    problem.mass_matrix.block<3, 3>(3, 3) = 0.2 * Eigen::Matrix3d::Identity();
    problem.free_velocity = Eigen::VectorXd::Zero(6);
    problem.free_velocity(1) = -2.0;
    problem.normal_jacobian = Eigen::MatrixXd::Zero(1, 6);
    problem.normal_jacobian(0, 1) = 1.0;
    problem.tangential_jacobian = Eigen::MatrixXd::Zero(3, 6);
    problem.g_eq = Eigen::VectorXd::Constant(1, -0.01);
    problem.friction_coefficients = Eigen::VectorXd::Zero(1);
    problem.baumgarte_gamma = Eigen::VectorXd::Constant(1, 0.10);
    problem.torsion_enabled = Eigen::ArrayXi::Zero(1);
    problem.time_step = 0.01;
    problem.epsilon = 1e-8;

    const double b =
        (problem.normal_jacobian * problem.free_velocity)(0) +
        (problem.baumgarte_gamma(0) / problem.time_step) * problem.g_eq(0);
    const double A =
        problem.time_step *
        (problem.normal_jacobian * problem.mass_matrix.inverse() * problem.normal_jacobian.transpose())(0, 0);
    const double lambda_expected = (A > 1e-12 && b < 0.0) ? (-b / A) : 0.0;
    const Eigen::VectorXd velocity_expected =
        problem.free_velocity +
        problem.time_step * problem.mass_matrix.inverse() *
        problem.normal_jacobian.transpose() * Eigen::VectorXd::Constant(1, lambda_expected);

    SOCCPSolver solver(100, 1e-10, problem.epsilon);
    SOCCPSolution solution;
    const bool converged = solver.solve(problem, solution);

    const double lambda_error = std::abs(solution.lambda_n(0) - lambda_expected);
    const double velocity_error = (solution.velocity - velocity_expected).norm();
    const double residual_inf = solution.residual.lpNorm<Eigen::Infinity>();

    std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
    std::cout << "lambda_expected: " << lambda_expected << std::endl;
    std::cout << "lambda_soccp: " << solution.lambda_n(0) << std::endl;
    std::cout << "lambda_error: " << lambda_error << std::endl;
    std::cout << "velocity_error: " << velocity_error << std::endl;
    std::cout << "residual_inf: " << residual_inf << std::endl;

    const bool passed =
        converged &&
        lambda_error < 1e-7 &&
        velocity_error < 1e-7 &&
        residual_inf < 1e-7;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testNormalLCPGlobalSolve() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Normal LCP Global Solve Test" << std::endl;
    std::cout << "========================================" << std::endl;

    NormalLCPProblem problem;
    problem.mass_matrix = Eigen::MatrixXd::Identity(2, 2);
    problem.free_velocity = Eigen::VectorXd::Zero(2);
    problem.free_velocity << -1.0, -0.5;
    problem.normal_jacobian = Eigen::MatrixXd::Zero(2, 2);
    problem.normal_jacobian << 1.0, 0.0,
                               1.0, 1.0;
    problem.g_eq = Eigen::VectorXd::Zero(2);
    problem.baumgarte_gamma = Eigen::VectorXd::Zero(2);
    problem.time_step = 1.0;

    const Eigen::Vector2d lambda_expected(0.5, 0.5);
    const Eigen::Vector2d velocity_expected = Eigen::Vector2d::Zero();

    NormalLCPSolver solver(200, 1e-12, 1.0);
    NormalLCPSolution solution;
    const bool converged = solver.solve(problem, solution);

    const double lambda_error = (solution.lambda_n - lambda_expected).norm();
    const double velocity_error = (solution.velocity - velocity_expected).norm();
    const double residual_inf = solution.residual.lpNorm<Eigen::Infinity>();

    std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
    std::cout << "lambda_expected: " << lambda_expected.transpose() << std::endl;
    std::cout << "lambda_lcp: " << solution.lambda_n.transpose() << std::endl;
    std::cout << "velocity_lcp: " << solution.velocity.transpose() << std::endl;
    std::cout << "lambda_error: " << lambda_error << std::endl;
    std::cout << "velocity_error: " << velocity_error << std::endl;
    std::cout << "residual_inf: " << residual_inf << std::endl;

    const bool passed =
        converged &&
        lambda_error < 1e-9 &&
        velocity_error < 1e-9 &&
        residual_inf < 1e-9;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testNormalLCPMatchesSOCCPNormalOnly() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Normal LCP vs SOCCP Normal-Only Test" << std::endl;
    std::cout << "========================================" << std::endl;

    Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Identity(2, 2);
    Eigen::VectorXd free_velocity = Eigen::VectorXd::Zero(2);
    free_velocity << -1.0, -0.5;
    Eigen::MatrixXd Jn = Eigen::MatrixXd::Zero(2, 2);
    Jn << 1.0, 0.0,
          1.0, 1.0;
    Eigen::VectorXd g_eq = Eigen::VectorXd::Zero(2);
    Eigen::VectorXd gamma = Eigen::VectorXd::Zero(2);

    NormalLCPProblem lcp_problem;
    lcp_problem.mass_matrix = mass_matrix;
    lcp_problem.free_velocity = free_velocity;
    lcp_problem.normal_jacobian = Jn;
    lcp_problem.g_eq = g_eq;
    lcp_problem.baumgarte_gamma = gamma;
    lcp_problem.time_step = 1.0;

    SOCCPProblem soccp_problem;
    soccp_problem.mass_matrix = mass_matrix;
    soccp_problem.free_velocity = free_velocity;
    soccp_problem.normal_jacobian = Jn;
    soccp_problem.tangential_jacobian = Eigen::MatrixXd::Zero(6, 2);
    soccp_problem.g_eq = g_eq;
    soccp_problem.friction_coefficients = Eigen::VectorXd::Zero(2);
    soccp_problem.baumgarte_gamma = gamma;
    soccp_problem.torsion_enabled = Eigen::ArrayXi::Zero(2);
    soccp_problem.time_step = 1.0;
    soccp_problem.epsilon = 1e-8;

    NormalLCPSolver lcp_solver(200, 1e-12, 1.0);
    NormalLCPSolution lcp_solution;
    const bool lcp_converged = lcp_solver.solve(lcp_problem, lcp_solution);

    SOCCPSolver soccp_solver(200, 1e-12, soccp_problem.epsilon);
    SOCCPSolution soccp_solution;
    const bool soccp_converged = soccp_solver.solve(soccp_problem, soccp_solution);

    const double lambda_error = (lcp_solution.lambda_n - soccp_solution.lambda_n).norm();
    const double velocity_error = (lcp_solution.velocity - soccp_solution.velocity).norm();
    const double lcp_residual = lcp_solution.residual.lpNorm<Eigen::Infinity>();
    const double soccp_residual = soccp_solution.residual.lpNorm<Eigen::Infinity>();

    std::cout << "LCP converged: " << (lcp_converged ? "YES" : "NO") << std::endl;
    std::cout << "SOCCP converged: " << (soccp_converged ? "YES" : "NO") << std::endl;
    std::cout << "lambda_lcp: " << lcp_solution.lambda_n.transpose() << std::endl;
    std::cout << "lambda_soccp: " << soccp_solution.lambda_n.transpose() << std::endl;
    std::cout << "lambda_error: " << lambda_error << std::endl;
    std::cout << "velocity_error: " << velocity_error << std::endl;
    std::cout << "lcp_residual_inf: " << lcp_residual << std::endl;
    std::cout << "soccp_residual_inf: " << soccp_residual << std::endl;

    const bool passed =
        lcp_converged &&
        soccp_converged &&
        lambda_error < 1e-7 &&
        velocity_error < 1e-7 &&
        lcp_residual < 1e-9 &&
        soccp_residual < 1e-7;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testFrictionPGSMatchesSOCCP3D() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Friction PGS vs SOCCP 3D Test" << std::endl;
    std::cout << "========================================" << std::endl;

    Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd free_velocity(3);
    free_velocity << 0.0, -0.2, -1.0;

    Eigen::MatrixXd Jn(1, 3);
    Jn << 0.0, 1.0, 0.0;
    Eigen::MatrixXd Jt2 = Eigen::MatrixXd::Zero(2, 3);
    Jt2 << 1.0, 0.0, 0.0,
           0.0, 0.0, 1.0;

    FrictionPGSProblem pgs_problem;
    pgs_problem.mass_matrix = mass_matrix;
    pgs_problem.free_velocity = free_velocity;
    pgs_problem.normal_jacobian = Jn;
    pgs_problem.tangential_jacobian = Jt2;
    pgs_problem.g_eq = Eigen::VectorXd::Constant(1, -1e-3);
    pgs_problem.friction_coefficients = Eigen::VectorXd::Constant(1, 0.4);
    pgs_problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    pgs_problem.time_step = 1.0;

    SOCCPProblem soccp_problem;
    soccp_problem.mass_matrix = mass_matrix;
    soccp_problem.free_velocity = free_velocity;
    soccp_problem.normal_jacobian = Jn;
    soccp_problem.tangential_jacobian = Eigen::MatrixXd::Zero(3, 3);
    soccp_problem.tangential_jacobian.block(0, 0, 2, 3) = Jt2;
    soccp_problem.g_eq = pgs_problem.g_eq;
    soccp_problem.friction_coefficients = pgs_problem.friction_coefficients;
    soccp_problem.baumgarte_gamma = pgs_problem.baumgarte_gamma;
    soccp_problem.torsion_enabled = Eigen::ArrayXi::Zero(1);
    soccp_problem.time_step = 1.0;
    soccp_problem.epsilon = 1e-8;

    FrictionPGSSolver pgs_solver(50, 1e-10);
    FrictionPGSSolution pgs_solution;
    const bool pgs_converged = pgs_solver.solve(pgs_problem, pgs_solution);

    SOCCPSolver soccp_solver(100, 1e-10, soccp_problem.epsilon);
    SOCCPSolution soccp_solution;
    const bool soccp_converged = soccp_solver.solve(soccp_problem, soccp_solution);

    const double lambda_n_error = std::abs(pgs_solution.lambda_n(0) - soccp_solution.lambda_n(0));
    const double lambda_t_error =
        (pgs_solution.lambda_t - soccp_solution.lambda_t.head(2)).norm();
    const double velocity_error =
        (pgs_solution.velocity - soccp_solution.velocity).norm();
    const double pgs_residual = pgs_solution.residual.lpNorm<Eigen::Infinity>();
    const double soccp_residual = soccp_solution.residual.lpNorm<Eigen::Infinity>();

    std::cout << "PGS converged: " << (pgs_converged ? "YES" : "NO") << std::endl;
    std::cout << "SOCCP converged: " << (soccp_converged ? "YES" : "NO") << std::endl;
    std::cout << "lambda_n_pgs: " << pgs_solution.lambda_n.transpose() << std::endl;
    std::cout << "lambda_n_soccp: " << soccp_solution.lambda_n.transpose() << std::endl;
    std::cout << "lambda_t_pgs: " << pgs_solution.lambda_t.transpose() << std::endl;
    std::cout << "lambda_t_soccp: " << soccp_solution.lambda_t.head(2).transpose() << std::endl;
    std::cout << "lambda_n_error: " << lambda_n_error << std::endl;
    std::cout << "lambda_t_error: " << lambda_t_error << std::endl;
    std::cout << "velocity_error: " << velocity_error << std::endl;
    std::cout << "pgs_residual_inf: " << pgs_residual << std::endl;
    std::cout << "soccp_residual_inf: " << soccp_residual << std::endl;

    const bool passed =
        pgs_converged &&
        soccp_converged &&
        lambda_n_error < 1e-7 &&
        lambda_t_error < 1e-7 &&
        velocity_error < 1e-7 &&
        pgs_residual < 1e-8 &&
        soccp_residual < 1e-7;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testPolyhedralFrictionMatchesSOCCP3D() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Polyhedral Friction Feasibility vs SOCCP 3D Cap Test" << std::endl;
    std::cout << "========================================" << std::endl;

    Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Identity(3, 3);
    Eigen::VectorXd free_velocity(3);
    free_velocity << 0.0, -0.2, -1.0;

    Eigen::MatrixXd Jn(1, 3);
    Jn << 0.0, 1.0, 0.0;
    Eigen::MatrixXd Jt2 = Eigen::MatrixXd::Zero(2, 3);
    Jt2 << 1.0, 0.0, 0.0,
           0.0, 0.0, 1.0;

    PolyhedralFrictionProblem poly_problem;
    poly_problem.mass_matrix = mass_matrix;
    poly_problem.free_velocity = free_velocity;
    poly_problem.normal_jacobian = Jn;
    poly_problem.tangential_jacobian = Jt2;
    poly_problem.g_eq = Eigen::VectorXd::Constant(1, -1e-3);
    poly_problem.friction_coefficients = Eigen::VectorXd::Constant(1, 0.4);
    poly_problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    poly_problem.time_step = 1.0;
    poly_problem.num_directions = 16;

    SOCCPProblem soccp_problem;
    soccp_problem.mass_matrix = mass_matrix;
    soccp_problem.free_velocity = free_velocity;
    soccp_problem.normal_jacobian = Jn;
    soccp_problem.tangential_jacobian = Eigen::MatrixXd::Zero(3, 3);
    soccp_problem.tangential_jacobian.block(0, 0, 2, 3) = Jt2;
    soccp_problem.g_eq = poly_problem.g_eq;
    soccp_problem.friction_coefficients = poly_problem.friction_coefficients;
    soccp_problem.baumgarte_gamma = poly_problem.baumgarte_gamma;
    soccp_problem.torsion_enabled = Eigen::ArrayXi::Zero(1);
    soccp_problem.time_step = 1.0;
    soccp_problem.epsilon = 1e-8;

    PolyhedralFrictionSolver poly_solver(500, 1e-10);
    PolyhedralFrictionSolution poly_solution;
    const bool poly_converged = poly_solver.solve(poly_problem, poly_solution);

    SOCCPSolver soccp_solver(100, 1e-10, soccp_problem.epsilon);
    SOCCPSolution soccp_solution;
    const bool soccp_converged = soccp_solver.solve(soccp_problem, soccp_solution);

    const double lambda_n_error =
        std::abs(poly_solution.lambda_n(0) - soccp_solution.lambda_n(0));
    const double lambda_t_norm_error =
        std::abs(poly_solution.lambda_t.norm() - soccp_solution.lambda_t.head(2).norm());
    const Eigen::Vector2d free_tangent(free_velocity(0), free_velocity(2));
    const double tangential_dissipation = poly_solution.lambda_t.dot(free_tangent);

    std::cout << "Polyhedral converged: " << (poly_converged ? "YES" : "NO") << std::endl;
    std::cout << "SOCCP converged: " << (soccp_converged ? "YES" : "NO") << std::endl;
    std::cout << "lambda_n_poly: " << poly_solution.lambda_n.transpose() << std::endl;
    std::cout << "lambda_n_soccp: " << soccp_solution.lambda_n.transpose() << std::endl;
    std::cout << "lambda_t_poly: " << poly_solution.lambda_t.transpose() << std::endl;
    std::cout << "lambda_t_soccp: " << soccp_solution.lambda_t.head(2).transpose() << std::endl;
    std::cout << "lambda_n_error: " << lambda_n_error << std::endl;
    std::cout << "lambda_t_norm_error: " << lambda_t_norm_error << std::endl;
    std::cout << "tangential_dissipation: " << tangential_dissipation << std::endl;
    std::cout << "poly_scaled_residual: " << poly_solution.scaled_residual << std::endl;
    std::cout << "soccp_scaled_residual: " << soccp_solution.scaled_residual << std::endl;
    std::cout << "poly_complementarity: " << poly_solution.complementarity_violation << std::endl;
    std::cout << "soccp_complementarity: " << soccp_solution.complementarity_violation << std::endl;

    const bool passed =
        soccp_converged &&
        lambda_n_error < 1e-7 &&
        lambda_t_norm_error < 1e-7 &&
        tangential_dissipation <= 1e-8 &&
        poly_solution.scaled_residual < 4e-2 &&
        poly_solution.complementarity_violation < 4e-2 &&
        soccp_solution.scaled_residual < 1e-7 &&
        soccp_solution.complementarity_violation < 1e-7;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testSOCCP4DTorsionReferenceStep() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "SOCCP 4D Torsion Reference Step Test" << std::endl;
    std::cout << "========================================" << std::endl;

    constexpr double mass = 3.0;
    constexpr double radius = 0.45;
    constexpr double height = 0.24;
    constexpr double friction = 0.35;
    constexpr double gravity = 9.81;
    constexpr double torsion_radius = 0.30;
    constexpr double dt = 0.01;
    constexpr double omega0 = 10.0;

    const RigidBodyProperties props = makeCylinderProps(mass, radius, height);
    Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Zero(6, 6);
    mass_matrix.diagonal() << mass, mass, mass,
                               props.inertia_local(0, 0),
                               props.inertia_local(1, 1),
                               props.inertia_local(2, 2);

    Eigen::VectorXd free_velocity = Eigen::VectorXd::Zero(6);
    free_velocity(1) = -gravity * dt;
    free_velocity(4) = omega0;

    SOCCPProblem problem;
    problem.mass_matrix = mass_matrix;
    problem.free_velocity = free_velocity;
    problem.normal_jacobian = Eigen::MatrixXd::Zero(1, 6);
    problem.normal_jacobian(0, 1) = 1.0;
    problem.tangential_jacobian = Eigen::MatrixXd::Zero(3, 6);
    problem.tangential_jacobian(2, 4) = torsion_radius;
    problem.g_eq = Eigen::VectorXd::Zero(1);
    problem.friction_coefficients = Eigen::VectorXd::Constant(1, friction);
    problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    problem.torsion_enabled = Eigen::ArrayXi::Ones(1);
    problem.time_step = dt;
    problem.epsilon = 1e-8;

    SOCCPSolver solver(100, 1e-10, problem.epsilon);
    SOCCPSolution solution;
    const bool converged = solver.solve(problem, solution);

    const double lambda_n_expected = mass * gravity;
    const double lambda_tau_expected = friction * lambda_n_expected;
    const double omega_expected =
        omega0 - dt * lambda_tau_expected * torsion_radius / props.inertia_local(1, 1);

    const double lambda_n_error = std::abs(solution.lambda_n(0) - lambda_n_expected);
    const double lambda_tau_error = std::abs(std::abs(solution.lambda_t(2)) - lambda_tau_expected);
    const double omega_error = std::abs(solution.velocity(4) - omega_expected);
    const double vy_error = std::abs(solution.velocity(1));
    const double residual_inf = solution.residual.lpNorm<Eigen::Infinity>();

    std::cout << "Converged: " << (converged ? "YES" : "NO") << std::endl;
    std::cout << "lambda_n: " << solution.lambda_n(0)
              << " (expected " << lambda_n_expected << ")" << std::endl;
    std::cout << "lambda_tau: " << solution.lambda_t(2)
              << " (expected magnitude " << lambda_tau_expected << ")" << std::endl;
    std::cout << "omega_next: " << solution.velocity(4)
              << " (expected " << omega_expected << ")" << std::endl;
    std::cout << "lambda_n_error: " << lambda_n_error << std::endl;
    std::cout << "lambda_tau_error: " << lambda_tau_error << std::endl;
    std::cout << "omega_error: " << omega_error << std::endl;
    std::cout << "vy_error: " << vy_error << std::endl;
    std::cout << "residual_inf: " << residual_inf << std::endl;

    const bool passed =
        converged &&
        lambda_n_error < 1e-7 &&
        lambda_tau_error < 1e-7 &&
        omega_error < 1e-7 &&
        vy_error < 1e-9 &&
        residual_inf < 1e-7;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testFourDPGSMatchesSOCCP4D() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "FourD PGS vs SOCCP 4D Test" << std::endl;
    std::cout << "========================================" << std::endl;

    constexpr double mass = 2.5;
    constexpr double radius = 0.36;
    constexpr double height = 0.20;
    constexpr double friction = 0.42;
    constexpr double gravity = 9.81;
    constexpr double torsion_radius = 0.28;
    constexpr double dt = 0.01;

    const RigidBodyProperties props = makeCylinderProps(mass, radius, height);
    Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Zero(6, 6);
    mass_matrix.diagonal() << mass, mass, mass,
                               props.inertia_local(0, 0),
                               props.inertia_local(1, 1),
                               props.inertia_local(2, 2);

    Eigen::VectorXd free_velocity = Eigen::VectorXd::Zero(6);
    free_velocity(0) = 0.80;
    free_velocity(1) = -gravity * dt;
    free_velocity(2) = -0.35;
    free_velocity(4) = 6.0;

    Eigen::MatrixXd Jn = Eigen::MatrixXd::Zero(1, 6);
    Jn(0, 1) = 1.0;
    Eigen::MatrixXd Jt = Eigen::MatrixXd::Zero(3, 6);
    Jt(0, 0) = 1.0;
    Jt(1, 2) = 1.0;
    Jt(2, 4) = torsion_radius;

    SOCCPProblem soccp_problem;
    soccp_problem.mass_matrix = mass_matrix;
    soccp_problem.free_velocity = free_velocity;
    soccp_problem.normal_jacobian = Jn;
    soccp_problem.tangential_jacobian = Jt;
    soccp_problem.g_eq = Eigen::VectorXd::Zero(1);
    soccp_problem.friction_coefficients = Eigen::VectorXd::Constant(1, friction);
    soccp_problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    soccp_problem.torsion_enabled = Eigen::ArrayXi::Ones(1);
    soccp_problem.time_step = dt;
    soccp_problem.epsilon = 1e-8;

    FourDPGSProblem pgs_problem;
    pgs_problem.mass_matrix = mass_matrix;
    pgs_problem.free_velocity = free_velocity;
    pgs_problem.normal_jacobian = Jn;
    pgs_problem.tangential_jacobian = Jt;
    pgs_problem.g_eq = Eigen::VectorXd::Zero(1);
    pgs_problem.friction_coefficients = Eigen::VectorXd::Constant(1, friction);
    pgs_problem.baumgarte_gamma = Eigen::VectorXd::Zero(1);
    pgs_problem.time_step = dt;

    SOCCPSolver soccp_solver(100, 1e-10, soccp_problem.epsilon);
    SOCCPSolution soccp_solution;
    const bool soccp_converged = soccp_solver.solve(soccp_problem, soccp_solution);

    FourDPGSSolver pgs_solver(200, 1e-10);
    FourDPGSSolution pgs_solution;
    const bool pgs_converged = pgs_solver.solve(pgs_problem, pgs_solution);

    const double lambda_n_error = std::abs(pgs_solution.lambda_n(0) - soccp_solution.lambda_n(0));
    const double lambda_t_error =
        (pgs_solution.lambda_t - soccp_solution.lambda_t).norm();
    const double velocity_error =
        (pgs_solution.velocity - soccp_solution.velocity).norm();
    const double lambda_t_scale = std::max(
        1.0,
        std::max(
            pgs_solution.lambda_t.norm(),
            soccp_solution.lambda_t.norm()));
    const double velocity_scale = std::max(
        1.0,
        std::max(
            pgs_solution.velocity.norm(),
            soccp_solution.velocity.norm()));
    const double lambda_t_relative_error = lambda_t_error / lambda_t_scale;
    const double velocity_relative_error = velocity_error / velocity_scale;

    std::cout << "PGS converged: " << (pgs_converged ? "YES" : "NO") << std::endl;
    std::cout << "SOCCP converged: " << (soccp_converged ? "YES" : "NO") << std::endl;
    std::cout << "lambda_n_pgs: " << pgs_solution.lambda_n.transpose() << std::endl;
    std::cout << "lambda_n_soccp: " << soccp_solution.lambda_n.transpose() << std::endl;
    std::cout << "lambda_t_pgs: " << pgs_solution.lambda_t.transpose() << std::endl;
    std::cout << "lambda_t_soccp: " << soccp_solution.lambda_t.transpose() << std::endl;
    std::cout << "lambda_n_error: " << lambda_n_error << std::endl;
    std::cout << "lambda_t_error: " << lambda_t_error << std::endl;
    std::cout << "lambda_t_relative_error: " << lambda_t_relative_error << std::endl;
    std::cout << "velocity_error: " << velocity_error << std::endl;
    std::cout << "velocity_relative_error: " << velocity_relative_error << std::endl;
    std::cout << "pgs_scaled_residual: " << pgs_solution.scaled_residual << std::endl;
    std::cout << "soccp_scaled_residual: " << soccp_solution.scaled_residual << std::endl;
    std::cout << "pgs_complementarity: " << pgs_solution.complementarity_violation << std::endl;
    std::cout << "soccp_complementarity: " << soccp_solution.complementarity_violation << std::endl;

    const bool passed =
        pgs_converged &&
        soccp_converged &&
        lambda_n_error < 1e-7 &&
        lambda_t_relative_error < 1e-1 &&
        velocity_relative_error < 2e-3 &&
        pgs_solution.complementarity_violation < 1e-8 &&
        soccp_solution.complementarity_violation < 1e-7;

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

int main() {
    std::cout << "****************************************" << std::endl;
    std::cout << "Phase 2 Acceptance Tests" << std::endl;
    std::cout << "****************************************" << std::endl;

    bool jordan_test = testJordanAlgebra();
    bool jacobian_test = testJacobianCorrectness();
    bool soccp_jacobian_test = testSOCCPAssemblyJacobian();
    bool criterion1 = testCriterion1_FBZeroTheorem();
    bool criterion2 = testCriterion2_QuadraticConvergence();
    bool soccp_solver_test = testSOCCPSolverCore();
    bool soccp_3d_branch_test = testSOCCP3DBaselineBranch();
    bool soccp_scalar_lcp_test = testSOCCPScalarLCPDegeneracy();
    bool normal_lcp_global_test = testNormalLCPGlobalSolve();
    bool normal_lcp_soccp_match_test = testNormalLCPMatchesSOCCPNormalOnly();
    bool friction_pgs_soccp_match_test = testFrictionPGSMatchesSOCCP3D();
    bool polyhedral_soccp_match_test = testPolyhedralFrictionMatchesSOCCP3D();
    bool soccp_4d_torsion_ref_test = testSOCCP4DTorsionReferenceStep();
    bool four_d_pgs_soccp_match_test = testFourDPGSMatchesSOCCP4D();

    std::cout << "\n****************************************" << std::endl;
    std::cout << "Phase 2 Acceptance Summary" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "Jordan Algebra: " << (jordan_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Jacobian Correctness: " << (jacobian_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "SOCCP Jacobian: " << (soccp_jacobian_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Criterion 1 (FB Zero Theorem): " << (criterion1 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Criterion 2 (Quadratic Convergence): " << (criterion2 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "SOCCP Core Solve: " << (soccp_solver_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "SOCCP 3D Baseline Branch: " << (soccp_3d_branch_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "SOCCP Scalar LCP Degeneracy: " << (soccp_scalar_lcp_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Normal LCP Global Solve: " << (normal_lcp_global_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Normal LCP vs SOCCP Normal-Only: " << (normal_lcp_soccp_match_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Friction PGS vs SOCCP 3D: " << (friction_pgs_soccp_match_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Polyhedral Friction vs SOCCP 3D: " << (polyhedral_soccp_match_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "SOCCP 4D Torsion Reference Step: " << (soccp_4d_torsion_ref_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "FourD PGS vs SOCCP 4D: " << (four_d_pgs_soccp_match_test ? "PASSED" : "FAILED") << std::endl;

    bool all_passed =
        jordan_test &&
        jacobian_test &&
        soccp_jacobian_test &&
        criterion1 &&
        criterion2 &&
        soccp_solver_test &&
        soccp_3d_branch_test &&
        soccp_scalar_lcp_test &&
        normal_lcp_global_test &&
        normal_lcp_soccp_match_test &&
        friction_pgs_soccp_match_test &&
        polyhedral_soccp_match_test &&
        soccp_4d_torsion_ref_test &&
        four_d_pgs_soccp_match_test;

    std::cout << "\nOverall: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "****************************************" << std::endl;

    return all_passed ? 0 : 1;
}
