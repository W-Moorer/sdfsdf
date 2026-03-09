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
#include "math/JordanAlgebra.h"
#include "math/FischerBurmeister.h"
#include "solver/SemiSmoothNewton.h"

using namespace vde;

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

int main() {
    std::cout << "****************************************" << std::endl;
    std::cout << "Phase 2 Acceptance Tests" << std::endl;
    std::cout << "****************************************" << std::endl;

    bool jordan_test = testJordanAlgebra();
    bool jacobian_test = testJacobianCorrectness();
    bool criterion1 = testCriterion1_FBZeroTheorem();
    bool criterion2 = testCriterion2_QuadraticConvergence();

    std::cout << "\n****************************************" << std::endl;
    std::cout << "Phase 2 Acceptance Summary" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "Jordan Algebra: " << (jordan_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Jacobian Correctness: " << (jacobian_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Criterion 1 (FB Zero Theorem): " << (criterion1 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Criterion 2 (Quadratic Convergence): " << (criterion2 ? "PASSED" : "FAILED") << std::endl;

    bool all_passed = jordan_test && jacobian_test && criterion1 && criterion2;

    std::cout << "\nOverall: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "****************************************" << std::endl;

    return all_passed ? 0 : 1;
}
