/**
 * @file SemiSmoothNewton.h
 * @brief Semi-smooth Newton solver for SOCCP
 *
 * Implements the semi-smooth Newton method with line search for solving
 * second-order cone complementarity problems.
 */

#pragma once

#include "math/FischerBurmeister.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>
#include <iostream>
#include <vector>

namespace vde {

/**
 * @brief Solver statistics
 */
struct NewtonStats {
    int iterations;
    double final_residual;
    bool converged;
    std::vector<double> residual_history;

    NewtonStats()
        : iterations(0),
          final_residual(0.0),
          converged(false) {}
};

/**
 * @brief Semi-smooth Newton solver for SOCCP
 *
 * Solves the nonlinear equation: Phi_FB(x, y) = 0
 * subject to complementarity conditions
 */
class SemiSmoothNewton {
public:
    /**
     * @brief Constructor
     * @param max_iterations Maximum number of Newton iterations
     * @param tolerance Convergence tolerance
     * @param epsilon Smoothing parameter for FB function
     */
    SemiSmoothNewton(int max_iterations = 100,
                     double tolerance = 1e-10,
                     double epsilon = 1e-8)
        : max_iterations_(max_iterations),
          tolerance_(tolerance),
          epsilon_(epsilon) {}

    /**
     * @brief Solve the SOCCP using semi-smooth Newton method
     *
     * @param x Initial guess for x (will be modified)
     * @param y Initial guess for y (will be modified)
     * @param stats Output statistics
     * @return True if converged
     */
    bool solve(Vector4d& x, Vector4d& y, NewtonStats& stats) {
        stats = NewtonStats();
        stats.residual_history.clear();

        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Compute residual
            Vector4d residual = fischerBurmeister(x, y);
            double residual_norm = residual.norm();

            stats.iterations = iter + 1;
            stats.final_residual = residual_norm;
            stats.residual_history.push_back(residual_norm);

            // Check convergence
            if (residual_norm < tolerance_) {
                stats.converged = true;
                return true;
            }

            // Compute Jacobian
            Eigen::Matrix<double, 4, 8> J = fischerBurmeisterJacobian(x, y, epsilon_);

            // Extract dPhi/dx and dPhi/dy
            Eigen::Matrix4d dPhi_dx = J.block<4, 4>(0, 0);
            Eigen::Matrix4d dPhi_dy = J.block<4, 4>(0, 4);

            // For SOCCP, we typically have y = F(x) for some function F
            // Here we solve the simplified system by treating x and y as independent
            // In practice, y is often related to x through the physics (e.g., y = g - M*x)

            // Solve for Newton direction: J * [dx; dy] = -residual
            // We use a simplified approach: solve for dx and dy separately

            // For now, let's use a damped update
            // In a full implementation, we would solve the full 4x8 system with constraints

            // Simplified: update x and y using pseudo-inverse
            Eigen::Matrix<double, 8, 4> J_pinv = J.completeOrthogonalDecomposition().pseudoInverse();
            Eigen::Matrix<double, 8, 1> delta = -J_pinv * residual;

            Vector4d dx = delta.head<4>();
            Vector4d dy = delta.tail<4>();

            // Line search
            double alpha = lineSearch(x, y, dx, dy, residual_norm);

            // Update
            x += alpha * dx;
            y += alpha * dy;
        }

        stats.converged = false;
        return false;
    }

    /**
     * @brief Solve with complementarity constraints
     *
     * This version handles the case where y = g - M*x (linear complementarity)
     *
     * @param x Initial guess for x
     * @param g Constant vector
     * @param M Matrix in the constraint y = g - M*x
     * @param stats Output statistics
     * @return True if converged
     */
    bool solveLCP(Vector4d& x,
                  const Vector4d& g,
                  const Eigen::Matrix4d& M,
                  NewtonStats& stats) {
        stats = NewtonStats();
        stats.residual_history.clear();

        Vector4d y = g - M * x;

        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Update y based on current x
            y = g - M * x;

            // Compute residual
            Vector4d residual = fischerBurmeister(x, y);
            double residual_norm = residual.norm();

            stats.iterations = iter + 1;
            stats.final_residual = residual_norm;
            stats.residual_history.push_back(residual_norm);

            // Check convergence
            if (residual_norm < tolerance_) {
                stats.converged = true;
                return true;
            }

            // Compute Jacobian with respect to x only
            // Phi(x) = FB(x, g - M*x)
            // dPhi/dx = dPhi/dx - dPhi/dy * M

            Eigen::Matrix<double, 4, 8> J_full = fischerBurmeisterJacobian(x, y, epsilon_);
            Eigen::Matrix4d dPhi_dx = J_full.block<4, 4>(0, 0);
            Eigen::Matrix4d dPhi_dy = J_full.block<4, 4>(0, 4);

            Eigen::Matrix4d J = dPhi_dx - dPhi_dy * M;

            // Solve for Newton direction
            Vector4d dx = -J.completeOrthogonalDecomposition().solve(residual);

            // Line search
            double alpha = lineSearchLCP(x, g, M, dx, residual_norm);

            // Update
            x += alpha * dx;
        }

        stats.converged = false;
        return false;
    }

    /**
     * @brief Simple damped Newton (fallback method)
     *
     * Uses simple damping without line search
     *
     * @param x Initial guess for x
     * @param y Initial guess for y
     * @param stats Output statistics
     * @return True if converged
     */
    bool solveDamped(Vector4d& x, Vector4d& y, NewtonStats& stats) {
        stats = NewtonStats();
        stats.residual_history.clear();

        double damping = 1.0;

        for (int iter = 0; iter < max_iterations_; ++iter) {
            // Compute residual
            Vector4d residual = fischerBurmeister(x, y);
            double residual_norm = residual.norm();

            stats.iterations = iter + 1;
            stats.final_residual = residual_norm;
            stats.residual_history.push_back(residual_norm);

            // Check convergence
            if (residual_norm < tolerance_) {
                stats.converged = true;
                return true;
            }

            // Compute Jacobian
            Eigen::Matrix<double, 4, 8> J = fischerBurmeisterJacobian(x, y, epsilon_);

            // Solve for Newton direction using least squares
            Eigen::Matrix<double, 8, 1> delta = -J.transpose() * (J * J.transpose()).inverse() * residual;

            // Apply damping
            Vector4d dx = damping * delta.head<4>();
            Vector4d dy = damping * delta.tail<4>();

            // Update
            x += dx;
            y += dy;

            // Adjust damping based on progress
            if (iter > 0) {
                double prev_residual = stats.residual_history[iter - 1];
                if (residual_norm > prev_residual) {
                    // Residual increased, reduce damping
                    damping *= 0.5;
                } else if (residual_norm < 0.5 * prev_residual) {
                    // Good progress, increase damping
                    damping = std::min(1.0, damping * 1.2);
                }
            }
        }

        stats.converged = false;
        return false;
    }

    // Getters and setters
    int maxIterations() const { return max_iterations_; }
    double tolerance() const { return tolerance_; }
    double epsilon() const { return epsilon_; }

    void setMaxIterations(int max_iter) { max_iterations_ = max_iter; }
    void setTolerance(double tol) { tolerance_ = tol; }
    void setEpsilon(double eps) { epsilon_ = eps; }

private:
    int max_iterations_;
    double tolerance_;
    double epsilon_;

    /**
     * @brief Line search for the full Newton method
     *
     * @param x Current x
     * @param y Current y
     * @param dx Newton direction for x
     * @param dy Newton direction for y
     * @param current_residual Current residual norm
     * @return Step size alpha
     */
    double lineSearch(const Vector4d& x,
                      const Vector4d& y,
                      const Vector4d& dx,
                      const Vector4d& dy,
                      double current_residual) {
        double alpha = 1.0;
        const double c = 1e-4;  // Armijo parameter
        const double rho = 0.5;  // Backtracking parameter
        const int max_line_search = 20;

        for (int i = 0; i < max_line_search; ++i) {
            Vector4d x_new = x + alpha * dx;
            Vector4d y_new = y + alpha * dy;

            Vector4d residual_new = fischerBurmeister(x_new, y_new);
            double residual_norm_new = residual_new.norm();

            // Armijo condition
            if (residual_norm_new < (1.0 - c * alpha) * current_residual) {
                return alpha;
            }

            alpha *= rho;
        }

        return alpha;
    }

    /**
     * @brief Line search for LCP
     *
     * @param x Current x
     * @param g Constant vector
     * @param M Constraint matrix
     * @param dx Newton direction
     * @param current_residual Current residual norm
     * @return Step size alpha
     */
    double lineSearchLCP(const Vector4d& x,
                         const Vector4d& g,
                         const Eigen::Matrix4d& M,
                         const Vector4d& dx,
                         double current_residual) {
        double alpha = 1.0;
        const double c = 1e-4;
        const double rho = 0.5;
        const int max_line_search = 20;

        for (int i = 0; i < max_line_search; ++i) {
            Vector4d x_new = x + alpha * dx;
            Vector4d y_new = g - M * x_new;

            Vector4d residual_new = fischerBurmeister(x_new, y_new);
            double residual_norm_new = residual_new.norm();

            if (residual_norm_new < (1.0 - c * alpha) * current_residual) {
                return alpha;
            }

            alpha *= rho;
        }

        return alpha;
    }
};

}  // namespace vde
