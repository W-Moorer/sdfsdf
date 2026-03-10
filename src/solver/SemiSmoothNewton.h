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
#include <algorithm>
#include <functional>
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
    std::vector<double> step_history;

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
    using ResidualFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
    using JacobianFunction = std::function<Eigen::MatrixXd(const Eigen::VectorXd&)>;

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
        Eigen::VectorXd state(8);
        state << x, y;

        auto residual_fn = [this](const Eigen::VectorXd& current_state) {
            const Vector4d x_local = current_state.head<4>();
            const Vector4d y_local = current_state.tail<4>();
            const Vector4d residual = fischerBurmeister(x_local, y_local);
            return Eigen::VectorXd(residual);
        };

        auto jacobian_fn = [this](const Eigen::VectorXd& current_state) {
            const Vector4d x_local = current_state.head<4>();
            const Vector4d y_local = current_state.tail<4>();
            const Eigen::Matrix<double, 4, 8> J = fischerBurmeisterJacobian(x_local, y_local, epsilon_);
            return Eigen::MatrixXd(J);
        };

        const bool converged = solveSystem(state, residual_fn, jacobian_fn, stats);
        x = state.head<4>();
        y = state.tail<4>();
        return converged;
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
        Eigen::VectorXd state(4);
        state = x;

        auto residual_fn = [this, &g, &M](const Eigen::VectorXd& current_state) {
            const Vector4d x_local = current_state;
            const Vector4d y_local = g - M * x_local;
            const Vector4d residual = fischerBurmeister(x_local, y_local);
            return Eigen::VectorXd(residual);
        };

        auto jacobian_fn = [this, &g, &M](const Eigen::VectorXd& current_state) {
            const Vector4d x_local = current_state;
            const Vector4d y_local = g - M * x_local;
            const Eigen::Matrix<double, 4, 8> J_full =
                fischerBurmeisterJacobian(x_local, y_local, epsilon_);
            const Eigen::Matrix4d dPhi_dx = J_full.block<4, 4>(0, 0);
            const Eigen::Matrix4d dPhi_dy = J_full.block<4, 4>(0, 4);
            const Eigen::Matrix4d J = dPhi_dx - dPhi_dy * M;
            return Eigen::MatrixXd(J);
        };

        const bool converged = solveSystem(state, residual_fn, jacobian_fn, stats);
        x = state;
        return converged;
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
        return solve(x, y, stats);
    }

    /**
     * @brief Solve a general nonlinear system H(state) = 0
     *
     * @param state Initial guess and solution vector
     * @param residual_fn Residual evaluator
     * @param jacobian_fn Jacobian evaluator
     * @param stats Output statistics
     * @return True if converged
     */
    bool solveSystem(Eigen::VectorXd& state,
                     const ResidualFunction& residual_fn,
                     const JacobianFunction& jacobian_fn,
                     NewtonStats& stats) const {
        stats = NewtonStats();
        stats.residual_history.clear();
        stats.step_history.clear();

        for (int iter = 0; iter < max_iterations_; ++iter) {
            const Eigen::VectorXd residual = residual_fn(state);
            if (!residual.allFinite()) {
                break;
            }

            const double residual_norm = residual.norm();
            stats.iterations = iter + 1;
            stats.final_residual = residual_norm;
            stats.residual_history.push_back(residual_norm);

            if (residual_norm < tolerance_) {
                stats.converged = true;
                return true;
            }

            const Eigen::MatrixXd J = jacobian_fn(state);
            if (!J.allFinite()) {
                break;
            }

            const Eigen::VectorXd delta =
                -J.completeOrthogonalDecomposition().solve(residual);
            if (!delta.allFinite()) {
                break;
            }

            const double alpha = lineSearch(state, delta, residual_norm, residual_fn);
            if (alpha <= 1e-12) {
                break;
            }

            stats.step_history.push_back(alpha);
            state += alpha * delta;
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
    double lineSearch(const Eigen::VectorXd& state,
                      const Eigen::VectorXd& delta,
                      double current_residual,
                      const ResidualFunction& residual_fn) const {
        double alpha = 1.0;
        double best_alpha = 0.0;
        double best_residual = current_residual;
        const double c = 1e-4;
        const double rho = 0.5;
        const int max_line_search = 20;

        for (int i = 0; i < max_line_search; ++i) {
            const Eigen::VectorXd trial_state = state + alpha * delta;
            const Eigen::VectorXd trial_residual = residual_fn(trial_state);
            if (!trial_residual.allFinite()) {
                alpha *= rho;
                continue;
            }

            const double trial_norm = trial_residual.norm();
            if (trial_norm < best_residual) {
                best_residual = trial_norm;
                best_alpha = alpha;
            }

            if (trial_norm <= (1.0 - c * alpha) * current_residual ||
                trial_norm < tolerance_) {
                return alpha;
            }

            alpha *= rho;
        }

        return best_alpha;
    }
};

}  // namespace vde
