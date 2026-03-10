/**
 * @file NormalLCPSolver.h
 * @brief Global frictionless normal-contact LCP baseline solver
 */

#pragma once

#include "../dynamics/ContactDynamics.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <algorithm>
#include <vector>

namespace vde {

/**
 * @brief Global normal-contact LCP problem
 *
 * Frictionless discrete contact step:
 *   M (v - v_free) - h J_n^T lambda = 0
 *   0 <= lambda \perp J_n v + gamma/h * g_eq >= 0
 */
struct NormalLCPProblem {
    Eigen::MatrixXd mass_matrix;
    Eigen::VectorXd free_velocity;
    Eigen::MatrixXd normal_jacobian;
    Eigen::VectorXd g_eq;
    Eigen::VectorXd baumgarte_gamma;
    double time_step = 1.0;

    int velocityDofs() const {
        return static_cast<int>(free_velocity.size());
    }

    int numContacts() const {
        return static_cast<int>(g_eq.size());
    }

    bool isValid() const {
        const int nv = velocityDofs();
        const int nc = numContacts();
        return nv > 0 &&
               nc >= 0 &&
               mass_matrix.rows() == nv &&
               mass_matrix.cols() == nv &&
               normal_jacobian.rows() == nc &&
               normal_jacobian.cols() == nv &&
               baumgarte_gamma.size() == nc &&
               time_step > 0.0;
    }
};

/**
 * @brief Iteration statistics for the normal LCP solver
 */
struct NormalLCPStats {
    int iterations = 0;
    double final_residual = 0.0;
    bool converged = false;
    std::vector<double> residual_history;
};

/**
 * @brief Solution of the global normal-contact LCP
 */
struct NormalLCPSolution {
    Eigen::VectorXd lambda_n;
    Eigen::VectorXd velocity;
    Eigen::VectorXd complementarity;
    Eigen::VectorXd residual;
    Eigen::MatrixXd delassus;
    Eigen::VectorXd bias;
    NormalLCPStats stats;
};

/**
 * @brief Projected Gauss-Seidel solver for global frictionless contact LCPs
 */
class NormalLCPSolver {
public:
    NormalLCPSolver(int max_iterations = 200,
                    double tolerance = 1e-10,
                    double relaxation = 1.0)
        : max_iterations_(max_iterations),
          tolerance_(tolerance),
          relaxation_(relaxation) {}

    bool solve(const NormalLCPProblem& problem, NormalLCPSolution& solution) const {
        return solve(problem, nullptr, solution);
    }

    bool solve(const NormalLCPProblem& problem,
               const Eigen::VectorXd* initial_lambda,
               NormalLCPSolution& solution) const {
        solution = NormalLCPSolution();
        if (!problem.isValid()) {
            return false;
        }

        const int nc = problem.numContacts();
        if (nc == 0) {
            solution.lambda_n = Eigen::VectorXd::Zero(0);
            solution.velocity = problem.free_velocity;
            solution.complementarity = Eigen::VectorXd::Zero(0);
            solution.residual = Eigen::VectorXd::Zero(0);
            solution.delassus = Eigen::MatrixXd::Zero(0, 0);
            solution.bias = Eigen::VectorXd::Zero(0);
            solution.stats.converged = true;
            solution.stats.final_residual = 0.0;
            return true;
        }

        Eigen::LDLT<Eigen::MatrixXd> ldlt(problem.mass_matrix);
        if (ldlt.info() != Eigen::Success) {
            return false;
        }

        const Eigen::MatrixXd Minv_Jt = ldlt.solve(problem.normal_jacobian.transpose());
        const Eigen::MatrixXd delassus =
            problem.time_step * problem.normal_jacobian * Minv_Jt;
        const Eigen::VectorXd bias =
            problem.normal_jacobian * problem.free_velocity + stabilization(problem);

        Eigen::VectorXd lambda = Eigen::VectorXd::Zero(nc);
        if (initial_lambda != nullptr &&
            initial_lambda->size() == nc &&
            initial_lambda->allFinite()) {
            lambda = initial_lambda->cwiseMax(0.0);
        }

        for (int iter = 0; iter < max_iterations_; ++iter) {
            for (int i = 0; i < nc; ++i) {
                const double diag = delassus(i, i);
                if (diag <= 1e-14) {
                    lambda(i) = 0.0;
                    continue;
                }

                const double wi = bias(i) + delassus.row(i).dot(lambda);
                const double update = lambda(i) - relaxation_ * wi / diag;
                lambda(i) = std::max(0.0, update);
            }

            const Eigen::VectorXd w = bias + delassus * lambda;
            const Eigen::VectorXd residual = residualVector(lambda, w);
            const double residual_inf = residual.lpNorm<Eigen::Infinity>();

            solution.stats.iterations = iter + 1;
            solution.stats.final_residual = residual_inf;
            solution.stats.residual_history.push_back(residual_inf);

            if (residual_inf < tolerance_) {
                solution.stats.converged = true;
                break;
            }
        }

        solution.lambda_n = lambda;
        solution.delassus = delassus;
        solution.bias = bias;
        solution.complementarity = bias + delassus * lambda;
        solution.residual = residualVector(solution.lambda_n, solution.complementarity);
        solution.velocity = problem.free_velocity + problem.time_step * Minv_Jt * solution.lambda_n;
        solution.stats.final_residual = solution.residual.lpNorm<Eigen::Infinity>();
        solution.stats.converged = solution.stats.final_residual < tolerance_;
        return solution.stats.converged;
    }

    Eigen::VectorXd computeComplementarity(const NormalLCPProblem& problem,
                                           const Eigen::VectorXd& lambda_n) const {
        Eigen::LDLT<Eigen::MatrixXd> ldlt(problem.mass_matrix);
        const Eigen::MatrixXd Minv_Jt = ldlt.solve(problem.normal_jacobian.transpose());
        return problem.normal_jacobian * problem.free_velocity +
               stabilization(problem) +
               problem.time_step * problem.normal_jacobian * Minv_Jt * lambda_n;
    }

    static NormalLCPProblem fromConstraints(const Eigen::MatrixXd& mass_matrix,
                                            const Eigen::VectorXd& free_velocity,
                                            const std::vector<ContactConstraint>& constraints,
                                            double time_step,
                                            const std::vector<double>& baumgarte_gamma) {
        NormalLCPProblem problem;
        const int nc = static_cast<int>(constraints.size());
        const int nv = static_cast<int>(free_velocity.size());

        problem.mass_matrix = mass_matrix;
        problem.free_velocity = free_velocity;
        problem.normal_jacobian = Eigen::MatrixXd::Zero(nc, nv);
        problem.g_eq = Eigen::VectorXd::Zero(nc);
        problem.baumgarte_gamma = Eigen::VectorXd::Zero(nc);
        problem.time_step = time_step;

        for (int i = 0; i < nc; ++i) {
            problem.normal_jacobian.row(i) = constraints[i].J_n;
            problem.g_eq(i) = constraints[i].g_eq;
            problem.baumgarte_gamma(i) = baumgarte_gamma[i];
        }

        return problem;
    }

private:
    int max_iterations_;
    double tolerance_;
    double relaxation_;

    static Eigen::VectorXd stabilization(const NormalLCPProblem& problem) {
        return (problem.baumgarte_gamma.array() / problem.time_step) * problem.g_eq.array();
    }

    static Eigen::VectorXd residualVector(const Eigen::VectorXd& lambda_n,
                                          const Eigen::VectorXd& w) {
        Eigen::VectorXd residual(lambda_n.size());
        for (int i = 0; i < lambda_n.size(); ++i) {
            residual(i) = std::max({
                std::max(0.0, -lambda_n(i)),
                std::max(0.0, -w(i)),
                std::abs(lambda_n(i) * w(i))
            });
        }
        return residual;
    }
};

}  // namespace vde
