/**
 * @file FrictionPGSSolver.h
 * @brief Global torsion-free frictional contact baseline solved by projected Gauss-Seidel
 */

#pragma once

#include "../dynamics/ContactDynamics.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <algorithm>
#include <vector>

namespace vde {

/**
 * @brief Global frictional contact problem without torsion
 *
 * Discrete step:
 *   M (v - v_free) - h J_n^T lambda_n - h J_t^T lambda_t = 0
 *   0 <= lambda_n ⟂ J_n v + gamma / h * g_eq >= 0
 *   lambda_t = Proj_{||.|| <= mu lambda_n}(lambda_t - W_tt^{-1} J_t v)
 */
struct FrictionPGSProblem {
    Eigen::MatrixXd mass_matrix;
    Eigen::VectorXd free_velocity;
    Eigen::MatrixXd normal_jacobian;      ///< nc x nv
    Eigen::MatrixXd tangential_jacobian;  ///< 2nc x nv
    Eigen::VectorXd g_eq;
    Eigen::VectorXd friction_coefficients;
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
               tangential_jacobian.rows() == 2 * nc &&
               tangential_jacobian.cols() == nv &&
               friction_coefficients.size() == nc &&
               baumgarte_gamma.size() == nc &&
               time_step > 0.0;
    }
};

struct FrictionPGSStats {
    int iterations = 0;
    double final_residual = 0.0;
    bool converged = false;
    std::vector<double> residual_history;
};

struct FrictionPGSSolution {
    Eigen::VectorXd lambda_n;
    Eigen::VectorXd lambda_t;
    Eigen::VectorXd velocity;
    Eigen::VectorXd residual;
    double scaled_residual = 0.0;
    double complementarity_violation = 0.0;
    FrictionPGSStats stats;
};

class FrictionPGSSolver {
public:
    FrictionPGSSolver(int max_iterations = 200,
                      double tolerance = 1e-10)
        : max_iterations_(max_iterations),
          tolerance_(tolerance) {}

    bool solve(const FrictionPGSProblem& problem, FrictionPGSSolution& solution) const {
        return solve(problem, nullptr, solution);
    }

    bool solve(const FrictionPGSProblem& problem,
               const Eigen::VectorXd* initial_impulses,
               FrictionPGSSolution& solution) const {
        solution = FrictionPGSSolution();
        if (!problem.isValid()) {
            return false;
        }

        const int nc = problem.numContacts();
        if (nc == 0) {
            solution.lambda_n = Eigen::VectorXd::Zero(0);
            solution.lambda_t = Eigen::VectorXd::Zero(0);
            solution.velocity = problem.free_velocity;
            solution.residual = Eigen::VectorXd::Zero(0);
            solution.stats.converged = true;
            return true;
        }

        Eigen::LDLT<Eigen::MatrixXd> ldlt(problem.mass_matrix);
        if (ldlt.info() != Eigen::Success) {
            return false;
        }

        const Eigen::MatrixXd Minv_JnT = ldlt.solve(problem.normal_jacobian.transpose());
        const Eigen::MatrixXd Minv_JtT = ldlt.solve(problem.tangential_jacobian.transpose());

        Eigen::VectorXd lambda_n = Eigen::VectorXd::Zero(nc);
        Eigen::VectorXd lambda_t = Eigen::VectorXd::Zero(2 * nc);
        if (initial_impulses != nullptr &&
            initial_impulses->size() == 3 * nc &&
            initial_impulses->allFinite()) {
            lambda_n = initial_impulses->head(nc).cwiseMax(0.0);
            lambda_t = initial_impulses->tail(2 * nc);
        }

        Eigen::VectorXd velocity =
            problem.free_velocity +
            problem.time_step * Minv_JnT * lambda_n +
            problem.time_step * Minv_JtT * lambda_t;

        std::vector<double> normal_diagonal(nc, 0.0);
        std::vector<Eigen::Matrix2d> tangential_delassus(nc, Eigen::Matrix2d::Zero());
        for (int i = 0; i < nc; ++i) {
            normal_diagonal[i] =
                problem.time_step * (problem.normal_jacobian.row(i) * Minv_JnT.col(i))(0);
            tangential_delassus[i] =
                problem.time_step *
                problem.tangential_jacobian.block(2 * i, 0, 2, problem.velocityDofs()) *
                Minv_JtT.block(0, 2 * i, problem.velocityDofs(), 2);
            tangential_delassus[i] += 1e-12 * Eigen::Matrix2d::Identity();
        }

        for (int iter = 0; iter < max_iterations_; ++iter) {
            for (int i = 0; i < nc; ++i) {
                const double diag_n = normal_diagonal[i];
                if (diag_n > 1e-14) {
                    const double u_n =
                        (problem.normal_jacobian.row(i) * velocity)(0) + stabilization(problem, i);
                    const double lambda_n_new = std::max(0.0, lambda_n(i) - u_n / diag_n);
                    const double delta_n = lambda_n_new - lambda_n(i);
                    if (std::abs(delta_n) > 0.0) {
                        lambda_n(i) = lambda_n_new;
                        velocity += problem.time_step * Minv_JnT.col(i) * delta_n;
                    }
                }

                const Eigen::MatrixXd J_t_i =
                    problem.tangential_jacobian.block(2 * i, 0, 2, problem.velocityDofs());
                const Eigen::Vector2d u_t = J_t_i * velocity;
                Eigen::LDLT<Eigen::Matrix2d> local_ldlt(tangential_delassus[i]);
                Eigen::Vector2d candidate = lambda_t.segment<2>(2 * i);
                if (local_ldlt.info() == Eigen::Success) {
                    candidate -= local_ldlt.solve(u_t);
                } else {
                    candidate -= u_t;
                }

                const double radius =
                    std::max(0.0, problem.friction_coefficients(i) * lambda_n(i));
                const Eigen::Vector2d lambda_t_new = projectDisk(candidate, radius);
                const Eigen::Vector2d delta_t = lambda_t_new - lambda_t.segment<2>(2 * i);
                if (delta_t.squaredNorm() > 0.0) {
                    lambda_t.segment<2>(2 * i) = lambda_t_new;
                    velocity +=
                        problem.time_step *
                        Minv_JtT.block(0, 2 * i, problem.velocityDofs(), 2) *
                        delta_t;
                }
            }

            solution.stats.iterations = iter + 1;
            solution.residual = computeResidual(
                problem,
                velocity,
                lambda_n,
                lambda_t,
                tangential_delassus);
            solution.stats.final_residual = solution.residual.lpNorm<Eigen::Infinity>();
            solution.stats.residual_history.push_back(solution.stats.final_residual);
            if (solution.stats.final_residual < tolerance_) {
                solution.stats.converged = true;
                break;
            }
        }

        solution.lambda_n = lambda_n;
        solution.lambda_t = lambda_t;
        solution.velocity = velocity;
        if (solution.residual.size() == 0) {
            solution.residual = computeResidual(
                problem,
                velocity,
                lambda_n,
                lambda_t,
                tangential_delassus);
        }
        solution.stats.final_residual = solution.residual.lpNorm<Eigen::Infinity>();
        solution.scaled_residual = computeScaledResidual(
            problem,
            velocity,
            lambda_n,
            lambda_t);
        solution.complementarity_violation = solution.scaled_residual;
        solution.stats.converged = solution.stats.final_residual < tolerance_;
        return solution.stats.converged;
    }

    static Eigen::VectorXd packWarmStart(const Eigen::VectorXd& lambda_n,
                                         const Eigen::VectorXd& lambda_t) {
        Eigen::VectorXd packed(lambda_n.size() + lambda_t.size());
        packed << lambda_n, lambda_t;
        return packed;
    }

private:
    int max_iterations_;
    double tolerance_;

    static double stabilization(const FrictionPGSProblem& problem, int contact_id) {
        return (problem.baumgarte_gamma(contact_id) / problem.time_step) * problem.g_eq(contact_id);
    }

    static Eigen::Vector2d projectDisk(const Eigen::Vector2d& value, double radius) {
        if (radius <= 0.0) {
            return Eigen::Vector2d::Zero();
        }

        const double norm = value.norm();
        if (norm <= radius || norm <= 1e-16) {
            return value;
        }
        return (radius / norm) * value;
    }

    static Eigen::VectorXd computeResidual(
        const FrictionPGSProblem& problem,
        const Eigen::VectorXd& velocity,
        const Eigen::VectorXd& lambda_n,
        const Eigen::VectorXd& lambda_t,
        const std::vector<Eigen::Matrix2d>& tangential_delassus) {

        const int nc = problem.numContacts();
        Eigen::VectorXd residual = Eigen::VectorXd::Zero(3 * nc);
        for (int i = 0; i < nc; ++i) {
            const double u_n =
                (problem.normal_jacobian.row(i) * velocity)(0) + stabilization(problem, i);
            residual(i) = std::max({
                std::max(0.0, -lambda_n(i)),
                std::max(0.0, -u_n),
                std::abs(lambda_n(i) * u_n)
            });

            const Eigen::MatrixXd J_t_i =
                problem.tangential_jacobian.block(2 * i, 0, 2, problem.velocityDofs());
            const Eigen::Vector2d u_t = J_t_i * velocity;
            Eigen::LDLT<Eigen::Matrix2d> local_ldlt(tangential_delassus[i]);
            Eigen::Vector2d projected = lambda_t.segment<2>(2 * i);
            if (local_ldlt.info() == Eigen::Success) {
                projected = projectDisk(
                    lambda_t.segment<2>(2 * i) - local_ldlt.solve(u_t),
                    std::max(0.0, problem.friction_coefficients(i) * lambda_n(i)));
            } else {
                projected = projectDisk(
                    lambda_t.segment<2>(2 * i) - u_t,
                    std::max(0.0, problem.friction_coefficients(i) * lambda_n(i)));
            }
            residual.segment<2>(nc + 2 * i) = lambda_t.segment<2>(2 * i) - projected;
        }
        return residual;
    }

    static double computeScaledResidual(
        const FrictionPGSProblem& problem,
        const Eigen::VectorXd& velocity,
        const Eigen::VectorXd& lambda_n,
        const Eigen::VectorXd& lambda_t) {
        double scale = std::max(
            {
                1.0,
                velocity.size() > 0 ? velocity.lpNorm<Eigen::Infinity>() : 0.0,
                problem.free_velocity.size() > 0
                    ? problem.free_velocity.lpNorm<Eigen::Infinity>()
                    : 0.0,
                lambda_n.size() > 0 ? lambda_n.lpNorm<Eigen::Infinity>() : 0.0,
                lambda_t.size() > 0 ? lambda_t.lpNorm<Eigen::Infinity>() : 0.0
            });
        for (int i = 0; i < problem.numContacts(); ++i) {
            const double u_n =
                (problem.normal_jacobian.row(i) * velocity)(0) + stabilization(problem, i);
            const Eigen::Vector2d u_t =
                problem.tangential_jacobian.block(2 * i, 0, 2, problem.velocityDofs()) * velocity;
            scale = std::max(scale, std::abs(u_n));
            scale = std::max(scale, u_t.lpNorm<Eigen::Infinity>());
            scale = std::max(scale, std::abs(problem.friction_coefficients(i) * lambda_n(i)));
        }
        const Eigen::VectorXd raw_residual = computeResidual(
            problem,
            velocity,
            lambda_n,
            lambda_t,
            buildTangentialDelassus(problem));
        return raw_residual.lpNorm<Eigen::Infinity>() / scale;
    }

    static std::vector<Eigen::Matrix2d> buildTangentialDelassus(
        const FrictionPGSProblem& problem) {
        std::vector<Eigen::Matrix2d> delassus(
            static_cast<size_t>(problem.numContacts()),
            Eigen::Matrix2d::Identity());
        Eigen::LDLT<Eigen::MatrixXd> ldlt(problem.mass_matrix);
        if (ldlt.info() != Eigen::Success) {
            return delassus;
        }
        const Eigen::MatrixXd Minv_JtT = ldlt.solve(problem.tangential_jacobian.transpose());
        for (int i = 0; i < problem.numContacts(); ++i) {
            delassus[static_cast<size_t>(i)] =
                problem.time_step *
                problem.tangential_jacobian.block(2 * i, 0, 2, problem.velocityDofs()) *
                Minv_JtT.block(0, 2 * i, problem.velocityDofs(), 2);
            delassus[static_cast<size_t>(i)] += 1e-12 * Eigen::Matrix2d::Identity();
        }
        return delassus;
    }
};

}  // namespace vde
