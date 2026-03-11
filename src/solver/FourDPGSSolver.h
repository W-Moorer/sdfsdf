/**
 * @file FourDPGSSolver.h
 * @brief Independent 4D cone friction baseline solved by projected Gauss-Seidel
 */

#pragma once

#include "../math/FischerBurmeister.h"
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <vector>

namespace vde {

struct FourDPGSProblem {
    Eigen::MatrixXd mass_matrix;
    Eigen::VectorXd free_velocity;
    Eigen::MatrixXd normal_jacobian;      ///< nc x nv
    Eigen::MatrixXd tangential_jacobian;  ///< 3nc x nv
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
               tangential_jacobian.rows() == 3 * nc &&
               tangential_jacobian.cols() == nv &&
               friction_coefficients.size() == nc &&
               baumgarte_gamma.size() == nc &&
               time_step > 0.0;
    }
};

struct FourDPGSStats {
    int iterations = 0;
    double final_residual = 0.0;
    bool converged = false;
    std::vector<double> residual_history;
};

struct FourDPGSSolution {
    Eigen::VectorXd lambda_n;
    Eigen::VectorXd lambda_t;
    Eigen::VectorXd velocity;
    Eigen::VectorXd residual;
    double scaled_residual = 0.0;
    double complementarity_violation = 0.0;
    FourDPGSStats stats;
};

class FourDPGSSolver {
public:
    FourDPGSSolver(int max_iterations = 200,
                   double tolerance = 1e-10)
        : max_iterations_(max_iterations),
          tolerance_(tolerance) {}

    bool solve(const FourDPGSProblem& problem, FourDPGSSolution& solution) const {
        return solve(problem, nullptr, solution);
    }

    bool solve(const FourDPGSProblem& problem,
               const Eigen::VectorXd* initial_impulses,
               FourDPGSSolution& solution) const {
        solution = FourDPGSSolution();
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
        Eigen::VectorXd lambda_t = Eigen::VectorXd::Zero(3 * nc);
        if (initial_impulses != nullptr &&
            initial_impulses->size() == 4 * nc &&
            initial_impulses->allFinite()) {
            lambda_n = initial_impulses->head(nc).cwiseMax(0.0);
            lambda_t = initial_impulses->tail(3 * nc);
        }

        Eigen::VectorXd velocity =
            problem.free_velocity +
            problem.time_step * Minv_JnT * lambda_n +
            problem.time_step * Minv_JtT * lambda_t;

        std::vector<double> normal_diagonal(static_cast<size_t>(nc), 0.0);
        std::vector<Eigen::Matrix3d> tangential_delassus(
            static_cast<size_t>(nc),
            Eigen::Matrix3d::Zero());
        for (int i = 0; i < nc; ++i) {
            normal_diagonal[static_cast<size_t>(i)] =
                problem.time_step * (problem.normal_jacobian.row(i) * Minv_JnT.col(i))(0);
            tangential_delassus[static_cast<size_t>(i)] =
                problem.time_step *
                problem.tangential_jacobian.block(3 * i, 0, 3, problem.velocityDofs()) *
                Minv_JtT.block(0, 3 * i, problem.velocityDofs(), 3);
            tangential_delassus[static_cast<size_t>(i)] +=
                1e-12 * Eigen::Matrix3d::Identity();
        }

        for (int iter = 0; iter < max_iterations_; ++iter) {
            for (int i = 0; i < nc; ++i) {
                const double diag_n = normal_diagonal[static_cast<size_t>(i)];
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
                    problem.tangential_jacobian.block(3 * i, 0, 3, problem.velocityDofs());
                const Eigen::Vector3d u_t = J_t_i * velocity;
                Eigen::LDLT<Eigen::Matrix3d> local_ldlt(
                    tangential_delassus[static_cast<size_t>(i)]);
                Eigen::Vector3d candidate = lambda_t.segment<3>(3 * i);
                if (local_ldlt.info() == Eigen::Success) {
                    candidate -= local_ldlt.solve(u_t);
                } else {
                    candidate -= u_t;
                }

                const double radius =
                    std::max(0.0, problem.friction_coefficients(i) * lambda_n(i));
                const Eigen::Vector3d lambda_t_new = projectBall(candidate, radius);
                const Eigen::Vector3d delta_t =
                    lambda_t_new - lambda_t.segment<3>(3 * i);
                if (delta_t.squaredNorm() > 0.0) {
                    lambda_t.segment<3>(3 * i) = lambda_t_new;
                    velocity +=
                        problem.time_step *
                        Minv_JtT.block(0, 3 * i, problem.velocityDofs(), 3) *
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
        solution.scaled_residual = computeScaledResidual(problem, solution);
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

    static double stabilization(const FourDPGSProblem& problem, int contact_id) {
        return (problem.baumgarte_gamma(contact_id) / problem.time_step) * problem.g_eq(contact_id);
    }

    static Eigen::Vector3d projectBall(const Eigen::Vector3d& value, double radius) {
        if (radius <= 0.0) {
            return Eigen::Vector3d::Zero();
        }

        const double norm = value.norm();
        if (norm <= radius || norm <= 1e-16) {
            return value;
        }
        return (radius / norm) * value;
    }

    static Eigen::VectorXd computeResidual(
        const FourDPGSProblem& problem,
        const Eigen::VectorXd& velocity,
        const Eigen::VectorXd& lambda_n,
        const Eigen::VectorXd& lambda_t,
        const std::vector<Eigen::Matrix3d>& tangential_delassus) {
        const int nc = problem.numContacts();
        Eigen::VectorXd residual = Eigen::VectorXd::Zero(4 * nc);
        for (int i = 0; i < nc; ++i) {
            const double u_n =
                (problem.normal_jacobian.row(i) * velocity)(0) + stabilization(problem, i);
            residual(i) = std::max({
                std::max(0.0, -lambda_n(i)),
                std::max(0.0, -u_n),
                std::abs(lambda_n(i) * u_n)
            });

            const Eigen::MatrixXd J_t_i =
                problem.tangential_jacobian.block(3 * i, 0, 3, problem.velocityDofs());
            const Eigen::Vector3d u_t = J_t_i * velocity;
            Eigen::LDLT<Eigen::Matrix3d> local_ldlt(tangential_delassus[static_cast<size_t>(i)]);
            Eigen::Vector3d candidate = lambda_t.segment<3>(3 * i);
            if (local_ldlt.info() == Eigen::Success) {
                candidate -= local_ldlt.solve(u_t);
            } else {
                candidate -= u_t;
            }
            const Eigen::Vector3d projected = projectBall(
                candidate,
                std::max(0.0, problem.friction_coefficients(i) * lambda_n(i)));
            residual.segment<3>(nc + 3 * i) = lambda_t.segment<3>(3 * i) - projected;
        }
        return residual;
    }

    static double computeScaledResidual(const FourDPGSProblem& problem,
                                        const FourDPGSSolution& solution) {
        double scale = std::max(
            {
                1.0,
                solution.velocity.size() > 0
                    ? solution.velocity.lpNorm<Eigen::Infinity>()
                    : 0.0,
                problem.free_velocity.size() > 0
                    ? problem.free_velocity.lpNorm<Eigen::Infinity>()
                    : 0.0,
                solution.lambda_n.size() > 0
                    ? solution.lambda_n.lpNorm<Eigen::Infinity>()
                    : 0.0,
                solution.lambda_t.size() > 0
                    ? solution.lambda_t.lpNorm<Eigen::Infinity>()
                    : 0.0
            });
        const int nc = problem.numContacts();
        for (int i = 0; i < nc; ++i) {
            const double u_n =
                (problem.normal_jacobian.row(i) * solution.velocity)(0) + stabilization(problem, i);
            const Eigen::Vector3d u_t =
                problem.tangential_jacobian.block(3 * i, 0, 3, problem.velocityDofs()) *
                solution.velocity;
            scale = std::max(scale, std::abs(u_n));
            scale = std::max(scale, u_t.lpNorm<Eigen::Infinity>());
            scale = std::max(
                scale,
                std::abs(problem.friction_coefficients(i) * solution.lambda_n(i)));
        }
        const double raw = solution.residual.size() > 0
            ? solution.residual.lpNorm<Eigen::Infinity>()
            : 0.0;
        return raw / scale;
    }
};

}  // namespace vde
