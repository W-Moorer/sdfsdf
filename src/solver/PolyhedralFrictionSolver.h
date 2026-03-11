/**
 * @file PolyhedralFrictionSolver.h
 * @brief Independent torsion-free polyhedral friction baseline
 *
 * Tangential friction is approximated by a finite set of friction rays in the
 * local tangent plane. The solver updates normal impulses and nonnegative ray
 * coefficients with projected Gauss-Seidel and enforces the per-contact cone
 * cap via Euclidean projection onto the simplex
 *
 *   beta >= 0, sum(beta) <= mu * lambda_n.
 *
 * This is intentionally independent from the SOCCP disk-cone formulation.
 */

#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace vde {

struct PolyhedralFrictionProblem {
    Eigen::MatrixXd mass_matrix;
    Eigen::VectorXd free_velocity;
    Eigen::MatrixXd normal_jacobian;      ///< nc x nv
    Eigen::MatrixXd tangential_jacobian;  ///< 2nc x nv
    Eigen::VectorXd g_eq;
    Eigen::VectorXd friction_coefficients;
    Eigen::VectorXd baumgarte_gamma;
    double time_step = 1.0;
    int num_directions = 4;

    int velocityDofs() const {
        return static_cast<int>(free_velocity.size());
    }

    int numContacts() const {
        return static_cast<int>(g_eq.size());
    }

    int numRayVariables() const {
        return numContacts() * num_directions;
    }

    bool isValid() const {
        const int nv = velocityDofs();
        const int nc = numContacts();
        return nv > 0 &&
               nc >= 0 &&
               num_directions >= 4 &&
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

struct PolyhedralFrictionStats {
    int iterations = 0;
    double final_residual = 0.0;
    bool converged = false;
    std::vector<double> residual_history;
};

struct PolyhedralFrictionSolution {
    Eigen::VectorXd lambda_n;
    Eigen::VectorXd beta;
    Eigen::VectorXd lambda_t;
    Eigen::VectorXd velocity;
    Eigen::VectorXd residual;
    double scaled_residual = 0.0;
    double complementarity_violation = 0.0;
    PolyhedralFrictionStats stats;
};

class PolyhedralFrictionSolver {
public:
    PolyhedralFrictionSolver(int max_iterations = 200,
                             double tolerance = 1e-10)
        : max_iterations_(max_iterations),
          tolerance_(tolerance) {}

    bool solve(const PolyhedralFrictionProblem& problem,
               PolyhedralFrictionSolution& solution) const {
        return solve(problem, nullptr, solution);
    }

    bool solve(const PolyhedralFrictionProblem& problem,
               const Eigen::VectorXd* initial_impulses,
               PolyhedralFrictionSolution& solution) const {
        solution = PolyhedralFrictionSolution();
        if (!problem.isValid()) {
            return false;
        }

        const int nc = problem.numContacts();
        const int nv = problem.velocityDofs();
        const int nd = problem.num_directions;
        if (nc == 0) {
            solution.lambda_n = Eigen::VectorXd::Zero(0);
            solution.beta = Eigen::VectorXd::Zero(0);
            solution.lambda_t = Eigen::VectorXd::Zero(0);
            solution.velocity = problem.free_velocity;
            solution.residual = Eigen::VectorXd::Zero(0);
            solution.stats.converged = true;
            return true;
        }

        Eigen::LDLT<Eigen::MatrixXd> mass_ldlt(problem.mass_matrix);
        if (mass_ldlt.info() != Eigen::Success) {
            return false;
        }

        const Eigen::MatrixXd directions = makeDirections(nd);
        const Eigen::MatrixXd ray_jacobian =
            buildRayJacobian(problem.tangential_jacobian, directions, nd);

        const Eigen::MatrixXd Minv_JnT =
            mass_ldlt.solve(problem.normal_jacobian.transpose());
        const Eigen::MatrixXd Minv_JdT =
            mass_ldlt.solve(ray_jacobian.transpose());

        Eigen::VectorXd lambda_n = Eigen::VectorXd::Zero(nc);
        Eigen::VectorXd beta = Eigen::VectorXd::Zero(nc * nd);
        if (initial_impulses != nullptr &&
            initial_impulses->size() == nc + nc * nd &&
            initial_impulses->allFinite()) {
            lambda_n = initial_impulses->head(nc).cwiseMax(0.0);
            beta = initial_impulses->tail(nc * nd).cwiseMax(0.0);
            for (int i = 0; i < nc; ++i) {
                beta.segment(i * nd, nd) =
                    projectSimplexCap(
                        beta.segment(i * nd, nd),
                        problem.friction_coefficients(i) * lambda_n(i));
            }
        }

        Eigen::VectorXd velocity =
            problem.free_velocity +
            problem.time_step * Minv_JnT * lambda_n +
            problem.time_step * Minv_JdT * beta;

        std::vector<double> normal_diagonal(nc, 0.0);
        std::vector<double> ray_diagonal(nc * nd, 0.0);
        for (int i = 0; i < nc; ++i) {
            normal_diagonal[i] =
                problem.time_step *
                (problem.normal_jacobian.row(i) * Minv_JnT.col(i))(0);
        }
        for (int ray = 0; ray < nc * nd; ++ray) {
            ray_diagonal[ray] =
                problem.time_step *
                (ray_jacobian.row(ray) * Minv_JdT.col(ray))(0);
        }

        for (int iter = 0; iter < max_iterations_; ++iter) {
            for (int i = 0; i < nc; ++i) {
                const double diag_n = normal_diagonal[i];
                if (diag_n > 1e-14) {
                    const double u_n =
                        (problem.normal_jacobian.row(i) * velocity)(0) +
                        stabilization(problem, i);
                    const double lambda_new = std::max(0.0, lambda_n(i) - u_n / diag_n);
                    const double delta = lambda_new - lambda_n(i);
                    if (std::abs(delta) > 0.0) {
                        lambda_n(i) = lambda_new;
                        velocity += problem.time_step * Minv_JnT.col(i) * delta;
                    }
                }

                const int ray_offset = i * nd;
                for (int local_ray = 0; local_ray < nd; ++local_ray) {
                    const int ray_id = ray_offset + local_ray;
                    const double diag = ray_diagonal[ray_id];
                    if (diag <= 1e-14) {
                        continue;
                    }

                    const double u_ray = (ray_jacobian.row(ray_id) * velocity)(0);
                    const double beta_new =
                        std::max(0.0, beta(ray_id) - u_ray / diag);
                    const double delta = beta_new - beta(ray_id);
                    if (std::abs(delta) > 0.0) {
                        beta(ray_id) = beta_new;
                        velocity += problem.time_step * Minv_JdT.col(ray_id) * delta;
                    }
                }

                const Eigen::VectorXd current_block = beta.segment(ray_offset, nd);
                const Eigen::VectorXd projected_block =
                    projectSimplexCap(
                        current_block,
                        problem.friction_coefficients(i) * lambda_n(i));
                const Eigen::VectorXd delta_block = projected_block - current_block;
                if (delta_block.squaredNorm() > 0.0) {
                    beta.segment(ray_offset, nd) = projected_block;
                    velocity +=
                        problem.time_step *
                        Minv_JdT.block(0, ray_offset, nv, nd) *
                        delta_block;
                }
            }

            solution.stats.iterations = iter + 1;
            solution.residual = computeResidual(
                problem,
                velocity,
                lambda_n,
                beta,
                ray_jacobian,
                ray_diagonal);
            solution.stats.final_residual = solution.residual.lpNorm<Eigen::Infinity>();
            solution.stats.residual_history.push_back(solution.stats.final_residual);
            if (solution.stats.final_residual < tolerance_) {
                solution.stats.converged = true;
                break;
            }
        }

        solution.lambda_n = lambda_n;
        solution.beta = beta;
        solution.lambda_t = reconstructTangentialImpulses(beta, directions, nc, nd);
        solution.velocity = velocity;
        if (solution.residual.size() == 0) {
            solution.residual = computeResidual(
                problem,
                velocity,
                lambda_n,
                beta,
                ray_jacobian,
                ray_diagonal);
        }
        solution.stats.final_residual = solution.residual.lpNorm<Eigen::Infinity>();
        solution.scaled_residual =
            computeScaledResidual(problem, velocity, lambda_n, beta, ray_jacobian, ray_diagonal);
        solution.complementarity_violation = solution.scaled_residual;
        solution.stats.converged = solution.stats.final_residual < tolerance_;
        return solution.stats.converged;
    }

    static Eigen::VectorXd packWarmStart(const Eigen::VectorXd& lambda_n,
                                         const Eigen::VectorXd& beta) {
        Eigen::VectorXd packed(lambda_n.size() + beta.size());
        packed << lambda_n, beta;
        return packed;
    }

private:
    int max_iterations_;
    double tolerance_;

    static double stabilization(const PolyhedralFrictionProblem& problem, int contact_id) {
        return (problem.baumgarte_gamma(contact_id) / problem.time_step) * problem.g_eq(contact_id);
    }

    static Eigen::MatrixXd makeDirections(int num_directions) {
        Eigen::MatrixXd directions(num_directions, 2);
        const double two_pi = 6.28318530717958647692;
        for (int i = 0; i < num_directions; ++i) {
            const double angle = two_pi * static_cast<double>(i) / static_cast<double>(num_directions);
            directions(i, 0) = std::cos(angle);
            directions(i, 1) = std::sin(angle);
        }
        return directions;
    }

    static Eigen::MatrixXd buildRayJacobian(const Eigen::MatrixXd& tangential_jacobian,
                                            const Eigen::MatrixXd& directions,
                                            int num_directions) {
        const int nc = static_cast<int>(tangential_jacobian.rows()) / 2;
        const int nv = static_cast<int>(tangential_jacobian.cols());
        Eigen::MatrixXd ray_jacobian = Eigen::MatrixXd::Zero(nc * num_directions, nv);
        for (int i = 0; i < nc; ++i) {
            const Eigen::RowVectorXd t1 = tangential_jacobian.row(2 * i);
            const Eigen::RowVectorXd t2 = tangential_jacobian.row(2 * i + 1);
            for (int local_ray = 0; local_ray < num_directions; ++local_ray) {
                ray_jacobian.row(i * num_directions + local_ray) =
                    directions(local_ray, 0) * t1 + directions(local_ray, 1) * t2;
            }
        }
        return ray_jacobian;
    }

    static Eigen::VectorXd reconstructTangentialImpulses(const Eigen::VectorXd& beta,
                                                         const Eigen::MatrixXd& directions,
                                                         int num_contacts,
                                                         int num_directions) {
        Eigen::VectorXd lambda_t = Eigen::VectorXd::Zero(2 * num_contacts);
        for (int i = 0; i < num_contacts; ++i) {
            lambda_t.segment<2>(2 * i) =
                directions.transpose() * beta.segment(i * num_directions, num_directions);
        }
        return lambda_t;
    }

    static Eigen::VectorXd projectSimplexCap(const Eigen::VectorXd& value, double cap) {
        if (value.size() == 0 || cap <= 0.0) {
            return Eigen::VectorXd::Zero(value.size());
        }

        Eigen::VectorXd clipped = value.cwiseMax(0.0);
        if (clipped.sum() <= cap) {
            return clipped;
        }

        std::vector<double> sorted(clipped.data(), clipped.data() + clipped.size());
        std::sort(sorted.begin(), sorted.end(), std::greater<double>());

        double cumulative = 0.0;
        double theta = 0.0;
        for (int i = 0; i < static_cast<int>(sorted.size()); ++i) {
            cumulative += sorted[i];
            const double candidate = (cumulative - cap) / static_cast<double>(i + 1);
            const double next = (i + 1 < static_cast<int>(sorted.size())) ? sorted[i + 1] : -1.0;
            if (candidate >= next) {
                theta = candidate;
                break;
            }
        }

        return (clipped.array() - theta).max(0.0);
    }

    static double naturalResidual(double x, double y) {
        return std::sqrt(x * x + y * y) - x - y;
    }

    static Eigen::VectorXd computeResidual(const PolyhedralFrictionProblem& problem,
                                           const Eigen::VectorXd& velocity,
                                           const Eigen::VectorXd& lambda_n,
                                           const Eigen::VectorXd& beta,
                                           const Eigen::MatrixXd& ray_jacobian,
                                           const std::vector<double>& ray_diagonal) {
        const int nc = problem.numContacts();
        const int nd = problem.num_directions;
        Eigen::VectorXd residual = Eigen::VectorXd::Zero(nc + nc * nd + nc);
        for (int i = 0; i < nc; ++i) {
            const double u_n =
                (problem.normal_jacobian.row(i) * velocity)(0) + stabilization(problem, i);
            residual(i) = std::abs(naturalResidual(lambda_n(i), u_n));

            const int ray_offset = i * nd;
            const Eigen::VectorXd beta_block = beta.segment(ray_offset, nd);
            const double cap = problem.friction_coefficients(i) * lambda_n(i);
            const Eigen::VectorXd ray_velocity = ray_jacobian.block(ray_offset, 0, nd, velocity.size()) * velocity;
            Eigen::VectorXd candidate = beta_block;
            for (int local_ray = 0; local_ray < nd; ++local_ray) {
                const double diag = std::max(ray_diagonal[ray_offset + local_ray], 1e-12);
                candidate(local_ray) -= ray_velocity(local_ray) / diag;
            }
            const Eigen::VectorXd projected = projectSimplexCap(candidate, cap);
            residual.segment(ray_offset + nc, nd) = beta_block - projected;
            residual(nc + nc * nd + i) =
                std::max(0.0, beta_block.sum() - cap);
        }
        return residual;
    }

    static double computeScaledResidual(const PolyhedralFrictionProblem& problem,
                                        const Eigen::VectorXd& velocity,
                                        const Eigen::VectorXd& lambda_n,
                                        const Eigen::VectorXd& beta,
                                        const Eigen::MatrixXd& ray_jacobian,
                                        const std::vector<double>& ray_diagonal) {
        const Eigen::VectorXd residual =
            computeResidual(problem, velocity, lambda_n, beta, ray_jacobian, ray_diagonal);
        const double residual_inf = residual.lpNorm<Eigen::Infinity>();
        const double scale =
            1.0 + velocity.lpNorm<Eigen::Infinity>() +
            lambda_n.lpNorm<Eigen::Infinity>() +
            beta.lpNorm<Eigen::Infinity>();
        return residual_inf / scale;
    }

};

}  // namespace vde
