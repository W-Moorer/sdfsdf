/**
 * @file SOCCPSolver.h
 * @brief Global semi-smooth Newton solver for the contact SOCCP
 */

#pragma once

#include "SemiSmoothNewton.h"
#include "../dynamics/ContactDynamics.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <vector>

namespace vde {

/**
 * @brief Global SOCCP problem
 *
 * State layout:
 * W = [v, lambda_n(1..m), lambda_t_tilde(1..m), s(1..m)]
 * where each contact contributes one scalar normal impulse,
 * one 3D scaled tangential-torsional impulse, and one slack scalar.
 */
struct SOCCPProblem {
    Eigen::MatrixXd mass_matrix;
    Eigen::VectorXd free_velocity;
    Eigen::MatrixXd normal_jacobian;      ///< m x nv
    Eigen::MatrixXd tangential_jacobian;  ///< 3m x nv, already scaled as \tilde{J}_t

    Eigen::VectorXd g_eq;
    Eigen::VectorXd friction_coefficients;
    Eigen::VectorXd baumgarte_gamma;
    Eigen::ArrayXi torsion_enabled;
    double time_step = 1.0;
    double epsilon = 1e-8;

    int velocityDofs() const {
        return static_cast<int>(free_velocity.size());
    }

    int numContacts() const {
        return static_cast<int>(g_eq.size());
    }

    int stateSize() const {
        return velocityDofs() + 5 * numContacts();
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
               (torsion_enabled.size() == 0 || torsion_enabled.size() == nc) &&
               time_step > 0.0 &&
               epsilon > 0.0;
    }
};

/**
 * @brief Global SOCCP solution
 */
struct SOCCPSolution {
    Eigen::VectorXd state;
    Eigen::VectorXd velocity;
    Eigen::VectorXd lambda_n;
    Eigen::VectorXd lambda_t;
    Eigen::VectorXd slack;
    Eigen::VectorXd residual;
    NewtonStats stats;
};

/**
 * @brief Global SOCCP solver
 */
class SOCCPSolver {
public:
    SOCCPSolver(int max_iterations = 100,
                double tolerance = 1e-10,
                double epsilon = 1e-8)
        : newton_(max_iterations, tolerance, epsilon) {}

    bool solve(const SOCCPProblem& problem, SOCCPSolution& solution) const {
        return solve(problem, nullptr, solution);
    }

    bool solve(const SOCCPProblem& problem,
               const Eigen::VectorXd* initial_state,
               SOCCPSolution& solution) const {
        if (!problem.isValid()) {
            return false;
        }

        Eigen::VectorXd state = initialGuess(problem, initial_state);
        auto residual_fn = [this, &problem](const Eigen::VectorXd& current_state) {
            return computeResidual(problem, current_state);
        };
        auto jacobian_fn = [this, &problem](const Eigen::VectorXd& current_state) {
            return computeJacobian(problem, current_state);
        };

        const bool converged = newton_.solveSystem(state, residual_fn, jacobian_fn, solution.stats);

        solution.state = state;
        unpackSolution(problem, state, solution);
        solution.residual = computeResidual(problem, state);
        solution.stats.final_residual = solution.residual.norm();
        solution.stats.converged = converged && solution.residual.norm() < newton_.tolerance();
        return solution.stats.converged;
    }

    Eigen::VectorXd computeResidual(const SOCCPProblem& problem,
                                    const Eigen::VectorXd& state) const {
        const int nv = problem.velocityDofs();
        const int nc = problem.numContacts();
        Eigen::VectorXd residual = Eigen::VectorXd::Zero(problem.stateSize());

        const Eigen::VectorXd v = state.head(nv);
        const Eigen::VectorXd lambda_n = state.segment(normalOffset(problem), nc);
        const Eigen::VectorXd lambda_t = state.segment(tangentOffset(problem), 3 * nc);
        const Eigen::VectorXd slack = state.segment(slackOffset(problem), nc);

        residual.head(nv) =
            problem.mass_matrix * (v - problem.free_velocity)
            - problem.time_step * problem.normal_jacobian.transpose() * lambda_n
            - problem.time_step * problem.tangential_jacobian.transpose() * lambda_t;

        for (int i = 0; i < nc; ++i) {
            const double u_n =
                (problem.normal_jacobian.row(i) * v)(0) + stabilization(problem, i);
            residual(normalOffset(problem) + i) =
                scalarFB(lambda_n(i), u_n, problem.epsilon);

            const Eigen::Vector3d u_t =
                problem.tangential_jacobian.block(3 * i, 0, 3, nv) * v;
            const int row = socOffset(problem) + 4 * i;
            if (torsionEnabled(problem, i)) {
                const Vector4d x = makeFrictionConeVector(
                    problem.friction_coefficients(i),
                    lambda_n(i),
                    lambda_t.segment<3>(3 * i));
                const Vector4d y = makeDualConeVector(slack(i), u_t);
                residual.segment<4>(row) = fischerBurmeister(x, y);
            } else {
                Vector4d x = Vector4d::Zero();
                Vector4d y = Vector4d::Zero();
                x(0) = problem.friction_coefficients(i) * lambda_n(i);
                x(1) = lambda_t(3 * i + 0);
                x(2) = lambda_t(3 * i + 1);
                y(0) = slack(i);
                y(1) = u_t(0);
                y(2) = u_t(1);

                const Vector4d fb = fischerBurmeister(x, y);
                residual.segment<3>(row) = fb.head<3>();
                residual(row + 3) = lambda_t(3 * i + 2);
            }
        }

        return residual;
    }

    Eigen::MatrixXd computeJacobian(const SOCCPProblem& problem,
                                    const Eigen::VectorXd& state) const {
        const int nv = problem.velocityDofs();
        const int nc = problem.numContacts();
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(problem.stateSize(), problem.stateSize());

        const Eigen::VectorXd v = state.head(nv);
        const Eigen::VectorXd lambda_n = state.segment(normalOffset(problem), nc);
        const Eigen::VectorXd lambda_t = state.segment(tangentOffset(problem), 3 * nc);
        const Eigen::VectorXd slack = state.segment(slackOffset(problem), nc);

        J.block(0, 0, nv, nv) = problem.mass_matrix;
        J.block(0, normalOffset(problem), nv, nc) =
            -problem.time_step * problem.normal_jacobian.transpose();
        J.block(0, tangentOffset(problem), nv, 3 * nc) =
            -problem.time_step * problem.tangential_jacobian.transpose();

        for (int i = 0; i < nc; ++i) {
            const double u_n =
                (problem.normal_jacobian.row(i) * v)(0) + stabilization(problem, i);
            const double denom = std::sqrt(
                lambda_n(i) * lambda_n(i) + u_n * u_n + problem.epsilon * problem.epsilon);
            const double dphi_dlambda = 1.0 - lambda_n(i) / denom;
            const double dphi_du = 1.0 - u_n / denom;

            J.block(normalOffset(problem) + i, 0, 1, nv) =
                dphi_du * problem.normal_jacobian.row(i);
            J(normalOffset(problem) + i, normalOffset(problem) + i) = dphi_dlambda;

            const Eigen::Vector3d u_t =
                problem.tangential_jacobian.block(3 * i, 0, 3, nv) * v;
            const int row = socOffset(problem) + 4 * i;
            if (torsionEnabled(problem, i)) {
                const Vector4d x = makeFrictionConeVector(
                    problem.friction_coefficients(i),
                    lambda_n(i),
                    lambda_t.segment<3>(3 * i));
                const Vector4d y = makeDualConeVector(slack(i), u_t);
                const Eigen::Matrix<double, 4, 8> J_fb =
                    fischerBurmeisterJacobian(x, y, problem.epsilon);
                const Eigen::Matrix4d dphi_dx = J_fb.block<4, 4>(0, 0);
                const Eigen::Matrix4d dphi_dy = J_fb.block<4, 4>(0, 4);

                J.block(row, 0, 4, nv) =
                    dphi_dy.block<4, 3>(0, 1) *
                    problem.tangential_jacobian.block(3 * i, 0, 3, nv);
                J.block(row, normalOffset(problem) + i, 4, 1) =
                    problem.friction_coefficients(i) * dphi_dx.col(0);
                J.block(row, tangentOffset(problem) + 3 * i, 4, 3) =
                    dphi_dx.block<4, 3>(0, 1);
                J.block(row, slackOffset(problem) + i, 4, 1) = dphi_dy.col(0);
            } else {
                Vector4d x = Vector4d::Zero();
                Vector4d y = Vector4d::Zero();
                x(0) = problem.friction_coefficients(i) * lambda_n(i);
                x(1) = lambda_t(3 * i + 0);
                x(2) = lambda_t(3 * i + 1);
                y(0) = slack(i);
                y(1) = u_t(0);
                y(2) = u_t(1);

                const Eigen::Matrix<double, 4, 8> J_fb =
                    fischerBurmeisterJacobian(x, y, problem.epsilon);
                const Eigen::Matrix4d dphi_dx = J_fb.block<4, 4>(0, 0);
                const Eigen::Matrix4d dphi_dy = J_fb.block<4, 4>(0, 4);

                J.block(row, 0, 3, nv) =
                    dphi_dy.block<3, 3>(0, 1) *
                    problem.tangential_jacobian.block(3 * i, 0, 3, nv);
                J.block(row, normalOffset(problem) + i, 3, 1) =
                    problem.friction_coefficients(i) * dphi_dx.block<3, 1>(0, 0);
                J.block(row, tangentOffset(problem) + 3 * i, 3, 2) =
                    dphi_dx.block<3, 2>(0, 1);
                J.block(row, slackOffset(problem) + i, 3, 1) =
                    dphi_dy.block<3, 1>(0, 0);
                J(row + 3, tangentOffset(problem) + 3 * i + 2) = 1.0;
            }
        }

        return J;
    }

    static SOCCPProblem fromConstraint(const Eigen::MatrixXd& mass_matrix,
                                       const Eigen::VectorXd& free_velocity,
                                       const ContactConstraint& constraint,
                                       double time_step,
                                       double friction_coefficient,
                                       double baumgarte_gamma = 0.0,
                                       double epsilon = 1e-8) {
        return fromConstraints(
            mass_matrix,
            free_velocity,
            std::vector<ContactConstraint>{constraint},
            time_step,
            std::vector<double>{friction_coefficient},
            std::vector<double>{baumgarte_gamma},
            epsilon);
    }

    static SOCCPProblem fromConstraints(const Eigen::MatrixXd& mass_matrix,
                                        const Eigen::VectorXd& free_velocity,
                                        const std::vector<ContactConstraint>& constraints,
                                        double time_step,
                                        const std::vector<double>& friction_coefficients,
                                        const std::vector<double>& baumgarte_gamma,
                                        double epsilon = 1e-8) {
        SOCCPProblem problem;
        const int nc = static_cast<int>(constraints.size());
        const int nv = static_cast<int>(free_velocity.size());

        problem.mass_matrix = mass_matrix;
        problem.free_velocity = free_velocity;
        problem.normal_jacobian = Eigen::MatrixXd::Zero(nc, nv);
        problem.tangential_jacobian = Eigen::MatrixXd::Zero(3 * nc, nv);
        problem.g_eq = Eigen::VectorXd::Zero(nc);
        problem.friction_coefficients = Eigen::VectorXd::Zero(nc);
        problem.baumgarte_gamma = Eigen::VectorXd::Zero(nc);
        problem.torsion_enabled = Eigen::ArrayXi::Ones(nc);
        problem.time_step = time_step;
        problem.epsilon = epsilon;

        for (int i = 0; i < nc; ++i) {
            problem.normal_jacobian.row(i) = constraints[i].J_n;
            problem.tangential_jacobian.block(3 * i, 0, 3, nv) = constraints[i].tangentialJacobian();
            problem.g_eq(i) = constraints[i].g_eq;
            problem.friction_coefficients(i) = friction_coefficients[i];
            problem.baumgarte_gamma(i) = baumgarte_gamma[i];
        }

        return problem;
    }

    static Vector4d makeFrictionConeVector(double friction_coefficient,
                                           double lambda_n,
                                           const Eigen::Vector3d& lambda_t) {
        Vector4d x;
        x(0) = friction_coefficient * lambda_n;
        x.tail<3>() = lambda_t;
        return x;
    }

    static Vector4d makeDualConeVector(double slack,
                                       const Eigen::Vector3d& tangential_velocity) {
        Vector4d y;
        y(0) = slack;
        y.tail<3>() = tangential_velocity;
        return y;
    }

    static bool isInsideSOC(const Vector4d& x, double tolerance = 1e-10) {
        return x(0) + tolerance >= x.tail<3>().norm();
    }

private:
    SemiSmoothNewton newton_;

    static int normalOffset(const SOCCPProblem& problem) {
        return problem.velocityDofs();
    }

    static int tangentOffset(const SOCCPProblem& problem) {
        return problem.velocityDofs() + problem.numContacts();
    }

    static int slackOffset(const SOCCPProblem& problem) {
        return problem.velocityDofs() + 4 * problem.numContacts();
    }

    static int socOffset(const SOCCPProblem& problem) {
        return problem.velocityDofs() + problem.numContacts();
    }

    static double stabilization(const SOCCPProblem& problem, int contact_id) {
        return (problem.baumgarte_gamma(contact_id) / problem.time_step) * problem.g_eq(contact_id);
    }

    static double scalarFB(double a, double b, double epsilon) {
        return a + b - std::sqrt(a * a + b * b + epsilon * epsilon);
    }

    static Eigen::VectorXd initialGuess(const SOCCPProblem& problem,
                                        const Eigen::VectorXd* initial_state) {
        const int nv = problem.velocityDofs();
        const int nc = problem.numContacts();
        if (initial_state != nullptr &&
            initial_state->size() == problem.stateSize() &&
            initial_state->allFinite()) {
            Eigen::VectorXd state = *initial_state;
            state.segment(normalOffset(problem), nc) =
                state.segment(normalOffset(problem), nc).cwiseMax(0.0);
            state.segment(slackOffset(problem), nc) =
                state.segment(slackOffset(problem), nc).cwiseMax(problem.epsilon);
            for (int i = 0; i < nc; ++i) {
                if (!torsionEnabled(problem, i)) {
                    state(tangentOffset(problem) + 3 * i + 2) = 0.0;
                }
            }
            return state;
        }

        Eigen::VectorXd state = Eigen::VectorXd::Zero(problem.stateSize());
        state.head(nv) = problem.free_velocity;

        for (int i = 0; i < nc; ++i) {
            const double u_n =
                (problem.normal_jacobian.row(i) * problem.free_velocity)(0) + stabilization(problem, i);
            state(normalOffset(problem) + i) = std::max(0.0, -u_n);

            const Eigen::Vector3d u_t =
                problem.tangential_jacobian.block(3 * i, 0, 3, nv) * problem.free_velocity;
            state.segment<3>(tangentOffset(problem) + 3 * i).setZero();
            state(slackOffset(problem) + i) = std::max(problem.epsilon, u_t.norm());
            if (!torsionEnabled(problem, i)) {
                state(tangentOffset(problem) + 3 * i + 2) = 0.0;
            }
        }

        return state;
    }

    static void unpackSolution(const SOCCPProblem& problem,
                               const Eigen::VectorXd& state,
                               SOCCPSolution& solution) {
        const int nv = problem.velocityDofs();
        const int nc = problem.numContacts();

        solution.velocity = state.head(nv);
        solution.lambda_n = state.segment(normalOffset(problem), nc);
        solution.lambda_t = state.segment(tangentOffset(problem), 3 * nc);
        solution.slack = state.segment(slackOffset(problem), nc);
    }

    static bool torsionEnabled(const SOCCPProblem& problem, int contact_id) {
        return problem.torsion_enabled.size() == 0 || problem.torsion_enabled(contact_id) != 0;
    }
};

}  // namespace vde
