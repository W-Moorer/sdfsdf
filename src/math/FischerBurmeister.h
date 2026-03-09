/**
 * @file FischerBurmeister.h
 * @brief Fischer-Burmeister function for Second-Order Cone Complementarity Problem (SOCCP)
 *
 * Implements the FB function and its generalized Jacobian for the SOCCP formulation.
 */

#pragma once

#include "JordanAlgebra.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <functional>

namespace vde {

/**
 * @brief Fischer-Burmeister function for SOC
 *
 * Phi_FB(x, y) = x + y - sqrt(x^2 + y^2)
 *
 * where x^2 = x o x (Jordan algebra)
 *
 * @param x First 4D vector
 * @param y Second 4D vector
 * @return FB function value
 */
inline Vector4d fischerBurmeister(const Vector4d& x, const Vector4d& y) {
    // Compute x^2 + y^2 using Jordan algebra
    Vector4d x_squared = jordanMultiply(x, x);
    Vector4d y_squared = jordanMultiply(y, y);
    Vector4d sum_squares = x_squared + y_squared;

    // Compute sqrt(x^2 + y^2)
    Vector4d sqrt_sum;
    try {
        sqrt_sum = socSqrt(sum_squares);
    } catch (const std::domain_error&) {
        // If not in SOC, project first
        sqrt_sum = socSqrt(projectOntoSOC(sum_squares));
    }

    return x + y - sqrt_sum;
}

/**
 * @brief Smoothed Fischer-Burmeister function with smoothing parameter epsilon
 *
 * Phi_FB^epsilon(x, y) = x + y - sqrt(x^2 + y^2 + epsilon^2 * e)
 *
 * where e = (1, 0, 0, 0) is the identity element
 *
 * @param x First 4D vector
 * @param y Second 4D vector
 * @param epsilon Smoothing parameter (small positive value)
 * @return Smoothed FB function value
 */
inline Vector4d smoothedFischerBurmeister(const Vector4d& x, const Vector4d& y, double epsilon) {
    // Compute x^2 + y^2 + epsilon^2 * e
    Vector4d x_squared = jordanMultiply(x, x);
    Vector4d y_squared = jordanMultiply(y, y);
    Vector4d sum_squares = x_squared + y_squared;

    // Add epsilon^2 * e (e = (1, 0, 0, 0))
    sum_squares(0) += epsilon * epsilon;

    // Compute sqrt
    Vector4d sqrt_sum = socSqrt(sum_squares);

    return x + y - sqrt_sum;
}

/**
 * @brief Clarke generalized Jacobian of the FB function
 *
 * This is a 4x8 matrix that represents the subgradient of Phi_FB with respect to (x, y)
 *
 * @param x First 4D vector
 * @param y Second 4D vector
 * @param epsilon Smoothing parameter for numerical stability
 * @return 4x8 Jacobian matrix [dPhi/dx, dPhi/dy]
 */
inline Eigen::Matrix<double, 4, 8> fischerBurmeisterJacobian(
    const Vector4d& x,
    const Vector4d& y,
    double epsilon = 1e-8) {

    Eigen::Matrix<double, 4, 8> J;

    // Compute x^2 + y^2 + epsilon^2 * e
    Vector4d x_squared = jordanMultiply(x, x);
    Vector4d y_squared = jordanMultiply(y, y);
    Vector4d sum_squares = x_squared + y_squared;
    sum_squares(0) += epsilon * epsilon;

    // Compute w = sqrt(x^2 + y^2 + epsilon^2 * e)
    Vector4d w = socSqrt(sum_squares);

    // Compute L_w^{-1} (inverse of the Lyapunov operator L_w)
    // L_w is the matrix representation of the linear operator: z -> w o z

    double w0 = w(0);
    Eigen::Vector3d w_bar = w.tail<3>();
    double w_bar_norm = w_bar.norm();

    // For the inverse, we use the formula for the inverse of a Lyapunov operator
    // L_w^{-1} = (1/det(w)) * L_{w^c} where w^c is the conjugate
    double det_w = socDeterminant(w);

    // Build L_w matrix (4x4)
    Eigen::Matrix4d L_w;
    L_w.setZero();
    L_w(0, 0) = w0;
    L_w(0, 1) = w_bar(0);
    L_w(0, 2) = w_bar(1);
    L_w(0, 3) = w_bar(2);
    L_w(1, 0) = w_bar(0);
    L_w(2, 0) = w_bar(1);
    L_w(3, 0) = w_bar(2);
    L_w(1, 1) = w0;
    L_w(2, 2) = w0;
    L_w(3, 3) = w0;

    // Compute L_w^{-1}
    Eigen::Matrix4d L_w_inv;
    if (std::abs(det_w) > 1e-12) {
        L_w_inv = L_w.inverse();
    } else {
        // Use pseudo-inverse for numerical stability
        L_w_inv = L_w.completeOrthogonalDecomposition().pseudoInverse();
    }

    // Build L_x and L_y (Lyapunov operators)
    double x0 = x(0);
    double y0 = y(0);
    Eigen::Vector3d x_bar = x.tail<3>();
    Eigen::Vector3d y_bar = y.tail<3>();

    Eigen::Matrix4d L_x;
    L_x.setZero();
    L_x(0, 0) = x0;
    L_x(0, 1) = x_bar(0);
    L_x(0, 2) = x_bar(1);
    L_x(0, 3) = x_bar(2);
    L_x(1, 0) = x_bar(0);
    L_x(2, 0) = x_bar(1);
    L_x(3, 0) = x_bar(2);
    L_x(1, 1) = x0;
    L_x(2, 2) = x0;
    L_x(3, 3) = x0;

    Eigen::Matrix4d L_y;
    L_y.setZero();
    L_y(0, 0) = y0;
    L_y(0, 1) = y_bar(0);
    L_y(0, 2) = y_bar(1);
    L_y(0, 3) = y_bar(2);
    L_y(1, 0) = y_bar(0);
    L_y(2, 0) = y_bar(1);
    L_y(3, 0) = y_bar(2);
    L_y(1, 1) = y0;
    L_y(2, 2) = y0;
    L_y(3, 3) = y0;

    // Compute dPhi/dx = I - L_w^{-1} * L_x
    Eigen::Matrix4d dPhi_dx = Eigen::Matrix4d::Identity() - L_w_inv * L_x;

    // Compute dPhi/dy = I - L_w^{-1} * L_y
    Eigen::Matrix4d dPhi_dy = Eigen::Matrix4d::Identity() - L_w_inv * L_y;

    // Assemble full Jacobian [dPhi/dx, dPhi/dy]
    J.block<4, 4>(0, 0) = dPhi_dx;
    J.block<4, 4>(0, 4) = dPhi_dy;

    return J;
}

/**
 * @brief Verify FB function properties
 *
 * Check that Phi_FB(x, y) = 0 iff x in SOC, y in SOC, x^T y = 0 (complementarity)
 *
 * @param x First 4D vector
 * @param y Second 4D vector
 * @return True if FB function is zero (within tolerance)
 */
inline bool checkFBZero(const Vector4d& x, const Vector4d& y, double tolerance = 1e-10) {
    Vector4d phi = fischerBurmeister(x, y);
    return phi.norm() < tolerance;
}

/**
 * @brief Natural residual function for SOC
 *
 * R(x, y) = x - proj_SOC(x - y)
 *
 * This is another merit function for SOCCP
 *
 * @param x First 4D vector
 * @param y Second 4D vector
 * @return Natural residual
 */
inline Vector4d naturalResidual(const Vector4d& x, const Vector4d& y) {
    Vector4d diff = x - y;
    Vector4d proj = projectOntoSOC(diff);
    return x - proj;
}

}  // namespace vde
