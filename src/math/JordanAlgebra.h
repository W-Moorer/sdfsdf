/**
 * @file JordanAlgebra.h
 * @brief Jordan Algebra for 4D vectors (Second-Order Cone)
 *
 * Implements Jordan multiplication and spectral decomposition for 4D vectors
 * used in the Fischer-Burmeister SOCCP formulation.
 */

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <stdexcept>

namespace vde {

/**
 * @brief 4D vector in the second-order cone (SOC)
 *
 * A 4D vector x = (x0, x1, x2, x3) where x0 is the scalar part
 * and (x1, x2, x3) is the vector part.
 */
using Vector4d = Eigen::Vector4d;

/**
 * @brief Jordan multiplication for 4D SOC vectors
 *
 * For x = (x0, x_bar) and y = (y0, y_bar):
 * x o y = (x^T y, x0*y_bar + y0*x_bar)
 *
 * @param x First 4D vector
 * @param y Second 4D vector
 * @return Jordan product x o y
 */
inline Vector4d jordanMultiply(const Vector4d& x, const Vector4d& y) {
    Vector4d result;
    double x0 = x(0);
    double y0 = y(0);
    Eigen::Vector3d x_bar = x.tail<3>();
    Eigen::Vector3d y_bar = y.tail<3>();

    // Scalar part: x^T y = x0*y0 + x_bar^T y_bar
    result(0) = x0 * y0 + x_bar.dot(y_bar);

    // Vector part: x0*y_bar + y0*x_bar
    result.tail<3>() = x0 * y_bar + y0 * x_bar;

    return result;
}

/**
 * @brief Compute the determinant of a 4D SOC vector
 *
 * det(x) = x0^2 - ||x_bar||^2
 *
 * @param x 4D vector
 * @return Determinant
 */
inline double socDeterminant(const Vector4d& x) {
    double x0 = x(0);
    double x_bar_norm_sq = x.tail<3>().squaredNorm();
    return x0 * x0 - x_bar_norm_sq;
}

/**
 * @brief Compute the spectral decomposition of a 4D SOC vector
 *
 * Any x in SOC can be written as:
 * x = lambda_1 * u_1 + lambda_2 * u_2
 *
 * where lambda_i are eigenvalues and u_i are eigenvectors (Jordan frame)
 *
 * @param x Input 4D vector
 * @param lambda1 First eigenvalue (output)
 * @param lambda2 Second eigenvalue (output)
 * @param u1 First eigenvector (output)
 * @param u2 Second eigenvector (output)
 */
inline void spectralDecomposition(const Vector4d& x,
                                   double& lambda1,
                                   double& lambda2,
                                   Vector4d& u1,
                                   Vector4d& u2) {
    double x0 = x(0);
    Eigen::Vector3d x_bar = x.tail<3>();
    double x_bar_norm = x_bar.norm();

    // Eigenvalues
    lambda1 = x0 + x_bar_norm;
    lambda2 = x0 - x_bar_norm;

    // Eigenvectors (Jordan frame)
    if (x_bar_norm < 1e-15) {
        // x_bar is zero, use standard basis
        u1 << 0.5, 0.0, 0.0, 0.0;
        u2 << 0.5, 0.0, 0.0, 0.0;
        u1(0) = 0.5;
        u2(0) = 0.5;
    } else {
        Eigen::Vector3d x_bar_unit = x_bar / x_bar_norm;
        u1 << 0.5, 0.5 * x_bar_unit(0), 0.5 * x_bar_unit(1), 0.5 * x_bar_unit(2);
        u2 << 0.5, -0.5 * x_bar_unit(0), -0.5 * x_bar_unit(1), -0.5 * x_bar_unit(2);
    }
}

/**
 * @brief Compute the square root of a 4D SOC vector
 *
 * x^{1/2} = sqrt(lambda_1) * u_1 + sqrt(lambda_2) * u_2
 *
 * @param x Input 4D vector (must be in SOC, i.e., x0 >= ||x_bar||)
 * @return Square root of x
 */
inline Vector4d socSqrt(const Vector4d& x) {
    double lambda1, lambda2;
    Vector4d u1, u2;
    spectralDecomposition(x, lambda1, lambda2, u1, u2);

    if (lambda1 < -1e-15 || lambda2 < -1e-15) {
        throw std::domain_error("Cannot compute square root: vector not in SOC");
    }

    // Ensure non-negative for numerical stability
    lambda1 = std::max(0.0, lambda1);
    lambda2 = std::max(0.0, lambda2);

    return std::sqrt(lambda1) * u1 + std::sqrt(lambda2) * u2;
}

/**
 * @brief Project a vector onto the SOC (second-order cone)
 *
 * SOC = {x | x0 >= ||x_bar||}
 *
 * @param x Input 4D vector
 * @return Projected vector
 */
inline Vector4d projectOntoSOC(const Vector4d& x) {
    double x0 = x(0);
    Eigen::Vector3d x_bar = x.tail<3>();
    double x_bar_norm = x_bar.norm();

    if (x0 >= x_bar_norm) {
        // Already in SOC
        return x;
    } else if (x0 <= -x_bar_norm) {
        // Project to origin
        return Vector4d::Zero();
    } else {
        // Project to boundary
        double scale = (x0 + x_bar_norm) / (2.0 * x_bar_norm);
        Vector4d result;
        result(0) = scale * x_bar_norm;
        result.tail<3>() = scale * x_bar;
        return result;
    }
}

/**
 * @brief Check if a vector is in the SOC interior
 *
 * @param x 4D vector
 * @param tolerance Numerical tolerance
 * @return True if x0 > ||x_bar|| + tolerance
 */
inline bool isInSOCInterior(const Vector4d& x, double tolerance = 1e-10) {
    return x(0) > x.tail<3>().norm() + tolerance;
}

/**
 * @brief Check if a vector is on the SOC boundary
 *
 * @param x 4D vector
 * @param tolerance Numerical tolerance
 * @return True if |x0 - ||x_bar||| < tolerance
 */
inline bool isOnSOCBoundary(const Vector4d& x, double tolerance = 1e-10) {
    return std::abs(x(0) - x.tail<3>().norm()) < tolerance;
}

}  // namespace vde
