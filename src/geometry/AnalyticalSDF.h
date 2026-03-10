/**
 * @file AnalyticalSDF.h
 * @brief Analytical Signed Distance Field (SDF) interface and basic shape implementations
 *
 * Provides analytical distance fields and gradient calculations for spheres, boxes, and half-spaces
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>

namespace vde {

/**
 * @brief SDF base class interface
 *
 * All analytical SDF shapes must inherit from this class and implement distance field and gradient calculations
 */
class SDF {
public:
    virtual ~SDF() = default;

    /**
     * @brief Compute signed distance field value
     * @param x Query point coordinates
     * @return Distance value (positive outside, negative inside)
     */
    virtual double phi(const Eigen::Vector3d& x) const = 0;

    /**
     * @brief Compute distance field gradient (pointing outward, normalized)
     * @param x Query point coordinates
     * @return Normalized gradient vector
     */
    virtual Eigen::Vector3d gradient(const Eigen::Vector3d& x) const = 0;

    /**
     * @brief Compute both distance and gradient (efficiency optimization)
     * @param x Query point coordinates
     * @param out_phi Output distance value
     * @param out_grad Output gradient vector
     */
    virtual void phiAndGradient(const Eigen::Vector3d& x,
                                 double& out_phi,
                                 Eigen::Vector3d& out_grad) const {
        out_phi = phi(x);
        out_grad = gradient(x);
    }
};

/**
 * @brief Sphere SDF
 *
 * Analytical distance field for sphere: phi(x) = |x - center| - radius
 */
class SphereSDF : public SDF {
public:
    /**
     * @brief Constructor
     * @param center Sphere center position
     * @param radius Sphere radius (must be positive)
     */
    SphereSDF(const Eigen::Vector3d& center, double radius)
        : center_(center), radius_(radius) {
        if (radius <= 0) {
            throw std::invalid_argument("Radius must be positive");
        }
    }

    double phi(const Eigen::Vector3d& x) const override {
        return (x - center_).norm() - radius_;
    }

    Eigen::Vector3d gradient(const Eigen::Vector3d& x) const override {
        Eigen::Vector3d diff = x - center_;
        double dist = diff.norm();
        if (dist < 1e-12) {
            // At sphere center, gradient is undefined, return arbitrary direction
            return Eigen::Vector3d(1, 0, 0);
        }
        return diff / dist;  // Unit vector pointing outward
    }

    void phiAndGradient(const Eigen::Vector3d& x,
                         double& out_phi,
                         Eigen::Vector3d& out_grad) const override {
        Eigen::Vector3d diff = x - center_;
        double dist = diff.norm();
        out_phi = dist - radius_;
        if (dist < 1e-12) {
            out_grad = Eigen::Vector3d(1, 0, 0);
        } else {
            out_grad = diff / dist;
        }
    }

    const Eigen::Vector3d& center() const { return center_; }
    double radius() const { return radius_; }

private:
    Eigen::Vector3d center_;
    double radius_;
};

/**
 * @brief Axis-aligned box SDF
 *
 * Analytical distance field for axis-aligned box
 */
class BoxSDF : public SDF {
public:
    /**
     * @brief Constructor
     * @param min_corner Minimum corner coordinates
     * @param max_corner Maximum corner coordinates
     */
    BoxSDF(const Eigen::Vector3d& min_corner, const Eigen::Vector3d& max_corner)
        : min_corner_(min_corner), max_corner_(max_corner) {
        if ((max_corner - min_corner).minCoeff() <= 0) {
            throw std::invalid_argument("Box dimensions must be positive");
        }
    }

    double phi(const Eigen::Vector3d& x) const override {
        Eigen::Vector3d d = x - clamp(x, min_corner_, max_corner_);
        double d_norm = d.norm();
        Eigen::Vector3d max_dist = min_corner_ - x;
        Eigen::Vector3d max_dist2 = x - max_corner_;
        double inside_dist = std::max(max_dist.maxCoeff(), max_dist2.maxCoeff());
        if (d_norm > 1e-12) {
            return d_norm;
        }
        return inside_dist;
    }

    Eigen::Vector3d gradient(const Eigen::Vector3d& x) const override {
        Eigen::Vector3d clamped = clamp(x, min_corner_, max_corner_);
        Eigen::Vector3d d = x - clamped;
        double d_norm = d.norm();

        if (d_norm > 1e-12) {
            // Outside: gradient points outward
            return d / d_norm;
        } else {
            // Inside: find nearest face/edge
            Eigen::Vector3d to_min = x - min_corner_;
            Eigen::Vector3d to_max = max_corner_ - x;

            // Find minimum distance face
            double min_dist = std::numeric_limits<double>::max();
            Eigen::Vector3d grad = Eigen::Vector3d::Zero();

            // Check 6 faces
            if (to_min.x() < min_dist) { min_dist = to_min.x(); grad = Eigen::Vector3d(-1, 0, 0); }
            if (to_max.x() < min_dist) { min_dist = to_max.x(); grad = Eigen::Vector3d(1, 0, 0); }
            if (to_min.y() < min_dist) { min_dist = to_min.y(); grad = Eigen::Vector3d(0, -1, 0); }
            if (to_max.y() < min_dist) { min_dist = to_max.y(); grad = Eigen::Vector3d(0, 1, 0); }
            if (to_min.z() < min_dist) { min_dist = to_min.z(); grad = Eigen::Vector3d(0, 0, -1); }
            if (to_max.z() < min_dist) { min_dist = to_max.z(); grad = Eigen::Vector3d(0, 0, 1); }

            return grad;
        }
    }

    void phiAndGradient(const Eigen::Vector3d& x,
                         double& out_phi,
                         Eigen::Vector3d& out_grad) const override {
        Eigen::Vector3d clamped = clamp(x, min_corner_, max_corner_);
        Eigen::Vector3d d = x - clamped;
        double d_norm = d.norm();

        if (d_norm > 1e-12) {
            out_phi = d_norm;
            out_grad = d / d_norm;
        } else {
            // Inside calculation
            Eigen::Vector3d to_min = x - min_corner_;
            Eigen::Vector3d to_max = max_corner_ - x;

            double min_dist = std::numeric_limits<double>::max();
            out_grad = Eigen::Vector3d::Zero();

            if (to_min.x() < min_dist) { min_dist = to_min.x(); out_grad = Eigen::Vector3d(-1, 0, 0); }
            if (to_max.x() < min_dist) { min_dist = to_max.x(); out_grad = Eigen::Vector3d(1, 0, 0); }
            if (to_min.y() < min_dist) { min_dist = to_min.y(); out_grad = Eigen::Vector3d(0, -1, 0); }
            if (to_max.y() < min_dist) { min_dist = to_max.y(); out_grad = Eigen::Vector3d(0, 1, 0); }
            if (to_min.z() < min_dist) { min_dist = to_min.z(); out_grad = Eigen::Vector3d(0, 0, -1); }
            if (to_max.z() < min_dist) { min_dist = to_max.z(); out_grad = Eigen::Vector3d(0, 0, 1); }

            out_phi = -min_dist;
        }
    }

    const Eigen::Vector3d& minCorner() const { return min_corner_; }
    const Eigen::Vector3d& maxCorner() const { return max_corner_; }

    Eigen::Vector3d center() const {
        return (min_corner_ + max_corner_) * 0.5;
    }

    Eigen::Vector3d size() const {
        return max_corner_ - min_corner_;
    }

private:
    Eigen::Vector3d min_corner_;
    Eigen::Vector3d max_corner_;

    static Eigen::Vector3d clamp(const Eigen::Vector3d& x,
                                  const Eigen::Vector3d& min_val,
                                  const Eigen::Vector3d& max_val) {
        return x.cwiseMax(min_val).cwiseMin(max_val);
    }
};

/**
 * @brief Infinite plane (Half-space) SDF
 *
 * Half-space defined by normal and point on plane: phi(x) = n dot (x - p)
 * Normal points outward (positive distance region)
 */
class HalfSpaceSDF : public SDF {
public:
    /**
     * @brief Constructor
     * @param normal Plane normal (pointing outward, will be normalized)
     * @param point_on_plane A point on the plane
     */
    HalfSpaceSDF(const Eigen::Vector3d& normal, const Eigen::Vector3d& point_on_plane)
        : normal_(normal.normalized()), point_on_plane_(point_on_plane) {
        if (normal.norm() < 1e-12) {
            throw std::invalid_argument("Normal vector cannot be zero");
        }
    }

    double phi(const Eigen::Vector3d& x) const override {
        return normal_.dot(x - point_on_plane_);
    }

    Eigen::Vector3d gradient(const Eigen::Vector3d& x) const override {
        (void)x;  // Unused parameter
        return normal_;  // Half-space gradient is constant
    }

    void phiAndGradient(const Eigen::Vector3d& x,
                         double& out_phi,
                         Eigen::Vector3d& out_grad) const override {
        out_phi = normal_.dot(x - point_on_plane_);
        out_grad = normal_;
    }

    const Eigen::Vector3d& normal() const { return normal_; }
    const Eigen::Vector3d& pointOnPlane() const { return point_on_plane_; }

private:
    Eigen::Vector3d normal_;
    Eigen::Vector3d point_on_plane_;
};

/**
 * @brief Finite cylinder aligned with the local Y axis
 */
class CylinderSDF : public SDF {
public:
    CylinderSDF(double radius, double height)
        : radius_(radius), height_(height) {
        if (radius <= 0.0 || height <= 0.0) {
            throw std::invalid_argument("Cylinder dimensions must be positive");
        }
    }

    double phi(const Eigen::Vector3d& x) const override {
        const Eigen::Vector2d d(
            Eigen::Vector2d(x.x(), x.z()).norm() - radius_,
            std::abs(x.y()) - 0.5 * height_);

        const Eigen::Vector2d d_out = d.cwiseMax(0.0);
        return d_out.norm() + std::min(std::max(d.x(), d.y()), 0.0);
    }

    Eigen::Vector3d gradient(const Eigen::Vector3d& x) const override {
        return finiteDifferenceGradient(x);
    }

    void phiAndGradient(const Eigen::Vector3d& x,
                        double& out_phi,
                        Eigen::Vector3d& out_grad) const override {
        out_phi = phi(x);
        out_grad = finiteDifferenceGradient(x);
    }

    double radius() const { return radius_; }
    double height() const { return height_; }

private:
    double radius_;
    double height_;

    Eigen::Vector3d finiteDifferenceGradient(const Eigen::Vector3d& x) const {
        constexpr double h = 1e-6;
        Eigen::Vector3d grad;
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d x_plus = x;
            Eigen::Vector3d x_minus = x;
            x_plus(axis) += h;
            x_minus(axis) -= h;
            grad(axis) = (phi(x_plus) - phi(x_minus)) / (2.0 * h);
        }

        const double norm = grad.norm();
        if (norm < 1e-12) {
            return Eigen::Vector3d::UnitY();
        }
        return grad / norm;
    }
};

/**
 * @brief Right circular cone aligned with local Y axis
 *
 * The cone center is at the midpoint of the height, apex at +height/2,
 * base at -height/2.
 */
class ConeSDF : public SDF {
public:
    ConeSDF(double base_radius, double height)
        : base_radius_(base_radius), height_(height) {
        if (base_radius <= 0.0 || height <= 0.0) {
            throw std::invalid_argument("Cone dimensions must be positive");
        }
    }

    double phi(const Eigen::Vector3d& x) const override {
        const double half_height = 0.5 * height_;
        const double y_apex = half_height;
        const double y_base = -half_height;
        const double radial = Eigen::Vector2d(x.x(), x.z()).norm();

        if (x.y() > y_apex) {
            return std::sqrt(radial * radial + (x.y() - y_apex) * (x.y() - y_apex));
        }
        if (x.y() < y_base) {
            const double dr = std::max(0.0, radial - base_radius_);
            return std::sqrt(dr * dr + (y_base - x.y()) * (y_base - x.y()));
        }

        const double t = (y_apex - x.y()) / height_;
        const double radius_at_y = base_radius_ * t;
        const double side_distance = radial - radius_at_y;

        if (radial <= radius_at_y) {
            const double distance_to_side = -sideDistance(x);
            const double distance_to_apex = y_apex - x.y();
            const double distance_to_base = x.y() - y_base;
            return -std::min({distance_to_side, distance_to_apex, distance_to_base});
        }

        return sideDistance(x);
    }

    Eigen::Vector3d gradient(const Eigen::Vector3d& x) const override {
        return finiteDifferenceGradient(x);
    }

    void phiAndGradient(const Eigen::Vector3d& x,
                        double& out_phi,
                        Eigen::Vector3d& out_grad) const override {
        out_phi = phi(x);
        out_grad = finiteDifferenceGradient(x);
    }

    double baseRadius() const { return base_radius_; }
    double height() const { return height_; }

private:
    double base_radius_;
    double height_;

    double sideDistance(const Eigen::Vector3d& x) const {
        const double radial = Eigen::Vector2d(x.x(), x.z()).norm();
        const double k = base_radius_ / height_;
        return (radial + k * (x.y() - 0.5 * height_)) / std::sqrt(1.0 + k * k);
    }

    Eigen::Vector3d finiteDifferenceGradient(const Eigen::Vector3d& x) const {
        constexpr double h = 1e-6;
        Eigen::Vector3d grad;
        for (int axis = 0; axis < 3; ++axis) {
            Eigen::Vector3d x_plus = x;
            Eigen::Vector3d x_minus = x;
            x_plus(axis) += h;
            x_minus(axis) -= h;
            grad(axis) = (phi(x_plus) - phi(x_minus)) / (2.0 * h);
        }

        const double norm = grad.norm();
        if (norm < 1e-12) {
            return Eigen::Vector3d::UnitY();
        }
        return grad / norm;
    }
};

/**
 * @brief World-space wrapper for a local analytical SDF
 *
 * Implements phi(x; q) by transforming world-space queries back to
 * the body's local frame.
 */
class TransformedSDF : public SDF {
public:
    TransformedSDF(std::shared_ptr<const SDF> local_sdf,
                   const Eigen::Vector3d& position,
                   const Eigen::Quaterniond& orientation)
        : local_sdf_(std::move(local_sdf)),
          position_(position),
          orientation_(orientation.normalized()) {
        if (!local_sdf_) {
            throw std::invalid_argument("TransformedSDF requires a valid local SDF");
        }
    }

    double phi(const Eigen::Vector3d& x) const override {
        return local_sdf_->phi(worldToLocal(x));
    }

    Eigen::Vector3d gradient(const Eigen::Vector3d& x) const override {
        const Eigen::Vector3d local_grad = local_sdf_->gradient(worldToLocal(x));
        const Eigen::Vector3d world_grad = orientation_ * local_grad;
        const double norm = world_grad.norm();
        if (norm < 1e-12) {
            return Eigen::Vector3d::UnitY();
        }
        return world_grad / norm;
    }

    void phiAndGradient(const Eigen::Vector3d& x,
                        double& out_phi,
                        Eigen::Vector3d& out_grad) const override {
        Eigen::Vector3d local_grad = Eigen::Vector3d::Zero();
        local_sdf_->phiAndGradient(worldToLocal(x), out_phi, local_grad);
        out_grad = orientation_ * local_grad;
        const double norm = out_grad.norm();
        if (norm < 1e-12) {
            out_grad = Eigen::Vector3d::UnitY();
        } else {
            out_grad /= norm;
        }
    }

private:
    std::shared_ptr<const SDF> local_sdf_;
    Eigen::Vector3d position_;
    Eigen::Quaterniond orientation_;

    Eigen::Vector3d worldToLocal(const Eigen::Vector3d& x) const {
        return orientation_.conjugate() * (x - position_);
    }
};

}  // namespace vde
