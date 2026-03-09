/**
 * @file VolumetricIntegrator.h
 * @brief Shape-aware volumetric integrator
 *
 * Uniform grid sampler based on AABB intersection, computes Lp-norm penetration depth integral
 * Outputs equivalent gap g_eq, torsion inertia radius r_tau, equivalent contact normal n_c, and contact center p_c
 */

#pragma once

#include "AnalyticalSDF.h"
#include "collision/SpatialHash.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <cmath>
#include <algorithm>

namespace vde {

/**
 * @brief Contact geometry feature structure
 *
 * Contains all contact geometric quantities computed by volumetric integration
 */
struct ContactGeometry {
    double g_eq;                    ///< Equivalent scalar gap (negative means penetration)
    double r_tau;                   ///< Torsion inertia radius
    Eigen::Vector3d n_c;            ///< Equivalent contact normal (unit vector)
    Eigen::Vector3d p_c;            ///< Contact center
    double volume;                  ///< Overlap volume
    Eigen::Vector3d weighted_centroid;  ///< Weighted centroid (used to compute contact center)

    ContactGeometry()
        : g_eq(0.0),
          r_tau(0.0),
          n_c(Eigen::Vector3d::UnitY()),
          p_c(Eigen::Vector3d::Zero()),
          volume(0.0),
          weighted_centroid(Eigen::Vector3d::Zero()) {}
};

// AABB is now defined in collision/SpatialHash.h

/**
 * @brief Volumetric integrator
 *
 * Computes contact geometry features between two SDFs using uniform grid sampling
 */
class VolumetricIntegrator {
public:
    /**
     * @brief Constructor
     * @param grid_resolution Grid resolution N (N x N x N per dimension)
     * @param p_norm Lp-norm p value (default 2.0)
     */
    VolumetricIntegrator(int grid_resolution = 32, double p_norm = 2.0)
        : grid_resolution_(grid_resolution),
          p_norm_(p_norm) {
        if (grid_resolution < 2) {
            throw std::invalid_argument("Grid resolution must be at least 2");
        }
        if (p_norm < 1.0) {
            throw std::invalid_argument("p-norm must be >= 1");
        }
    }

    /**
     * @brief Compute contact geometry features between two SDFs
     *
     * Performs volumetric integration in overlap region to compute equivalent gap, torsion radius, contact normal and center
     *
     * @param sdf_a First SDF (object A)
     * @param sdf_b Second SDF (object B)
     * @param aabb_a Bounding box of object A
     * @param aabb_b Bounding box of object B
     * @return ContactGeometry Contact geometry features
     */
    ContactGeometry computeContactGeometry(
        const SDF& sdf_a,
        const SDF& sdf_b,
        const AABB& aabb_a,
        const AABB& aabb_b) const {

        // Compute AABB intersection
        AABB intersection_aabb = aabb_a.intersection(aabb_b);

        // If no intersection, return zero contact
        if (intersection_aabb.isEmpty()) {
            return ContactGeometry();
        }

        // Expand bounding box to include possible penetration region
        // Use larger edge length as expansion amount
        double max_size = aabb_a.size().maxCoeff();
        max_size = std::max(max_size, aabb_b.size().maxCoeff());
        intersection_aabb.expand(max_size * 0.1);

        // Perform grid sampling in intersection region
        return integrateOverRegion(sdf_a, sdf_b, intersection_aabb);
    }

    /**
     * @brief Integrate over specified region to compute contact geometry
     */
    ContactGeometry integrateOverRegion(
        const SDF& sdf_a,
        const SDF& sdf_b,
        const AABB& region) const {

        ContactGeometry result;

        Eigen::Vector3d region_size = region.size();
        double cell_volume = 1.0;

        // Compute step size for each dimension
        double dx = region_size.x() / (grid_resolution_ - 1);
        double dy = region_size.y() / (grid_resolution_ - 1);
        double dz = region_size.z() / (grid_resolution_ - 1);

        cell_volume = dx * dy * dz;

        // Variables for accumulating integrals
        double integral_phi_p = 0.0;        // integral phi^p dV
        double integral_weight = 0.0;       // Weight integral for normalization
        Eigen::Vector3d integral_position(0, 0, 0);  // For computing contact center
        Eigen::Vector3d integral_normal(0, 0, 0);    // For computing equivalent normal

        // Sample point weights (for Lp-norm)
        std::vector<double> phi_samples;
        std::vector<Eigen::Vector3d> position_samples;
        std::vector<Eigen::Vector3d> normal_samples;
        std::vector<double> weight_samples;

        // Traverse grid
        for (int i = 0; i < grid_resolution_; ++i) {
            for (int j = 0; j < grid_resolution_; ++j) {
                for (int k = 0; k < grid_resolution_; ++k) {
                    // Compute sample point position
                    Eigen::Vector3d x(
                        region.min.x() + i * dx,
                        region.min.y() + j * dy,
                        region.min.z() + k * dz
                    );

                    // Compute SDF values for both objects
                    double phi_a = sdf_a.phi(x);
                    double phi_b = sdf_b.phi(x);

                    // Compute penetration depth (negative when both objects are inside)
                    // Penetration depth = max(-phi_a, -phi_b, 0) in some combination
                    // Here we use volume overlap approach: overlap region when both SDFs are negative

                    if (phi_a < 0 && phi_b < 0) {
                        // In overlap region
                        double penetration_a = -phi_a;  // Object A penetration depth
                        double penetration_b = -phi_b;  // Object B penetration depth

                        // Use smaller penetration depth as effective penetration (conservative estimate)
                        double effective_penetration = std::min(penetration_a, penetration_b);

                        // Accumulate Lp-norm integral
                        if (p_norm_ > 0) {
                            integral_phi_p += std::pow(effective_penetration, p_norm_) * cell_volume;
                        }

                        // Accumulate weight and position
                        double weight = effective_penetration * cell_volume;
                        integral_weight += weight;
                        integral_position += x * weight;

                        // Compute normal (combination of two gradients)
                        Eigen::Vector3d grad_a = sdf_a.gradient(x);
                        Eigen::Vector3d grad_b = sdf_b.gradient(x);

                        // Contact normal is combination of two surface normals
                        // For object A contacting object B, normal direction is from A to B
                        Eigen::Vector3d contact_normal = (grad_b - grad_a).normalized();
                        if (contact_normal.norm() < 1e-10) {
                            contact_normal = Eigen::Vector3d(0, 1, 0);
                        }

                        integral_normal += contact_normal * weight;

                        // Store samples for subsequent torsion radius computation
                        phi_samples.push_back(effective_penetration);
                        position_samples.push_back(x);
                        normal_samples.push_back(contact_normal);
                        weight_samples.push_back(weight);
                    }
                }
            }
        }

        // Compute results
        if (integral_weight > 1e-15) {
            // Equivalent gap (Lp-norm)
            if (integral_phi_p > 0 && p_norm_ > 0) {
                result.g_eq = -std::pow(integral_phi_p, 1.0 / p_norm_);
            } else {
                result.g_eq = 0.0;
            }

            // Contact center
            result.p_c = integral_position / integral_weight;

            // Equivalent contact normal
            result.n_c = integral_normal.normalized();
            if (result.n_c.norm() < 1e-10) {
                result.n_c = Eigen::Vector3d(0, 1, 0);
            }

            // Compute torsion inertia radius
            result.r_tau = computeTorsionRadius(
                position_samples, normal_samples, weight_samples, result.p_c, result.n_c);

            result.volume = integral_weight;
            result.weighted_centroid = result.p_c;
        }

        return result;
    }

    /**
     * @brief Compute AABB bounding box for SDF
     *
     * Estimates bounding box based on SDF zero isosurface
     */
    static AABB estimateAABB(const SDF& sdf, const Eigen::Vector3d& center, double margin = 2.0) {
        // For general SDF, use center point and margin to estimate bounding box
        Eigen::Vector3d margin_vec(margin, margin, margin);
        return AABB(center - margin_vec, center + margin_vec);
    }

    /**
     * @brief Create AABB for sphere
     */
    static AABB sphereAABB(const SphereSDF& sphere, double margin = 0.1) {
        double r = sphere.radius();
        Eigen::Vector3d margin_vec(r + margin, r + margin, r + margin);
        return AABB(sphere.center() - margin_vec, sphere.center() + margin_vec);
    }

    /**
     * @brief Create AABB for box
     */
    static AABB boxAABB(const BoxSDF& box, double margin = 0.1) {
        Eigen::Vector3d margin_vec(margin, margin, margin);
        return AABB(box.minCorner() - margin_vec, box.maxCorner() + margin_vec);
    }

    /**
     * @brief Create AABB for half-space (finite region)
     */
    static AABB halfSpaceAABB(const HalfSpaceSDF& half_space,
                               const Eigen::Vector3d& center,
                               double extent = 5.0) {
        // Half-space is infinite, we limit to a finite region
        Eigen::Vector3d extent_vec(extent, extent, extent);
        return AABB(center - extent_vec, center + extent_vec);
    }

    int gridResolution() const { return grid_resolution_; }
    double pNorm() const { return p_norm_; }

    void setGridResolution(int resolution) {
        if (resolution < 2) {
            throw std::invalid_argument("Grid resolution must be at least 2");
        }
        grid_resolution_ = resolution;
    }

    void setPNorm(double p) {
        if (p < 1.0) {
            throw std::invalid_argument("p-norm must be >= 1");
        }
        p_norm_ = p;
    }

private:
    int grid_resolution_;
    double p_norm_;

    /**
     * @brief Compute torsion inertia radius
     *
     * r_tau = sqrt(integral ||(x - p_c) x n_c||^2 w(x) dV / integral w(x) dV)
     */
    double computeTorsionRadius(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<Eigen::Vector3d>& normals,
        const std::vector<double>& weights,
        const Eigen::Vector3d& p_c,
        const Eigen::Vector3d& n_c) const {

        if (positions.empty()) return 0.0;

        double integral_torque_squared = 0.0;
        double total_weight = 0.0;

        for (size_t i = 0; i < positions.size(); ++i) {
            // Compute lever arm
            Eigen::Vector3d lever_arm = positions[i] - p_c;

            // Compute torsion moment arm length (perpendicular distance to normal)
            // ||(x - p_c) x n_c|| = ||lever_arm|| * sin(theta)
            // This represents perpendicular distance from point to contact normal axis
            Eigen::Vector3d cross_prod = lever_arm.cross(n_c);
            double torque_arm = cross_prod.norm();

            double weight = weights[i];
            integral_torque_squared += torque_arm * torque_arm * weight;
            total_weight += weight;
        }

        if (total_weight < 1e-15) return 0.0;

        return std::sqrt(integral_torque_squared / total_weight);
    }
};

}  // namespace vde
