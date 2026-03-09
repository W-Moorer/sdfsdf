/**
 * @file PenaltySolver.h
 * @brief Penalty method solver for contact dynamics (Baseline A)
 *
 * Simple penalty method: F = k * V_overlap
 * Used as a baseline comparison for the SOCCP method.
 */

#pragma once

#include "dynamics/RigidBody.h"
#include "dynamics/ContactDynamics.h"
#include "geometry/VolumetricIntegrator.h"
#include <vector>
#include <memory>

namespace vde {

/**
 * @brief Penalty contact parameters
 */
struct PenaltyParameters {
    double stiffness;           // Normal stiffness (force per unit volume)
    double damping;             // Damping coefficient
    double friction_coefficient; // Coulomb friction coefficient

    PenaltyParameters()
        : stiffness(1000.0),
          damping(10.0),
          friction_coefficient(0.3) {}
};

/**
 * @brief Penalty-based contact solver
 *
 * Computes contact forces based on penetration volume:
 * F_normal = k * V_overlap - c * v_normal
 * F_tangential = -mu * F_normal * sign(v_tangential)
 */
class PenaltySolver {
public:
    PenaltySolver(const PenaltyParameters& params = PenaltyParameters())
        : params_(params) {}

    /**
     * @brief Solve contact and apply forces to bodies
     *
     * @param body_a Body A
     * @param body_b Body B
     * @param contact Contact constraint
     * @param dt Time step
     */
    void solveContact(RigidBody& body_a,
                      RigidBody& body_b,
                      const ContactConstraint& contact,
                      double dt) {
        // Only apply force if penetrating
        if (contact.g_eq >= 0) {
            return;
        }

        // Compute penetration volume (approximated)
        double penetration_depth = -contact.g_eq;
        double contact_area = estimateContactArea(contact);
        double overlap_volume = penetration_depth * contact_area;

        // Get velocities at contact point
        SpatialVector v_a, v_b;
        v_a << body_a.state().linear_velocity, body_a.state().angular_velocity;
        v_b << body_b.state().linear_velocity, body_b.state().angular_velocity;

        // Compute relative velocity
        double v_n = contact.computeNormalVelocity(v_a, v_b);
        Eigen::Vector2d v_t = contact.computeTangentialVelocity(v_a, v_b);

        // Normal force (penalty + damping)
        double F_n = params_.stiffness * overlap_volume - params_.damping * v_n;
        F_n = std::max(0.0, F_n);  // Only repulsive

        // Tangential force (Coulomb friction)
        Eigen::Vector2d F_t = -params_.friction_coefficient * F_n * v_t.normalized();
        if (v_t.norm() < 1e-10) {
            F_t.setZero();  // Static friction
        }

        // Convert to world frame force and torque
        Eigen::Vector3d F_world = F_n * contact.contact.normal +
                                   F_t(0) * contact.contact.tangent1 +
                                   F_t(1) * contact.contact.tangent2;

        // Apply equal and opposite forces
        body_a.applyForceAtPoint(-F_world, contact.contact.position);
        body_b.applyForceAtPoint(F_world, contact.contact.position);
    }

    /**
     * @brief Solve multiple contacts
     */
    void solveContacts(std::vector<RigidBody>& bodies,
                       const std::vector<ContactConstraint>& contacts,
                       double dt) {
        for (const auto& contact : contacts) {
            // Find body indices (simplified - in practice use proper indexing)
            // This is a placeholder implementation
        }
    }

    /**
     * @brief Get parameters
     */
    const PenaltyParameters& parameters() const { return params_; }
    void setParameters(const PenaltyParameters& params) { params_ = params; }

private:
    PenaltyParameters params_;

    /**
     * @brief Estimate contact area from constraint
     */
    double estimateContactArea(const ContactConstraint& contact) const {
        // Simplified: use r_tau to estimate area
        // Area ~ pi * r_tau^2 for circular contact
        return M_PI * contact.r_tau * contact.r_tau;
    }
};

}  // namespace vde
