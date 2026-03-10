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
#include <cmath>
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
 * @brief Diagnostic output for a single penalty contact evaluation
 */
struct PenaltyContactResult {
    bool active = false;
    double penetration_depth = 0.0;
    double contact_area = 0.0;
    double overlap_volume = 0.0;
    double normal_velocity = 0.0;
    double normal_force = 0.0;
    Eigen::Vector2d tangential_velocity = Eigen::Vector2d::Zero();
    Eigen::Vector2d tangential_force = Eigen::Vector2d::Zero();
    Eigen::Vector3d world_force = Eigen::Vector3d::Zero();
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
        solveContactDetailed(body_a, body_b, contact, dt);
    }

    /**
     * @brief Solve contact and return diagnostic force information
     */
    PenaltyContactResult solveContactDetailed(RigidBody& body_a,
                                             RigidBody& body_b,
                                             const ContactConstraint& contact,
                                             double dt) {
        (void)dt;

        PenaltyContactResult result;
        if (contact.g_eq >= 0) {
            return result;
        }

        result.active = true;
        result.penetration_depth = -contact.g_eq;
        result.contact_area = estimateContactArea(contact);
        result.overlap_volume = result.penetration_depth * result.contact_area;

        SpatialVector v_a, v_b;
        v_a << body_a.state().linear_velocity, body_a.state().angular_velocity;
        v_b << body_b.state().linear_velocity, body_b.state().angular_velocity;

        result.normal_velocity = contact.computeNormalVelocity(v_a, v_b);
        result.tangential_velocity = contact.computeTangentialVelocity(v_a, v_b);

        result.normal_force =
            params_.stiffness * result.overlap_volume - params_.damping * result.normal_velocity;
        result.normal_force = std::max(0.0, result.normal_force);

        if (result.tangential_velocity.norm() >= 1e-10) {
            result.tangential_force =
                -params_.friction_coefficient * result.normal_force *
                result.tangential_velocity.normalized();
        }

        result.world_force =
            result.normal_force * contact.contact.normal +
            result.tangential_force(0) * contact.contact.tangent1 +
            result.tangential_force(1) * contact.contact.tangent2;

        // Positive contact normal is the repulsive action on body A.
        body_a.applyForceAtPoint(result.world_force, contact.contact.position);
        body_b.applyForceAtPoint(-result.world_force, contact.contact.position);
        return result;
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
    static constexpr double kPi = 3.14159265358979323846;

    PenaltyParameters params_;

    /**
     * @brief Estimate contact area from constraint
     */
    double estimateContactArea(const ContactConstraint& contact) const {
        // Simplified: use r_tau to estimate area
        // Area ~ pi * r_tau^2 for circular contact
        return kPi * contact.r_tau * contact.r_tau;
    }
};

}  // namespace vde
