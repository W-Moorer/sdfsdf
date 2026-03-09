/**
 * @file ContactDynamics.h
 * @brief Contact dynamics and kinematic Jacobian computation
 *
 * Computes kinematic Jacobians and generates generalized normal/tangential Jacobians
 * for 6-DOF rigid body contact.
 */

#pragma once

#include "RigidBody.h"
#include "../geometry/VolumetricIntegrator.h"
#include <Eigen/Core>

namespace vde {

/**
 * @brief Contact point information
 */
struct ContactPoint {
    Eigen::Vector3d position;           // Contact point in world frame
    Eigen::Vector3d normal;             // Contact normal (pointing from body A to B)
    Eigen::Vector3d tangent1;           // First tangent direction
    Eigen::Vector3d tangent2;           // Second tangent direction
    double penetration_depth;           // Penetration depth (negative if separated)

    ContactPoint()
        : position(Eigen::Vector3d::Zero()),
          normal(Eigen::Vector3d::UnitY()),
          tangent1(Eigen::Vector3d::UnitX()),
          tangent2(Eigen::Vector3d::UnitZ()),
          penetration_depth(0.0) {}

    /**
     * @brief Compute tangent directions from normal
     */
    void computeTangentDirections() {
        // Find a vector not parallel to normal
        Eigen::Vector3d ref = Eigen::Vector3d::UnitX();
        if (std::abs(normal.dot(ref)) > 0.9) {
            ref = Eigen::Vector3d::UnitY();
        }

        // First tangent
        tangent1 = normal.cross(ref).normalized();

        // Second tangent
        tangent2 = normal.cross(tangent1).normalized();
    }
};

/**
 * @brief Kinematic Jacobian for a contact point
 *
 * Maps generalized velocities to contact point velocities
 */
class ContactJacobian {
public:
    /**
     * @brief Compute kinematic Jacobian for single body contact
     *
     * J_kin * v = contact point velocity
     * where v = [v_linear; v_angular] is the 6D spatial velocity
     *
     * @param contact_point Contact point in world frame
     * @param body_center Body center of mass in world frame
     * @return 3x6 kinematic Jacobian
     */
    static Eigen::Matrix<double, 3, 6> computeKinematic(
        const Eigen::Vector3d& contact_point,
        const Eigen::Vector3d& body_center) {

        Eigen::Matrix<double, 3, 6> J;
        J.setZero();

        // Linear part: identity
        J.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

        // Angular part: -[r]x (cross product matrix)
        Eigen::Vector3d r = contact_point - body_center;
        J.block<3, 3>(0, 3) = -crossProductMatrix(r);

        return J;
    }

    /**
     * @brief Compute generalized normal Jacobian (1x6)
     *
     * J_n * v = normal velocity at contact point
     */
    static Eigen::Matrix<double, 1, 6> computeNormal(
        const Eigen::Vector3d& contact_point,
        const Eigen::Vector3d& body_center,
        const Eigen::Vector3d& normal) {

        Eigen::Matrix<double, 3, 6> J_kin = computeKinematic(contact_point, body_center);
        return normal.transpose() * J_kin;
    }

    /**
     * @brief Compute generalized tangential Jacobian (2x6)
     *
     * J_t * v = tangential velocity at contact point
     */
    static Eigen::Matrix<double, 2, 6> computeTangential(
        const Eigen::Vector3d& contact_point,
        const Eigen::Vector3d& body_center,
        const Eigen::Vector3d& tangent1,
        const Eigen::Vector3d& tangent2) {

        Eigen::Matrix<double, 3, 6> J_kin = computeKinematic(contact_point, body_center);
        Eigen::Matrix<double, 2, 6> J_t;
        J_t.row(0) = tangent1.transpose() * J_kin;
        J_t.row(1) = tangent2.transpose() * J_kin;
        return J_t;
    }

    /**
     * @brief Compute full 4D friction cone Jacobian (4x6)
     *
     * J_f * v = [normal_velocity; tangential_velocity1; tangential_velocity2; torsion]
     *
     * The torsion component is computed based on the contact geometry's r_tau
     */
    static Eigen::Matrix<double, 4, 6> computeFrictionCone(
        const Eigen::Vector3d& contact_point,
        const Eigen::Vector3d& body_center,
        const ContactPoint& contact,
        double r_tau) {

        Eigen::Matrix<double, 4, 6> J_f;
        J_f.setZero();

        // Normal component
        J_f.row(0) = computeNormal(contact_point, body_center, contact.normal);

        // Tangential components
        Eigen::Matrix<double, 2, 6> J_t = computeTangential(
            contact_point, body_center, contact.tangent1, contact.tangent2);
        J_f.row(1) = J_t.row(0);
        J_f.row(2) = J_t.row(1);

        // Torsion component (approximated as angular velocity about normal)
        // J_torsion = [0, 0, 0, n_x * r_tau, n_y * r_tau, n_z * r_tau]
        J_f.block<1, 3>(3, 3) = r_tau * contact.normal.transpose();

        return J_f;
    }

    /**
     * @brief Compute two-body contact Jacobian (for contact between two bodies)
     *
     * @param contact Contact point information
     * @param center_a Body A center
     * @param center_b Body B center
     * @return 3x12 Jacobian [J_a, -J_b] (relative velocity at contact)
     */
    static Eigen::Matrix<double, 3, 12> computeTwoBodyKinematic(
        const ContactPoint& contact,
        const Eigen::Vector3d& center_a,
        const Eigen::Vector3d& center_b) {

        Eigen::Matrix<double, 3, 6> J_a = computeKinematic(contact.position, center_a);
        Eigen::Matrix<double, 3, 6> J_b = computeKinematic(contact.position, center_b);

        Eigen::Matrix<double, 3, 12> J;
        J.block<3, 6>(0, 0) = J_a;
        J.block<3, 6>(0, 6) = -J_b;

        return J;
    }

private:
    /**
     * @brief Compute cross product matrix
     *
     * [a]x * b = a x b
     */
    static Eigen::Matrix3d crossProductMatrix(const Eigen::Vector3d& a) {
        Eigen::Matrix3d ax;
        ax <<     0, -a(2),  a(1),
              a(2),     0, -a(0),
             -a(1),  a(0),     0;
        return ax;
    }
};

/**
 * @brief Contact constraint for a pair of bodies
 */
struct ContactConstraint {
    ContactPoint contact;

    // Jacobians
    Eigen::Matrix<double, 1, 12> J_n;   // Normal Jacobian (1x12 for two bodies)
    Eigen::Matrix<double, 4, 12> J_f;   // Friction cone Jacobian (4x12)

    // Contact geometry
    double g_eq;                        // Equivalent gap
    double r_tau;                       // Torsion radius

    ContactConstraint() : g_eq(0.0), r_tau(0.0) {
        J_n.setZero();
        J_f.setZero();
    }

    /**
     * @brief Compute Jacobians from contact geometry and body centers
     */
    void computeJacobians(const Eigen::Vector3d& center_a,
                          const Eigen::Vector3d& center_b) {
        // Compute kinematic Jacobians
        Eigen::Matrix<double, 3, 6> J_kin_a = ContactJacobian::computeKinematic(
            contact.position, center_a);
        Eigen::Matrix<double, 3, 6> J_kin_b = ContactJacobian::computeKinematic(
            contact.position, center_b);

        // Normal Jacobian
        J_n.block<1, 6>(0, 0) = contact.normal.transpose() * J_kin_a;
        J_n.block<1, 6>(0, 6) = -contact.normal.transpose() * J_kin_b;

        // Friction cone Jacobian
        // Row 0: Normal
        J_f.block<1, 6>(0, 0) = J_n.block<1, 6>(0, 0);
        J_f.block<1, 6>(0, 6) = J_n.block<1, 6>(0, 6);

        // Rows 1-2: Tangential
        J_f.block<1, 6>(1, 0) = contact.tangent1.transpose() * J_kin_a;
        J_f.block<1, 6>(1, 6) = -contact.tangent1.transpose() * J_kin_b;
        J_f.block<1, 6>(2, 0) = contact.tangent2.transpose() * J_kin_a;
        J_f.block<1, 6>(2, 6) = -contact.tangent2.transpose() * J_kin_b;

        // Row 3: Torsion
        J_f.block<1, 3>(3, 3) = r_tau * contact.normal.transpose();
        J_f.block<1, 3>(3, 9) = -r_tau * contact.normal.transpose();
    }

    /**
     * @brief Compute relative normal velocity
     */
    double computeNormalVelocity(const SpatialVector& v_a,
                                  const SpatialVector& v_b) const {
        Eigen::Matrix<double, 12, 1> v;
        v.head<6>() = v_a;
        v.tail<6>() = v_b;
        return (J_n * v)(0);
    }

    /**
     * @brief Compute relative tangential velocity vector
     */
    Eigen::Vector2d computeTangentialVelocity(const SpatialVector& v_a,
                                               const SpatialVector& v_b) const {
        Eigen::Matrix<double, 12, 1> v;
        v.head<6>() = v_a;
        v.tail<6>() = v_b;
        return (J_f.block<2, 12>(1, 0) * v);
    }
};

}  // namespace vde
