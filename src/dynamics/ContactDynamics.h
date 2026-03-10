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

inline ContactPoint makeContactPoint(const ContactGeometry& geometry) {
    ContactPoint point;
    point.position = geometry.p_c;
    point.normal = geometry.n_c;
    point.penetration_depth = -geometry.g_eq;
    point.computeTangentDirections();
    return point;
}

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
    double r_tau_hat;                   // Regularized torsion radius used by the SOCCP scaling

    ContactConstraint() : g_eq(0.0), r_tau(0.0), r_tau_hat(0.0) {
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

        // Row 3: Scaled torsion velocity used in the 4D SOC formulation.
        const double torsion_scale = effectiveTorsionRadius();
        J_f.block<1, 3>(3, 3) = torsion_scale * contact.normal.transpose();
        J_f.block<1, 3>(3, 9) = -torsion_scale * contact.normal.transpose();
    }

    /**
     * @brief Compute Jacobians from the volumetric contact geometry samples
     *
     * This implements the paper's integrated Jacobian definitions:
     * - J_n from the directional derivative of the matrix-ratio gap
     * - \tilde{J}_t from the unified weight measure w_p
     */
    void computeJacobiansFromGeometry(const ContactGeometry& geometry,
                                      const Eigen::Vector3d& center_a,
                                      const Eigen::Vector3d& center_b,
                                      double torsion_regularization = 0.0) {
        contact = makeContactPoint(geometry);
        g_eq = geometry.g_eq;
        r_tau = geometry.r_tau;
        r_tau_hat = std::max(r_tau, torsion_regularization);

        if (geometry.sample_positions.empty() || geometry.moment_pm1 <= 1e-15) {
            computeJacobians(center_a, center_b);
            return;
        }

        const double p = geometry.p_norm;
        const double I_p = geometry.moment_p;
        const double I_pm1 = geometry.moment_pm1;

        Eigen::Matrix<double, 1, 12> integral_p_minus_1 = Eigen::Matrix<double, 1, 12>::Zero();
        Eigen::Matrix<double, 1, 12> integral_p_minus_2 = Eigen::Matrix<double, 1, 12>::Zero();
        Eigen::Matrix<double, 3, 12> integrated_tangent = Eigen::Matrix<double, 3, 12>::Zero();

        for (size_t i = 0; i < geometry.sample_positions.size(); ++i) {
            const Eigen::Vector3d& x = geometry.sample_positions[i];
            const Eigen::Vector3d& grad_delta = geometry.sample_gradients[i];
            const double delta = geometry.sample_depths[i];
            const double weight = geometry.sample_weights[i];
            const int active_body = (i < geometry.sample_active_body.size())
                ? geometry.sample_active_body[i]
                : 0;

            const Eigen::Matrix<double, 3, 6> J_a =
                ContactJacobian::computeKinematic(x, center_a);
            const Eigen::Matrix<double, 3, 6> J_b =
                ContactJacobian::computeKinematic(x, center_b);

            Eigen::Matrix<double, 3, 12> J_rel;
            J_rel.block<3, 6>(0, 0) = J_a;
            J_rel.block<3, 6>(0, 6) = -J_b;

            Eigen::Matrix<double, 1, 12> gap_row = Eigen::Matrix<double, 1, 12>::Zero();
            if (active_body == 0) {
                gap_row.block<1, 6>(0, 0) = -grad_delta.transpose() * J_a;
            } else {
                gap_row.block<1, 6>(0, 6) = -grad_delta.transpose() * J_b;
            }

            integral_p_minus_1 += weight * gap_row;
            integral_p_minus_2 += (weight / std::max(delta, 1e-12)) * gap_row;

            const double normalized_weight = weight / I_pm1;
            integrated_tangent.row(0) +=
                normalized_weight * contact.tangent1.transpose() * J_rel;
            integrated_tangent.row(1) +=
                normalized_weight * contact.tangent2.transpose() * J_rel;

            const Eigen::Vector3d torsion_arm = (x - geometry.p_c).cross(contact.normal);
            integrated_tangent.row(2) += normalized_weight * torsion_arm.transpose() * J_rel;
        }

        J_n =
            -(p * I_pm1 * integral_p_minus_1 - (p - 1.0) * I_p * integral_p_minus_2) /
            (I_pm1 * I_pm1);

        J_f.setZero();
        J_f.row(0) = J_n;
        J_f.row(1) = integrated_tangent.row(0);
        J_f.row(2) = integrated_tangent.row(1);
        J_f.row(3) = effectiveTorsionRadius() * integrated_tangent.row(2);
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

    /**
     * @brief Compute relative tangential-torsional velocity vector
     */
    Eigen::Vector3d computeTangentialTorsionalVelocity(const SpatialVector& v_a,
                                                        const SpatialVector& v_b) const {
        Eigen::Matrix<double, 12, 1> v;
        v.head<6>() = v_a;
        v.tail<6>() = v_b;
        return (J_f.block<3, 12>(1, 0) * v);
    }

    /**
     * @brief Return the 3x12 tangential-torsional Jacobian block
     */
    Eigen::Matrix<double, 3, 12> tangentialJacobian() const {
        return J_f.block<3, 12>(1, 0);
    }

private:
    double effectiveTorsionRadius() const {
        return (r_tau_hat > 0.0) ? r_tau_hat : r_tau;
    }
};

inline ContactConstraint makeContactConstraint(const ContactGeometry& geometry,
                                               const Eigen::Vector3d& center_a,
                                               const Eigen::Vector3d& center_b) {
    ContactConstraint constraint;
    constraint.computeJacobiansFromGeometry(geometry, center_a, center_b);
    return constraint;
}

inline Eigen::Quaterniond applyRotationIncrement(const Eigen::Quaterniond& orientation,
                                                 const Eigen::Vector3d& delta_rotation) {
    const double angle = delta_rotation.norm();
    if (angle < 1e-14) {
        return orientation;
    }

    const Eigen::Vector3d axis = delta_rotation / angle;
    const Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, axis));
    return (dq * orientation).normalized();
}

inline double evaluateEquivalentGap(const VolumetricIntegrator& integrator,
                                    const std::shared_ptr<const SDF>& local_sdf_a,
                                    const AABB& fixed_world_aabb_a,
                                    const Eigen::Vector3d& position_a,
                                    const Eigen::Quaterniond& orientation_a,
                                    const std::shared_ptr<const SDF>& local_sdf_b,
                                    const AABB& fixed_world_aabb_b,
                                    const Eigen::Vector3d& position_b,
                                    const Eigen::Quaterniond& orientation_b) {
    TransformedSDF sdf_a(local_sdf_a, position_a, orientation_a);
    TransformedSDF sdf_b(local_sdf_b, position_b, orientation_b);
    return integrator.computeContactGeometry(
        sdf_a, sdf_b, fixed_world_aabb_a, fixed_world_aabb_b).g_eq;
}

inline Eigen::Matrix<double, 1, 12> computeIntegratedNormalJacobianFiniteDifference(
    const VolumetricIntegrator& integrator,
    const std::shared_ptr<const SDF>& local_sdf_a,
    const AABB& fixed_world_aabb_a,
    const Eigen::Vector3d& position_a,
    const Eigen::Quaterniond& orientation_a,
    const std::shared_ptr<const SDF>& local_sdf_b,
    const AABB& fixed_world_aabb_b,
    const Eigen::Vector3d& position_b,
    const Eigen::Quaterniond& orientation_b,
    double epsilon = 1e-5) {

    Eigen::Matrix<double, 1, 12> J_numerical = Eigen::Matrix<double, 1, 12>::Zero();

    for (int dof = 0; dof < 12; ++dof) {
        Eigen::Vector3d pos_a_plus = position_a;
        Eigen::Vector3d pos_a_minus = position_a;
        Eigen::Quaterniond ori_a_plus = orientation_a;
        Eigen::Quaterniond ori_a_minus = orientation_a;
        Eigen::Vector3d pos_b_plus = position_b;
        Eigen::Vector3d pos_b_minus = position_b;
        Eigen::Quaterniond ori_b_plus = orientation_b;
        Eigen::Quaterniond ori_b_minus = orientation_b;

        if (dof < 3) {
            pos_a_plus(dof) += epsilon;
            pos_a_minus(dof) -= epsilon;
        } else if (dof < 6) {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(dof - 3) = epsilon;
            ori_a_plus = applyRotationIncrement(orientation_a, delta);
            ori_a_minus = applyRotationIncrement(orientation_a, -delta);
        } else if (dof < 9) {
            pos_b_plus(dof - 6) += epsilon;
            pos_b_minus(dof - 6) -= epsilon;
        } else {
            Eigen::Vector3d delta = Eigen::Vector3d::Zero();
            delta(dof - 9) = epsilon;
            ori_b_plus = applyRotationIncrement(orientation_b, delta);
            ori_b_minus = applyRotationIncrement(orientation_b, -delta);
        }

        const double g_plus = evaluateEquivalentGap(
            integrator,
            local_sdf_a,
            fixed_world_aabb_a,
            pos_a_plus,
            ori_a_plus,
            local_sdf_b,
            fixed_world_aabb_b,
            pos_b_plus,
            ori_b_plus);
        const double g_minus = evaluateEquivalentGap(
            integrator,
            local_sdf_a,
            fixed_world_aabb_a,
            pos_a_minus,
            ori_a_minus,
            local_sdf_b,
            fixed_world_aabb_b,
            pos_b_minus,
            ori_b_minus);

        J_numerical(dof) = (g_plus - g_minus) / (2.0 * epsilon);
    }

    return J_numerical;
}

}  // namespace vde
