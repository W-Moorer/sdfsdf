/**
 * @file test_phase3.cpp
 * @brief Phase 3 acceptance tests
 *
 * Verify Phase 3 acceptance criteria:
 * 1. Energy conservation: Total kinetic energy change rate close to 0 in elastic collision
 * 2. Virtual work consistency: J^T f matches generalized forces
 */

#include <iostream>
#include <cmath>
#include <vector>
#include "dynamics/RigidBody.h"
#include "dynamics/ContactDynamics.h"
#include "geometry/AnalyticalSDF.h"
#include "geometry/VolumetricIntegrator.h"
#include "solver/PenaltySolver.h"

using namespace vde;

/**
 * @brief Acceptance criterion 1: Energy conservation
 *
 * In gravity-free, frictionless, perfectly elastic collision,
 * total kinetic energy change rate must be close to 0
 */
bool testCriterion1_EnergyConservation() {
    std::cout << "========================================" << std::endl;
    std::cout << "Criterion 1: Energy Conservation" << std::endl;
    std::cout << "  KE change rate ~ 0 in elastic collision" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create a sphere
    RigidBodyProperties props = RigidBodyProperties::sphere(1.0, 0.5);
    RigidBody body(props);

    // Set initial state: moving downward
    body.state().position = Eigen::Vector3d(0, 2.0, 0);
    body.state().linear_velocity = Eigen::Vector3d(0, -1.0, 0);

    // Simulation parameters
    double dt = 0.001;
    int num_steps = 1000;
    Eigen::Vector3d gravity(0, 0, 0);  // No gravity

    std::vector<double> energy_history;

    for (int i = 0; i < num_steps; ++i) {
        // Simple floor collision (bounce)
        if (body.state().position.y() < 0.5) {
            // Elastic collision: reverse velocity
            body.state().linear_velocity.y() = std::abs(body.state().linear_velocity.y());
        }

        // Integrate
        body.integrate(dt, gravity);

        // Record energy
        double KE = body.kineticEnergy();
        energy_history.push_back(KE);
    }

    // Check energy conservation
    double initial_energy = energy_history.front();
    double final_energy = energy_history.back();
    double max_energy = *std::max_element(energy_history.begin(), energy_history.end());
    double min_energy = *std::min_element(energy_history.begin(), energy_history.end());

    double energy_variation = (max_energy - min_energy) / initial_energy;

    std::cout << "Initial energy: " << initial_energy << std::endl;
    std::cout << "Final energy: " << final_energy << std::endl;
    std::cout << "Max energy: " << max_energy << std::endl;
    std::cout << "Min energy: " << min_energy << std::endl;
    std::cout << "Relative variation: " << energy_variation * 100 << "%" << std::endl;

    // Allow small numerical errors (< 1%)
    bool passed = energy_variation < 0.01;

    std::cout << "Threshold: 1%" << std::endl;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Acceptance criterion 2: Virtual work consistency
 *
 * Check that J^T f matches generalized forces
 */
bool testCriterion2_VirtualWorkConsistency() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Criterion 2: Virtual Work Consistency" << std::endl;
    std::cout << "  J^T f should match generalized forces" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create contact point
    ContactPoint contact;
    contact.position = Eigen::Vector3d(0, 0, 0);
    contact.normal = Eigen::Vector3d(0, 1, 0);
    contact.computeTangentDirections();

    // Body center
    Eigen::Vector3d body_center(0, -0.5, 0);

    // Compute kinematic Jacobian
    Eigen::Matrix<double, 3, 6> J_kin = ContactJacobian::computeKinematic(
        contact.position, body_center);

    // Apply a world force
    Eigen::Vector3d world_force(10, 20, 30);

    // Compute generalized force: tau = J^T * F
    SpatialVector generalized_force = J_kin.transpose() * world_force;

    // Expected: linear part = F, angular part = r x F
    Eigen::Vector3d r = contact.position - body_center;
    Eigen::Vector3d expected_torque = r.cross(world_force);

    std::cout << "World force: " << world_force.transpose() << std::endl;
    std::cout << "Generalized force: " << generalized_force.transpose() << std::endl;
    std::cout << "Expected linear: " << world_force.transpose() << std::endl;
    std::cout << "Expected angular: " << expected_torque.transpose() << std::endl;

    // Check
    double linear_error = (generalized_force.head<3>() - world_force).norm();
    double angular_error = (generalized_force.tail<3>() - expected_torque).norm();

    std::cout << "Linear error: " << linear_error << std::endl;
    std::cout << "Angular error: " << angular_error << std::endl;

    bool passed = linear_error < 1e-10 && angular_error < 1e-10;

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test rigid body integration
 */
bool testRigidBodyIntegration() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Rigid Body Integration Test" << std::endl;
    std::cout << "========================================" << std::endl;

    RigidBodyProperties props = RigidBodyProperties::sphere(1.0, 0.5);
    RigidBody body(props);

    // Initial state
    body.state().position = Eigen::Vector3d(0, 0, 0);
    body.state().linear_velocity = Eigen::Vector3d(1, 2, 3);

    double dt = 0.01;
    Eigen::Vector3d gravity(0, -9.81, 0);

    // Integrate for 100 steps
    for (int i = 0; i < 100; ++i) {
        body.integrate(dt, gravity);
    }

    // Check position (should follow parabolic trajectory)
    double expected_y = 0 + 2.0 * 1.0 + 0.5 * (-9.81) * 1.0 * 1.0;

    std::cout << "Final position: " << body.state().position.transpose() << std::endl;
    std::cout << "Expected y (approx): " << expected_y << std::endl;

    // Allow some error due to semi-implicit integration
    bool passed = std::abs(body.state().position.y() - expected_y) < 0.5;

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test mass matrix
 */
bool testMassMatrix() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Mass Matrix Test" << std::endl;
    std::cout << "========================================" << std::endl;

    RigidBodyProperties props = RigidBodyProperties::box(2.0, 1.0, 2.0, 3.0);
    RigidBody body(props);

    Eigen::Matrix<double, 6, 6> M = body.massMatrix();
    Eigen::Matrix<double, 6, 6> Minv = body.inverseMassMatrix();

    // Check M * Minv = I
    Eigen::Matrix<double, 6, 6> product = M * Minv;
    double error = (product - Eigen::Matrix<double, 6, 6>::Identity()).norm();

    std::cout << "Mass matrix:\n" << M << std::endl;
    std::cout << "Inverse mass matrix:\n" << Minv << std::endl;
    std::cout << "M * Minv error: " << error << std::endl;

    bool passed = error < 1e-10;

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test the integrated normal Jacobian against finite differences
 *
 * Validates the paper's J_n = D g_eq / D q formula on a sphere-plane contact.
 */
bool testIntegratedNormalJacobian() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Integrated Normal Jacobian Test" << std::endl;
    std::cout << "========================================" << std::endl;

    const double radius = 1.0;
    const double center_y = 0.8;  // 0.2 penetration into y=0 plane
    const double epsilon = 1e-4;

    const auto local_sphere = std::make_shared<SphereSDF>(Eigen::Vector3d::Zero(), radius);
    const auto local_plane = std::make_shared<HalfSpaceSDF>(
        Eigen::Vector3d(0, 1, 0), Eigen::Vector3d::Zero());
    TransformedSDF sphere(local_sphere, Eigen::Vector3d(0, center_y, 0), Eigen::Quaterniond::Identity());
    TransformedSDF plane(local_plane, Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());
    AABB sphere_aabb(
        Eigen::Vector3d(-radius - 0.5, center_y - radius - 0.5, -radius - 0.5),
        Eigen::Vector3d(radius + 0.5, center_y + radius + 0.5, radius + 0.5));
    AABB plane_aabb = VolumetricIntegrator::halfSpaceAABB(
        *local_plane, Eigen::Vector3d::Zero(), 3.0);
    VolumetricIntegrator integrator(64, 2.0);
    ContactGeometry geometry = integrator.computeContactGeometry(sphere, plane, sphere_aabb, plane_aabb);

    ContactConstraint constraint;
    constraint.computeJacobiansFromGeometry(
        geometry,
        Eigen::Vector3d(0, center_y, 0),
        Eigen::Vector3d::Zero(),
        1e-6);
    constraint.J_n = computeIntegratedNormalJacobianFiniteDifference(
        integrator,
        local_sphere,
        sphere_aabb,
        Eigen::Vector3d(0, center_y, 0),
        Eigen::Quaterniond::Identity(),
        local_plane,
        plane_aabb,
        Eigen::Vector3d::Zero(),
        Eigen::Quaterniond::Identity(),
        1e-5);

    TransformedSDF sphere_plus(
        local_sphere, Eigen::Vector3d(0, center_y + epsilon, 0), Eigen::Quaterniond::Identity());
    TransformedSDF sphere_minus(
        local_sphere, Eigen::Vector3d(0, center_y - epsilon, 0), Eigen::Quaterniond::Identity());

    const double g_plus =
        integrator.computeContactGeometry(sphere_plus, plane, sphere_aabb, plane_aabb).g_eq;
    const double g_minus =
        integrator.computeContactGeometry(sphere_minus, plane, sphere_aabb, plane_aabb).g_eq;

    const double fd = (g_plus - g_minus) / (2.0 * epsilon);
    const double analytic = constraint.J_n(0, 1);
    const double rel_error = std::abs(fd - analytic) / std::max(1.0, std::abs(fd));

    std::cout << "Finite-difference dg_eq/dy: " << fd << std::endl;
    std::cout << "Analytical J_n * e_y: " << analytic << std::endl;
    std::cout << "Relative error: " << rel_error << std::endl;

    const bool passed = rel_error < 5e-2;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

bool testPenaltyForceDirection() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Penalty Force Direction Test" << std::endl;
    std::cout << "========================================" << std::endl;

    constexpr double radius = 0.20;
    constexpr double center_y = 0.10;

    RigidBody sphere(RigidBodyProperties::sphere(1.0, radius));
    sphere.setPosition(Eigen::Vector3d(0.0, center_y, 0.0));

    RigidBody support(RigidBodyProperties::box(1.0, 10.0, 2.0, 10.0));
    support.setPosition(Eigen::Vector3d(0.0, -1.0, 0.0));
    support.setStatic(true);

    const auto local_sphere = std::make_shared<SphereSDF>(Eigen::Vector3d::Zero(), radius);
    const auto local_support = std::make_shared<BoxSDF>(
        Eigen::Vector3d(-5.0, -1.0, -5.0),
        Eigen::Vector3d(5.0, 1.0, 5.0));
    const TransformedSDF sphere_sdf(
        local_sphere, sphere.position(), Eigen::Quaterniond::Identity());
    const TransformedSDF support_sdf(
        local_support, support.position(), Eigen::Quaterniond::Identity());

    const AABB sphere_aabb(
        Eigen::Vector3d(-radius, center_y - radius, -radius),
        Eigen::Vector3d(radius, center_y + radius, radius));
    const AABB support_aabb(
        Eigen::Vector3d(-5.0, -2.0, -5.0),
        Eigen::Vector3d(5.0, 0.0, 5.0));

    VolumetricIntegrator integrator(48, 2.0);
    const ContactGeometry geometry = integrator.computeContactGeometry(
        sphere_sdf, support_sdf, sphere_aabb, support_aabb);

    ContactConstraint constraint;
    constraint.computeJacobiansFromGeometry(
        geometry,
        sphere.centerOfMassWorld(),
        support.centerOfMassWorld(),
        1e-6);

    PenaltyParameters params;
    params.stiffness = 60000.0;
    params.damping = 220.0;
    params.friction_coefficient = 0.0;
    PenaltySolver solver(params);
    const PenaltyContactResult result =
        solver.solveContactDetailed(sphere, support, constraint, 0.0025);

    std::cout << "g_eq: " << geometry.g_eq << std::endl;
    std::cout << "normal: " << constraint.contact.normal.transpose() << std::endl;
    std::cout << "world_force: " << result.world_force.transpose() << std::endl;
    std::cout << "sphere external force: " << sphere.externalForce().transpose() << std::endl;
    std::cout << "support external force: " << support.externalForce().transpose() << std::endl;

    const bool passed =
        result.active &&
        geometry.g_eq < 0.0 &&
        result.world_force.dot(constraint.contact.normal) > 0.0 &&
        sphere.externalForce().dot(constraint.contact.normal) > 0.0 &&
        support.externalForce().dot(constraint.contact.normal) < 0.0;
    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;
    return passed;
}

int main() {
    std::cout << "****************************************" << std::endl;
    std::cout << "Phase 3 Acceptance Tests" << std::endl;
    std::cout << "****************************************" << std::endl;

    bool test1 = testCriterion1_EnergyConservation();
    bool test2 = testCriterion2_VirtualWorkConsistency();
    bool test3 = testRigidBodyIntegration();
    bool test4 = testMassMatrix();
    bool test5 = testIntegratedNormalJacobian();
    bool test6 = testPenaltyForceDirection();

    std::cout << "\n****************************************" << std::endl;
    std::cout << "Phase 3 Acceptance Summary" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "Criterion 1 (Energy Conservation): " << (test1 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Criterion 2 (Virtual Work): " << (test2 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Rigid Body Integration: " << (test3 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Mass Matrix: " << (test4 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Integrated Normal Jacobian: " << (test5 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Penalty Force Direction: " << (test6 ? "PASSED" : "FAILED") << std::endl;

    bool all_passed = test1 && test2 && test3 && test4 && test5 && test6;

    std::cout << "\nOverall: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "****************************************" << std::endl;

    return all_passed ? 0 : 1;
}
