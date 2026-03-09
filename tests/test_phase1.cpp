/**
 * @file test_phase1.cpp
 * @brief Phase 1 acceptance tests
 *
 * Verify Phase 1 three acceptance criteria:
 * 1. Geometric correctness: when sphere just touches plane, output g_eq must be 0
 * 2. Normal stability: sphere pressed into plane at any depth, output equivalent normal n_c must remain vertical upward, error < 10^-6
 * 3. Torsion radius validity: larger contact area means significantly larger r_tau
 */

#include <iostream>
#include <cmath>
#include "geometry/AnalyticalSDF.h"
#include "geometry/VolumetricIntegrator.h"

using namespace vde;

/**
 * @brief Acceptance criterion 1: Geometric correctness
 *
 * When sphere just touches plane, output g_eq must be 0
 */
bool testCriterion1_GeometricCorrectness() {
    std::cout << "========================================" << std::endl;
    std::cout << "Criterion 1: Geometric Correctness" << std::endl;
    std::cout << "  When sphere just touches plane, g_eq must be 0" << std::endl;
    std::cout << "========================================" << std::endl;

    // Sphere at y = 1, radius 1, just touches y = 0 plane
    Eigen::Vector3d sphere_center(0, 1, 0);
    double sphere_radius = 1.0;
    SphereSDF sphere(sphere_center, sphere_radius);

    // Plane at y = 0, normal upward
    Eigen::Vector3d plane_normal(0, 1, 0);
    Eigen::Vector3d plane_point(0, 0, 0);
    HalfSpaceSDF plane(plane_normal, plane_point);

    // Create bounding boxes
    AABB sphere_aabb = VolumetricIntegrator::sphereAABB(sphere, 0.5);
    AABB plane_aabb = VolumetricIntegrator::halfSpaceAABB(plane, Eigen::Vector3d(0, 0, 0), 3.0);

    // Use high resolution integrator
    VolumetricIntegrator integrator(64, 2.0);
    ContactGeometry contact = integrator.computeContactGeometry(sphere, plane, sphere_aabb, plane_aabb);

    std::cout << "  Sphere center: (0, 1, 0), radius: 1" << std::endl;
    std::cout << "  Plane: y = 0" << std::endl;
    std::cout << "  Result g_eq: " << contact.g_eq << std::endl;

    // Acceptance criterion: g_eq should be very close to 0 (within numerical error)
    bool passed = std::abs(contact.g_eq) < 0.1;  // Allow some grid discretization error

    std::cout << "  Expected: g_eq approx 0" << std::endl;
    std::cout << "  Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Acceptance criterion 2: Normal stability
 *
 * Sphere pressed into plane at any depth, output equivalent normal n_c must remain vertical upward, error < 10^-6
 */
bool testCriterion2_NormalStability() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Criterion 2: Normal Stability" << std::endl;
    std::cout << "  Normal must stay vertical (0,1,0) within 1e-6 error" << std::endl;
    std::cout << "========================================" << std::endl;

    Eigen::Vector3d plane_normal(0, 1, 0);
    Eigen::Vector3d plane_point(0, 0, 0);
    HalfSpaceSDF plane(plane_normal, plane_point);
    AABB plane_aabb = VolumetricIntegrator::halfSpaceAABB(plane, Eigen::Vector3d(0, 0, 0), 3.0);

    VolumetricIntegrator integrator(64, 2.0);

    // Test different penetration depths
    std::vector<double> depths = {0.1, 0.3, 0.5, 0.8};
    bool all_passed = true;
    double max_error = 0.0;
    double sphere_radius = 1.0;

    for (double depth : depths) {
        // Sphere pressed into plane by depth
        Eigen::Vector3d sphere_center(0, sphere_radius - depth, 0);
        SphereSDF sphere_local(sphere_center, sphere_radius);
        AABB sphere_aabb = VolumetricIntegrator::sphereAABB(sphere_local, 0.5);

        ContactGeometry contact = integrator.computeContactGeometry(sphere_local, plane, sphere_aabb, plane_aabb);

        // Compute normal error (deviation from vertical upward)
        Eigen::Vector3d expected_normal(0, 1, 0);
        double error = (contact.n_c - expected_normal).norm();

        if (error > max_error) {
            max_error = error;
        }

        std::cout << "  Depth: " << depth << ", n_c: (" << contact.n_c.x() << ", "
                  << contact.n_c.y() << ", " << contact.n_c.z() << "), error: " << error << std::endl;

        if (error >= 1e-6) {
            all_passed = false;
        }
    }

    std::cout << "  Max error: " << max_error << std::endl;
    std::cout << "  Threshold: 1e-6" << std::endl;
    std::cout << "  Result: " << (all_passed ? "PASSED" : "FAILED") << std::endl;

    return all_passed;
}

/**
 * @brief Acceptance criterion 3: Torsion radius validity
 *
 * Larger contact area (e.g., flat box vs tall box) means significantly larger r_tau
 */
bool testCriterion3_TorsionRadiusValidity() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Criterion 3: Torsion Radius Validity" << std::endl;
    std::cout << "  Larger contact area -> larger r_tau" << std::endl;
    std::cout << "========================================" << std::endl;

    Eigen::Vector3d plane_normal(0, 1, 0);
    Eigen::Vector3d plane_point(0, 0, 0);
    HalfSpaceSDF plane(plane_normal, plane_point);
    AABB plane_aabb = VolumetricIntegrator::halfSpaceAABB(plane, Eigen::Vector3d(0, 0, 0), 5.0);

    VolumetricIntegrator integrator(64, 2.0);

    // Test 1: Flat box (large contact area)
    // Size: 4 x 4 x 1, bottom at y = 0
    Eigen::Vector3d flat_min(-2, 0, -2);
    Eigen::Vector3d flat_max(2, 1, 2);
    BoxSDF box_flat(flat_min, flat_max);
    AABB flat_aabb = VolumetricIntegrator::boxAABB(box_flat, 0.5);
    ContactGeometry contact_flat = integrator.computeContactGeometry(box_flat, plane, flat_aabb, plane_aabb);

    // Test 2: Tall box (small contact area)
    // Size: 1 x 4 x 1, bottom at y = 0
    Eigen::Vector3d tall_min(-0.5, 0, -0.5);
    Eigen::Vector3d tall_max(0.5, 4, 0.5);
    BoxSDF box_tall(tall_min, tall_max);
    AABB tall_aabb = VolumetricIntegrator::boxAABB(box_tall, 0.5);
    ContactGeometry contact_tall = integrator.computeContactGeometry(box_tall, plane, tall_aabb, plane_aabb);

    std::cout << "  Flat box (4x4 base): r_tau = " << contact_flat.r_tau << std::endl;
    std::cout << "  Tall box (1x1 base): r_tau = " << contact_tall.r_tau << std::endl;

    // Acceptance criterion: flat box torsion radius should be significantly larger than tall box
    bool passed = contact_flat.r_tau > contact_tall.r_tau * 2.0;  // At least 2x larger

    std::cout << "  Expected: r_tau(flat) >> r_tau(tall)" << std::endl;
    std::cout << "  Ratio: " << (contact_flat.r_tau / contact_tall.r_tau) << std::endl;
    std::cout << "  Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Additional test: verify g_eq changes with different sphere penetration depths
 */
bool testSpherePenetrationDepth() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Additional Test: Sphere Penetration" << std::endl;
    std::cout << "========================================" << std::endl;

    Eigen::Vector3d plane_normal(0, 1, 0);
    Eigen::Vector3d plane_point(0, 0, 0);
    HalfSpaceSDF plane(plane_normal, plane_point);
    AABB plane_aabb = VolumetricIntegrator::halfSpaceAABB(plane, Eigen::Vector3d(0, 0, 0), 3.0);

    VolumetricIntegrator integrator(64, 2.0);

    double sphere_radius = 1.0;
    std::vector<double> depths = {0.0, 0.1, 0.2, 0.3, 0.5};

    std::cout << "  Sphere radius: " << sphere_radius << std::endl;
    std::cout << "  Penetration depths and g_eq:" << std::endl;

    for (double depth : depths) {
        Eigen::Vector3d sphere_center(0, sphere_radius - depth, 0);
        SphereSDF sphere(sphere_center, sphere_radius);
        AABB sphere_aabb = VolumetricIntegrator::sphereAABB(sphere, 0.5);

        ContactGeometry contact = integrator.computeContactGeometry(sphere, plane, sphere_aabb, plane_aabb);

        std::cout << "    depth = " << depth << " -> g_eq = " << contact.g_eq
                  << ", r_tau = " << contact.r_tau << std::endl;
    }

    return true;
}

int main() {
    std::cout << "****************************************" << std::endl;
    std::cout << "Phase 1 Acceptance Tests" << std::endl;
    std::cout << "****************************************" << std::endl;

    bool criterion1 = testCriterion1_GeometricCorrectness();
    bool criterion2 = testCriterion2_NormalStability();
    bool criterion3 = testCriterion3_TorsionRadiusValidity();

    testSpherePenetrationDepth();

    std::cout << "\n****************************************" << std::endl;
    std::cout << "Phase 1 Acceptance Summary" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "Criterion 1 (Geometric Correctness): " << (criterion1 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Criterion 2 (Normal Stability): " << (criterion2 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Criterion 3 (Torsion Radius Validity): " << (criterion3 ? "PASSED" : "FAILED") << std::endl;

    bool all_passed = criterion1 && criterion2 && criterion3;

    std::cout << "\nOverall: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "****************************************" << std::endl;

    return all_passed ? 0 : 1;
}
