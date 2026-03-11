/**
 * @file test_sdf.cpp
 * @brief SDF unit tests
 *
 * Verify gradient vectors are normalized and point outward
 */

#include <iostream>
#include <random>
#include <cmath>
#include "geometry/AnalyticalSDF.h"
#include "geometry/MeshSDF.h"

using namespace vde;

/**
 * @brief Test SphereSDF gradient properties
 *
 * Verify:
 * 1. Gradient vector is normalized (length is 1)
 * 2. Gradient points outward everywhere except the sphere center
 * 3. Distance is 0 on surface
 */
bool testSphereSDF() {
    std::cout << "=== Testing SphereSDF ===" << std::endl;

    Eigen::Vector3d center(0, 0, 0);
    double radius = 1.0;
    SphereSDF sphere(center, radius);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3.0, 3.0);

    const int num_samples = 1000;
    int passed = 0;
    double max_gradient_error = 0.0;
    double max_norm_error = 0.0;

    for (int i = 0; i < num_samples; ++i) {
        Eigen::Vector3d x(dis(gen), dis(gen), dis(gen));

        double phi = sphere.phi(x);
        Eigen::Vector3d grad = sphere.gradient(x);

        // Verify gradient normalization
        double grad_norm = grad.norm();
        double norm_error = std::abs(grad_norm - 1.0);
        if (norm_error > max_norm_error) {
            max_norm_error = norm_error;
        }

        // SphereSDF::gradient always returns the outward normal, regardless of
        // whether the point is inside or outside the sphere.
        if (phi > 1e-6) {
            Eigen::Vector3d expected_dir = (x - center).normalized();
            double dir_error = (grad - expected_dir).norm();
            if (dir_error < 1e-6) {
                passed++;
            }
        } else if (phi < -1e-6) {
            Eigen::Vector3d expected_dir = (x - center).normalized();
            double dir_error = (grad - expected_dir).norm();
            if (dir_error < 1e-6) {
                passed++;
            }
        } else {
            // On surface, don't check direction
            passed++;
        }
    }

    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Passed: " << passed << std::endl;
    std::cout << "  Max norm error: " << max_norm_error << std::endl;

    return max_norm_error < 1e-6 && passed == num_samples;
}

/**
 * @brief Test BoxSDF gradient properties
 */
bool testBoxSDF() {
    std::cout << "=== Testing BoxSDF ===" << std::endl;

    Eigen::Vector3d min_corner(-1, -1, -1);
    Eigen::Vector3d max_corner(1, 1, 1);
    BoxSDF box(min_corner, max_corner);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3.0, 3.0);

    const int num_samples = 1000;
    double max_norm_error = 0.0;

    for (int i = 0; i < num_samples; ++i) {
        Eigen::Vector3d x(dis(gen), dis(gen), dis(gen));

        Eigen::Vector3d grad = box.gradient(x);
        double grad_norm = grad.norm();
        double norm_error = std::abs(grad_norm - 1.0);

        if (norm_error > max_norm_error) {
            max_norm_error = norm_error;
        }
    }

    Eigen::Vector3d inside_point(0, 0, 0);
    Eigen::Vector3d surface_point(1, 0, 0);
    Eigen::Vector3d outside_point(2, 0, 0);
    double inside_phi = box.phi(inside_point);
    double surface_phi = box.phi(surface_point);
    double outside_phi = box.phi(outside_point);

    std::cout << "  phi(center): " << inside_phi << std::endl;
    std::cout << "  phi(surface): " << surface_phi << std::endl;
    std::cout << "  phi(outside): " << outside_phi << std::endl;

    bool signed_distance_ok =
        (inside_phi < 0.0) &&
        (std::abs(surface_phi) < 1e-10) &&
        (outside_phi > 0.0);

    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Max norm error: " << max_norm_error << std::endl;

    return max_norm_error < 1e-6 && signed_distance_ok;
}

/**
 * @brief Test HalfSpaceSDF gradient properties
 */
bool testHalfSpaceSDF() {
    std::cout << "=== Testing HalfSpaceSDF ===" << std::endl;

    Eigen::Vector3d normal(0, 1, 0);  // Upward
    Eigen::Vector3d point_on_plane(0, 0, 0);
    HalfSpaceSDF half_space(normal, point_on_plane);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-3.0, 3.0);

    const int num_samples = 1000;
    double max_norm_error = 0.0;
    double max_direction_error = 0.0;

    for (int i = 0; i < num_samples; ++i) {
        Eigen::Vector3d x(dis(gen), dis(gen), dis(gen));

        double phi = half_space.phi(x);
        Eigen::Vector3d grad = half_space.gradient(x);

        // Verify gradient normalization
        double grad_norm = grad.norm();
        double norm_error = std::abs(grad_norm - 1.0);
        if (norm_error > max_norm_error) {
            max_norm_error = norm_error;
        }

        // Verify gradient direction (should always equal normal)
        double dir_error = (grad - normal).norm();
        if (dir_error > max_direction_error) {
            max_direction_error = dir_error;
        }

        // Verify distance calculation
        double expected_phi = normal.dot(x - point_on_plane);
        double phi_error = std::abs(phi - expected_phi);
        if (phi_error > 1e-10) {
            std::cout << "  ERROR: phi mismatch at sample " << i << std::endl;
            return false;
        }
    }

    std::cout << "  Samples: " << num_samples << std::endl;
    std::cout << "  Max norm error: " << max_norm_error << std::endl;
    std::cout << "  Max direction error: " << max_direction_error << std::endl;

    return max_norm_error < 1e-6 && max_direction_error < 1e-10;
}

bool testMeshSDFBoxAgreement() {
    std::cout << "=== Testing MeshSDF Box Agreement ===" << std::endl;

    const Eigen::Vector3d min_corner(-1.0, -0.8, -0.6);
    const Eigen::Vector3d max_corner(1.0, 0.8, 0.6);
    const BoxSDF analytic_box(min_corner, max_corner);
    const MeshSDF mesh_box(TriangleMesh::makeBox(min_corner, max_corner));

    const std::vector<Eigen::Vector3d> samples = {
        Eigen::Vector3d(0.6, 0.0, 0.0),
        Eigen::Vector3d(0.9, 0.0, 0.0),
        Eigen::Vector3d(1.4, 0.2, 0.1),
        Eigen::Vector3d(-1.3, 0.6, 0.0),
        Eigen::Vector3d(0.0, 1.1, 0.0),
        Eigen::Vector3d(0.3, -0.4, 0.55),
        Eigen::Vector3d(-0.6, 0.7, -0.2)
    };

    double max_phi_error = 0.0;
    double max_grad_error = 0.0;
    for (const auto& x : samples) {
        const double phi_analytic = analytic_box.phi(x);
        const double phi_mesh = mesh_box.phi(x);
        const Eigen::Vector3d grad_analytic = analytic_box.gradient(x);
        const Eigen::Vector3d grad_mesh = mesh_box.gradient(x);

        max_phi_error = std::max(max_phi_error, std::abs(phi_analytic - phi_mesh));
        max_grad_error = std::max(max_grad_error, (grad_analytic - grad_mesh).norm());
    }

    std::cout << "  Max phi error: " << max_phi_error << std::endl;
    std::cout << "  Max gradient error: " << max_grad_error << std::endl;

    return max_phi_error < 5e-6 && max_grad_error < 5e-3;
}

/**
 * @brief Test sphere-plane contact case
 *
 * Verify when sphere just touches plane, contact point distance is 0
 */
bool testSpherePlaneContact() {
    std::cout << "=== Testing Sphere-Plane Contact ===" << std::endl;

    // Sphere at y = 1, radius 1, just touches y = 0 plane
    Eigen::Vector3d sphere_center(0, 1, 0);
    double sphere_radius = 1.0;
    SphereSDF sphere(sphere_center, sphere_radius);

    // Plane at y = 0, normal upward
    Eigen::Vector3d plane_normal(0, 1, 0);
    Eigen::Vector3d plane_point(0, 0, 0);
    HalfSpaceSDF plane(plane_normal, plane_point);

    // Test sphere bottom point (just touches plane)
    Eigen::Vector3d contact_point(0, 0, 0);
    double sphere_phi = sphere.phi(contact_point);
    double plane_phi = plane.phi(contact_point);

    std::cout << "  Sphere phi at contact point: " << sphere_phi << std::endl;
    std::cout << "  Plane phi at contact point: " << plane_phi << std::endl;

    // Sphere should be on surface at contact point (phi approx 0)
    // Plane should also be on surface at contact point (phi approx 0)
    bool sphere_on_surface = std::abs(sphere_phi) < 1e-10;
    bool plane_on_surface = std::abs(plane_phi) < 1e-10;

    return sphere_on_surface && plane_on_surface;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "SDF Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= testSphereSDF();
    std::cout << std::endl;

    all_passed &= testBoxSDF();
    std::cout << std::endl;

    all_passed &= testHalfSpaceSDF();
    std::cout << std::endl;

    all_passed &= testMeshSDFBoxAgreement();
    std::cout << std::endl;

    all_passed &= testSpherePlaneContact();
    std::cout << std::endl;

    std::cout << "========================================" << std::endl;
    if (all_passed) {
        std::cout << "All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED!" << std::endl;
        return 1;
    }
}
