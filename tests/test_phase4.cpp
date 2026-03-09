/**
 * @file test_phase4.cpp
 * @brief Phase 4 acceptance tests
 *
 * Verify Phase 4 acceptance criteria:
 * 1. Hardening curve: F ~ delta^n (n ~ 1.5-2.0)
 * 2. Energy dissipation: energy loss < 5% of theoretical prediction
 */

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <iostream>
#include <cmath>
#include <vector>
#include "geometry/NayakSurface.h"
#include "dynamics/ViscoelasticContact.h"

using namespace vde;

/**
 * @brief Acceptance criterion 1: Hardening curve
 *
 * Contact force-displacement curve should follow power law F ~ delta^n
 * where n is approximately 1.5-2.0
 */
bool testCriterion1_HardeningCurve() {
    std::cout << "========================================" << std::endl;
    std::cout << "Criterion 1: Hardening Curve" << std::endl;
    std::cout << "  F ~ delta^n, n ~ 1.5-2.0" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create rough surface
    NayakSurface surface = NayakSurface::rough();

    // Test different penetrations
    std::vector<double> penetrations = {1e-6, 5e-6, 1e-5, 5e-5, 1e-4};
    std::vector<double> forces;

    std::cout << "Penetration (m)    Force (N)" << std::endl;
    std::cout << "---------------------------" << std::endl;

    for (double delta : penetrations) {
        double force = BushGibsonThomasModel::computeContactForce(surface, delta);
        forces.push_back(force);
        std::cout << std::scientific << delta << "    " << force << std::endl;
    }

    // Fit power law: F = C * delta^n
    // log(F) = log(C) + n * log(delta)
    // Linear regression on log-log data

    double sum_log_delta = 0.0, sum_log_force = 0.0;
    double sum_log_delta_sq = 0.0, sum_log_delta_log_force = 0.0;
    int n_points = 0;

    for (size_t i = 0; i < penetrations.size(); ++i) {
        if (forces[i] > 0 && penetrations[i] > 0) {
            double log_delta = std::log(penetrations[i]);
            double log_force = std::log(forces[i]);

            sum_log_delta += log_delta;
            sum_log_force += log_force;
            sum_log_delta_sq += log_delta * log_delta;
            sum_log_delta_log_force += log_delta * log_force;
            n_points++;
        }
    }

    // Linear regression
    double n = (n_points * sum_log_delta_log_force - sum_log_delta * sum_log_force) /
               (n_points * sum_log_delta_sq - sum_log_delta * sum_log_delta);

    std::cout << "\nFitted power law exponent n = " << n << std::endl;
    std::cout << "Expected range: 1.5 - 2.0" << std::endl;

    bool passed = (n >= 1.5 && n <= 2.0);

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Acceptance criterion 2: Energy dissipation
 *
 * Energy loss in single bounce should be within 5% of theoretical prediction
 */
bool testCriterion2_EnergyDissipation() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Criterion 2: Energy Dissipation" << std::endl;
    std::cout << "  Energy loss < 5% of theoretical" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create viscoelastic contact model
    ViscoelasticParams params;
    params.E = 1e6;           // 1 MPa (rubber)
    params.alpha = 0.3;       // 30% hysteresis
    ViscoelasticContactModel contact_model(params);

    // Simulate bouncing ball
    double mass = 0.1;        // 100g ball
    double radius = 0.02;     // 2cm radius
    double dt = 0.0001;       // Small time step

    double position = 0.5;    // Initial height 0.5m
    double velocity = 0.0;    // Start from rest
    double gravity = -9.81;

    BouncingBallTracker tracker;
    double max_height = position;
    bool was_falling = true;

    // Simulate for 2 seconds
    int num_steps = static_cast<int>(2.0 / dt);

    for (int i = 0; i < num_steps; ++i) {
        // Check for bounce (contact with ground)
        double penetration = radius - position;  // Positive when penetrating

        if (penetration > 0) {
            // Contact force
            double contact_area = M_PI * radius * radius;  // Simplified
            double F_contact = contact_model.computeNormalForce(penetration, contact_area, dt);

            // Apply contact force
            double accel = gravity + F_contact / mass;
            velocity += accel * dt;

            // Record bounce at maximum compression
            if (velocity > 0 && was_falling) {
                tracker.recordBounce(max_height, velocity);
                was_falling = false;
            }
        } else {
            contact_model.reset();
            double accel = gravity;
            velocity += accel * dt;
            was_falling = (velocity < 0);
        }

        // Update position
        position += velocity * dt;

        // Track max height
        if (position > max_height) {
            max_height = position;
        }

        // Bounce detection reset
        if (position > radius && std::abs(velocity) < 0.1) {
            max_height = position;
        }
    }

    // Compute restitution coefficient
    double e = tracker.computeRestitutionCoefficient();
    double energy_loss = tracker.computeEnergyLoss();

    std::cout << "Number of bounces: " << tracker.bounceCount() << std::endl;
    std::cout << "Restitution coefficient: " << e << std::endl;
    std::cout << "Energy loss per bounce: " << energy_loss << "%" << std::endl;

    // Theoretical energy loss from hysteresis
    double theoretical_loss = params.alpha * 100.0;  // alpha is the hysteresis factor

    std::cout << "Theoretical loss (from alpha): " << theoretical_loss << "%" << std::endl;

    double error = std::abs(energy_loss - theoretical_loss);
    std::cout << "Error: " << error << "%" << std::endl;

    bool passed = error < 5.0;  // Within 5%

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test Nayak surface model
 */
bool testNayakSurface() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Nayak Surface Model Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create rough surface
    RoughSurfaceParams params;
    params.sigma = 1e-6;      // 1 micron
    params.beta_star = 1e-4;  // 100 microns
    params.eta = 1e8;         // Asperity density

    NayakSurface surface(params);

    // Compute spectral moments
    double m0, m2, m4;
    surface.computeSpectralMoments(m0, m2, m4);

    std::cout << "RMS height (sigma): " << surface.sigma() << " m" << std::endl;
    std::cout << "Correlation length: " << surface.correlationLength() << " m" << std::endl;
    std::cout << "Spectral moments: m0=" << m0 << ", m2=" << m2 << ", m4=" << m4 << std::endl;

    // Verify spectral moments
    double expected_m0 = params.sigma * params.sigma;
    double expected_m2 = expected_m0 / (params.beta_star * params.beta_star);

    bool m0_correct = std::abs(m0 - expected_m0) < 1e-20;
    bool m2_correct = std::abs(m2 - expected_m2) < 1e-10;

    std::cout << "m0 correct: " << (m0_correct ? "YES" : "NO") << std::endl;
    std::cout << "m2 correct: " << (m2_correct ? "YES" : "NO") << std::endl;

    // Compute asperity properties
    double kappa = surface.meanAsperityCurvature();
    double sigma_s = surface.asperityHeightStdDev();

    std::cout << "Mean asperity curvature: " << kappa << std::endl;
    std::cout << "Asperity height std dev: " << sigma_s << std::endl;

    bool passed = m0_correct && m2_correct;

    std::cout << "Result: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

/**
 * @brief Test viscoelastic models
 */
bool testViscoelasticModels() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Viscoelastic Models Test" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    // Test Maxwell element
    {
        MaxwellElement maxwell(1000.0, 100.0);  // k=1000, eta=100

        double dt = 0.01;
        double displacement = 0.1;

        // Run multiple steps to reach steady state
        double force = 0.0;
        for (int i = 0; i < 100; ++i) {
            force = maxwell.update(displacement, dt);
        }

        std::cout << "Maxwell element force (steady state): " << force << " N" << std::endl;

        // At steady state with constant displacement, force should decay to 0
        // (Maxwell element is fluid-like)
        bool reasonable = (force >= 0 && force < 100.0);  // Should be close to 0
        std::cout << "Maxwell test: " << (reasonable ? "PASSED" : "FAILED") << std::endl;
        all_passed &= reasonable;
    }

    // Test Hysteresis model
    {
        HysteresisModel hysteresis(1000.0, 0.3);  // k=1000, h=0.3

        // Loading
        double F1 = hysteresis.computeForce(0.1);
        // Unloading
        double F2 = hysteresis.computeForce(0.05);

        std::cout << "Hysteresis loading force: " << F1 << " N" << std::endl;
        std::cout << "Hysteresis unloading force: " << F2 << " N" << std::endl;

        // Loading force should be higher than unloading force
        bool has_hysteresis = F1 > F2;
        std::cout << "Hysteresis test: " << (has_hysteresis ? "PASSED" : "FAILED") << std::endl;
        all_passed &= has_hysteresis;

        // Check dissipated energy
        double dissipated = hysteresis.dissipatedEnergy(0.1);
        std::cout << "Dissipated energy: " << dissipated << " J" << std::endl;
    }

    // Test Standard Linear Solid
    {
        StandardLinearSolid sls(1e6, 0.5e6, 0.1);  // E_inst, E_eq, tau

        double dt = 0.001;
        double force1 = sls.computeForce(0.01, dt);
        double force2 = sls.computeForce(0.01, dt);  // Same displacement

        std::cout << "SLS force (step 1): " << force1 << " N" << std::endl;
        std::cout << "SLS force (step 2): " << force2 << " N" << std::endl;

        // Force should relax over time
        bool relaxes = (force2 <= force1);
        std::cout << "SLS relaxation test: " << (relaxes ? "PASSED" : "FAILED") << std::endl;
        all_passed &= relaxes;
    }

    return all_passed;
}

/**
 * @brief Test Bush-Gibson-Thomas contact area
 */
bool testBGTContactArea() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Bush-Gibson-Thomas Contact Area Test" << std::endl;
    std::cout << "========================================" << std::endl;

    NayakSurface surface = NayakSurface::rough();

    double nominal_area = 1e-4;  // 1 cm^2
    double nominal_pressure = 1e6;  // 1 MPa

    double A_c = BushGibsonThomasModel::computeContactArea(
        surface, nominal_pressure, nominal_area);

    std::cout << "Nominal area: " << nominal_area << " m^2" << std::endl;
    std::cout << "Nominal pressure: " << nominal_pressure << " Pa" << std::endl;
    std::cout << "Real contact area: " << A_c << " m^2" << std::endl;
    std::cout << "Area ratio (A_c/A_n): " << A_c / nominal_area << std::endl;

    // Real contact area should be much smaller than nominal
    bool reasonable = (A_c > 0 && A_c < nominal_area);

    std::cout << "Result: " << (reasonable ? "PASSED" : "FAILED") << std::endl;

    return reasonable;
}

int main() {
    std::cout << "****************************************" << std::endl;
    std::cout << "Phase 4 Acceptance Tests" << std::endl;
    std::cout << "****************************************" << std::endl;

    bool test1 = testCriterion1_HardeningCurve();
    bool test2 = testCriterion2_EnergyDissipation();
    bool test3 = testNayakSurface();
    bool test4 = testViscoelasticModels();
    bool test5 = testBGTContactArea();

    std::cout << "\n****************************************" << std::endl;
    std::cout << "Phase 4 Acceptance Summary" << std::endl;
    std::cout << "****************************************" << std::endl;
    std::cout << "Criterion 1 (Hardening Curve): " << (test1 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Criterion 2 (Energy Dissipation): " << (test2 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Nayak Surface Model: " << (test3 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Viscoelastic Models: " << (test4 ? "PASSED" : "FAILED") << std::endl;
    std::cout << "BGT Contact Area: " << (test5 ? "PASSED" : "FAILED") << std::endl;

    bool all_passed = test1 && test2 && test3 && test4 && test5;

    std::cout << "\nOverall: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << "****************************************" << std::endl;

    return all_passed ? 0 : 1;
}
