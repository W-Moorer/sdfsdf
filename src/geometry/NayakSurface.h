/**
 * @file NayakSurface.h
 * @brief Nayak random surface model for rough surface contact
 *
 * Implements the Nayak model for generating Gaussian distributed rough surface
 * parameters (RMS height sigma, correlation length beta*).
 * Based on Bush-Gibson-Thomas contact theory.
 */

#pragma once

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <Eigen/Core>
#include <cmath>
#include <random>

namespace vde {

/**
 * @brief Rough surface parameters
 */
struct RoughSurfaceParams {
    double sigma;           // RMS height (root mean square)
    double beta_star;       // Correlation length
    double eta;             // Density of asperities

    RoughSurfaceParams()
        : sigma(1e-6),      // 1 micron default
          beta_star(1e-4),  // 100 micron default
          eta(1e8) {}       // 1e8 asperities/m^2 default
};

/**
 * @brief Nayak rough surface model
 *
 * Generates rough surface parameters based on random process theory.
 * The surface is modeled as a Gaussian random field.
 */
class NayakSurface {
public:
    /**
     * @brief Constructor with default parameters
     */
    NayakSurface(const RoughSurfaceParams& params = RoughSurfaceParams())
        : params_(params) {}

    /**
     * @brief Generate random surface height at given position
     *
     * Uses Gaussian random field with exponential correlation
     *
     * @param x X coordinate
     * @param y Y coordinate
     * @return Surface height
     */
    double height(double x, double y) const {
        // Simplified: use Gaussian distribution with correlation
        // In practice, this would use a more sophisticated random field generator
        static thread_local std::mt19937 gen(42);
        std::normal_distribution<> dist(0.0, params_.sigma);

        // Correlation factor based on distance from origin
        double r = std::sqrt(x*x + y*y);
        double correlation = std::exp(-r / params_.beta_star);

        return correlation * dist(gen);
    }

    /**
     * @brief Get RMS height (sigma)
     */
    double sigma() const { return params_.sigma; }

    /**
     * @brief Get correlation length (beta*)
     */
    double correlationLength() const { return params_.beta_star; }

    /**
     * @brief Get asperity density
     */
    double asperityDensity() const { return params_.eta; }

    /**
     * @brief Set parameters
     */
    void setParams(const RoughSurfaceParams& params) { params_ = params; }

    /**
     * @brief Compute spectral moments
     *
     * m0, m2, m4 are the 0th, 2nd, and 4th spectral moments
     * used in the Bush-Gibson-Thomas theory
     */
    void computeSpectralMoments(double& m0, double& m2, double& m4) const {
        // For exponential correlation function:
        // m0 = sigma^2
        // m2 = sigma^2 / beta_star^2
        // m4 = 3 * sigma^2 / beta_star^4

        m0 = params_.sigma * params_.sigma;
        m2 = m0 / (params_.beta_star * params_.beta_star);
        m4 = 3.0 * m0 / std::pow(params_.beta_star, 4);
    }

    /**
     * @brief Compute mean asperity curvature
     */
    double meanAsperityCurvature() const {
        double m0, m2, m4;
        computeSpectralMoments(m0, m2, m4);

        // Mean curvature kappa
        double kappa = (8.0 / 3.0) * std::sqrt(m4 / m2);
        return kappa;
    }

    /**
     * @brief Compute standard deviation of asperity heights
     */
    double asperityHeightStdDev() const {
        double m0, m2, m4;
        computeSpectralMoments(m0, m2, m4);

        // Standard deviation of summit heights
        double sigma_s = std::sqrt(m0 - m2*m2/m4);
        return sigma_s;
    }

    /**
     * @brief Compute equivalent roughness parameter
     *
     * For two rough surfaces in contact
     */
    static double equivalentRoughness(double sigma1, double sigma2) {
        return std::sqrt(sigma1*sigma1 + sigma2*sigma2);
    }

    /**
     * @brief Create smooth surface (for comparison)
     */
    static NayakSurface smooth() {
        RoughSurfaceParams params;
        params.sigma = 1e-12;  // Nearly zero roughness
        params.beta_star = 1e-6;
        params.eta = 1e12;
        return NayakSurface(params);
    }

    /**
     * @brief Create rough surface (high roughness)
     */
    static NayakSurface rough() {
        RoughSurfaceParams params;
        params.sigma = 1e-5;   // 10 microns
        params.beta_star = 5e-5;
        params.eta = 5e8;
        return NayakSurface(params);
    }

private:
    RoughSurfaceParams params_;
};

/**
 * @brief Bush-Gibson-Thomas contact theory
 *
 * Computes effective contact area and mean pressure based on
 * statistical rough surface contact model.
 */
class BushGibsonThomasModel {
public:
    /**
     * @brief Compute effective contact area
     *
     * @param surface Rough surface parameters
     * @param nominal_pressure Nominal contact pressure
     * @param nominal_area Nominal contact area
     * @return Effective (real) contact area
     */
    static double computeContactArea(const NayakSurface& surface,
                                      double nominal_pressure,
                                      double nominal_area) {
        // Simplified BGT model
        // A_c = A_n * (p / p_0)^(2/3) for elastic contact

        double sigma = surface.sigma();
        double E_star = 1e9;  // Effective modulus (default 1 GPa)

        // Characteristic pressure
        double p_0 = E_star * sigma / surface.correlationLength();

        double ratio = nominal_pressure / p_0;
        ratio = std::max(0.0, std::min(1.0, ratio));  // Clamp to [0, 1]

        double A_c = nominal_area * std::pow(ratio, 2.0/3.0);
        return A_c;
    }

    /**
     * @brief Compute mean contact pressure
     *
     * @param surface Rough surface parameters
     * @param penetration Penetration depth
     * @return Mean pressure
     */
    static double computeMeanPressure(const NayakSurface& surface,
                                       double penetration) {
        if (penetration <= 0) {
            return 0.0;
        }

        double sigma = surface.sigma();
        double E_star = 1e9;  // Effective modulus

        // Mean pressure increases with penetration
        // p_bar = E_star * sqrt(penetration / sigma)
        double p_bar = E_star * std::sqrt(penetration / sigma);

        return p_bar;
    }

    /**
     * @brief Compute total contact force using Hertz theory
     *
     * F = (4/3) * E_star * sqrt(R) * delta^(3/2)
     * where R is the effective radius of curvature
     *
     * @param surface Rough surface parameters
     * @param penetration Penetration depth
     * @return Contact force
     */
    static double computeContactForce(const NayakSurface& surface,
                                       double penetration) {
        if (penetration <= 0) {
            return 0.0;
        }

        double sigma = surface.sigma();
        double kappa = surface.meanAsperityCurvature();
        double E_star = 1e9;  // Effective modulus

        // Effective radius from mean asperity curvature
        // R = 1 / kappa
        double R_eff = 1.0 / kappa;

        // Hertz contact force: F = (4/3) * E_star * sqrt(R) * delta^(3/2)
        double force = (4.0 / 3.0) * E_star * std::sqrt(R_eff) * std::pow(penetration, 1.5);

        return force;
    }

    /**
     * @brief Compute contact stiffness
     *
     * dF/d(delta) = 2 * E_star * sqrt(A_c / pi)
     *
     * @param surface Rough surface parameters
     * @param contact_area Contact area
     * @return Contact stiffness
     */
    static double computeContactStiffness(const NayakSurface& surface,
                                           double contact_area) {
        double E_star = 1e9;  // Effective modulus
        double stiffness = 2.0 * E_star * std::sqrt(contact_area / M_PI);
        return stiffness;
    }

    /**
     * @brief Compute number of contacting asperities
     *
     * @param surface Rough surface parameters
     * @param separation Surface separation
     * @param nominal_area Nominal contact area
     * @return Number of contacting asperities
     */
    static double computeContactingAsperities(const NayakSurface& surface,
                                               double separation,
                                               double nominal_area) {
        double eta = surface.asperityDensity();
        double sigma_s = surface.asperityHeightStdDev();

        // Probability of contact for a given separation
        // P(contact) = exp(-d^2 / (2*sigma_s^2))
        double prob = std::exp(-separation*separation / (2.0*sigma_s*sigma_s));

        double N = eta * nominal_area * prob;
        return N;
    }
};

}  // namespace vde
