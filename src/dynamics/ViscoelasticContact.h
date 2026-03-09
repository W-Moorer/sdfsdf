/**
 * @file ViscoelasticContact.h
 * @brief Viscoelastic contact model with hysteresis
 *
 * Implements Maxwell and Voigt elements for normal and tangential
 * viscoelastic contact with energy dissipation.
 */

#pragma once

#include <Eigen/Core>
#include <cmath>
#include <deque>

namespace vde {

/**
 * @brief Viscoelastic material parameters
 */
struct ViscoelasticParams {
    // Elastic parameters
    double E;           // Young's modulus (Pa)
    double nu;          // Poisson's ratio

    // Viscous parameters (Maxwell element)
    double eta_normal;  // Normal viscosity (Pa*s)
    double eta_tangent; // Tangential viscosity (Pa*s)

    // Hysteresis parameters
    double alpha;       // Hysteresis shape factor
    double beta;        // Recovery rate

    ViscoelasticParams()
        : E(1e6),           // 1 MPa (rubber-like)
          nu(0.45),         // Nearly incompressible
          eta_normal(1e3),  // 1000 Pa*s
          eta_tangent(5e2), // 500 Pa*s
          alpha(0.3),       // 30% hysteresis
          beta(0.1) {}      // Slow recovery
};

/**
 * @brief Maxwell viscoelastic element
 *
 * Series combination of spring and damper
 * F + (eta/k) * dF/dt = eta * d(delta)/dt
 */
class MaxwellElement {
public:
    MaxwellElement(double stiffness, double viscosity)
        : k_(stiffness),
          eta_(viscosity),
          force_(0.0),
          displacement_(0.0) {}

    /**
     * @brief Update force based on displacement
     *
     * @param displacement Current displacement
     * @param dt Time step
     * @return Current force
     */
    double update(double displacement, double dt) {
        double velocity = (displacement - displacement_) / dt;

        // Maxwell model: dF/dt = k * (velocity - F/eta)
        double dF = k_ * (velocity - force_ / eta_) * dt;
        force_ += dF;

        displacement_ = displacement;
        return force_;
    }

    double force() const { return force_; }
    void reset() { force_ = 0.0; displacement_ = 0.0; }

private:
    double k_;          // Spring stiffness
    double eta_;        // Viscosity
    double force_;      // Current force
    double displacement_; // Current displacement
};

/**
 * @brief Voigt viscoelastic element
 *
 * Parallel combination of spring and damper
 * F = k * delta + eta * d(delta)/dt
 */
class VoigtElement {
public:
    VoigtElement(double stiffness, double viscosity)
        : k_(stiffness),
          eta_(viscosity),
          displacement_prev_(0.0) {}

    /**
     * @brief Compute force based on displacement
     *
     * @param displacement Current displacement
     * @param dt Time step
     * @return Current force
     */
    double computeForce(double displacement, double dt) {
        double velocity = (displacement - displacement_prev_) / dt;
        double force = k_ * displacement + eta_ * velocity;

        displacement_prev_ = displacement;
        return force;
    }

    void reset() { displacement_prev_ = 0.0; }

private:
    double k_;          // Spring stiffness
    double eta_;        // Viscosity
    double displacement_prev_; // Previous displacement
};

/**
 * @brief Standard Linear Solid (SLS) model
 *
 * Combination of Maxwell element in parallel with spring
 * Most accurate for modeling rubber-like materials
 */
class StandardLinearSolid {
public:
    StandardLinearSolid(double E_instant, double E_equilibrium, double tau)
        : E_inst_(E_instant),
          E_eq_(E_equilibrium),
          tau_(tau),
          displacement_(0.0),
          force_viscous_(0.0) {}

    /**
     * @brief Compute force
     *
     * F = E_eq * delta + F_viscous
     * where F_viscous evolves according to Maxwell element
     */
    double computeForce(double displacement, double dt) {
        double velocity = (displacement - displacement_) / dt;

        // Update viscous force (Maxwell element)
        double k_viscous = E_inst_ - E_eq_;
        double eta = tau_ * k_viscous;

        // dF_viscous/dt = k_viscous * velocity - F_viscous / tau
        double dF_viscous = (k_viscous * velocity - force_viscous_ / tau_) * dt;
        force_viscous_ += dF_viscous;

        // Total force
        double force = E_eq_ * displacement + force_viscous_;

        displacement_ = displacement;
        return force;
    }

    void reset() {
        displacement_ = 0.0;
        force_viscous_ = 0.0;
    }

private:
    double E_inst_;     // Instantaneous modulus
    double E_eq_;       // Equilibrium modulus
    double tau_;        // Relaxation time

    double displacement_;
    double force_viscous_;
};

/**
 * @brief Hysteresis model for contact
 *
 * Implements loading/unloading hysteresis loop
 */
class HysteresisModel {
public:
    HysteresisModel(double stiffness, double hysteresis_factor)
        : k_(stiffness),
          h_(hysteresis_factor),
          max_displacement_(0.0),
          loading_(true) {}

    /**
     * @brief Compute force with hysteresis
     *
     * Loading: F = k * delta
     * Unloading: F = k * (delta - h * delta_max)
     */
    double computeForce(double displacement) {
        if (displacement > max_displacement_) {
            // Loading phase
            max_displacement_ = displacement;
            loading_ = true;
            return k_ * displacement;
        } else {
            // Unloading phase
            loading_ = false;
            double force = k_ * (displacement - h_ * max_displacement_);
            return std::max(0.0, force);  // No tension
        }
    }

    /**
     * @brief Get dissipated energy in one cycle
     */
    double dissipatedEnergy(double delta_max) const {
        // Area of hysteresis loop
        return h_ * k_ * delta_max * delta_max;
    }

    void reset() {
        max_displacement_ = 0.0;
        loading_ = true;
    }

    bool isLoading() const { return loading_; }

private:
    double k_;                  // Stiffness
    double h_;                  // Hysteresis factor (0-1)
    double max_displacement_;   // Maximum displacement in current cycle
    bool loading_;              // Current loading state
};

/**
 * @brief Complete viscoelastic contact model
 *
 * Combines elastic, viscous, and hysteretic effects
 */
class ViscoelasticContactModel {
public:
    ViscoelasticContactModel(const ViscoelasticParams& params = ViscoelasticParams())
        : params_(params),
          normal_sls_(params.E, params.E * 0.5, params.eta_normal / params.E),
          hysteresis_(params.E, params.alpha) {}

    /**
     * @brief Compute normal contact force
     *
     * @param penetration Penetration depth (positive when in contact)
     * @param contact_area Contact area
     * @param dt Time step
     * @return Normal force (positive = repulsive)
     */
    double computeNormalForce(double penetration, double contact_area, double dt) {
        if (penetration <= 0) {
            normal_sls_.reset();
            hysteresis_.reset();
            return 0.0;
        }

        // Effective stiffness scales with sqrt(contact_area)
        double area_factor = std::sqrt(contact_area / M_PI);

        // SLS model force
        double elastic_force = normal_sls_.computeForce(penetration, dt);

        // Hysteresis
        double hysteretic_force = hysteresis_.computeForce(penetration);

        // Combined force
        double force = (elastic_force + hysteretic_force) * area_factor;

        return force;
    }

    /**
     * @brief Compute tangential friction force
     *
     * @param tangential_velocity Relative tangential velocity
     * @param normal_force Normal contact force
     * @param dt Time step
     * @return Tangential friction force
     */
    double computeTangentialForce(double tangential_velocity,
                                   double normal_force,
                                   double dt) {
        if (normal_force <= 0) {
            return 0.0;
        }

        // Viscous friction
        double viscous_force = params_.eta_tangent * tangential_velocity;

        // Coulomb friction limit
        double friction_limit = params_.alpha * normal_force;

        // Combined (rate-dependent friction)
        double friction_force = viscous_force;
        if (std::abs(friction_force) > friction_limit) {
            friction_force = friction_limit * (friction_force > 0 ? 1 : -1);
        }

        return friction_force;
    }

    /**
     * @brief Get energy dissipation rate
     */
    double getDissipationRate() const {
        // Approximate from hysteresis model
        return hysteresis_.dissipatedEnergy(1.0) * params_.beta;
    }

    const ViscoelasticParams& parameters() const { return params_; }

    void reset() {
        normal_sls_.reset();
        hysteresis_.reset();
    }

private:
    ViscoelasticParams params_;
    StandardLinearSolid normal_sls_;
    HysteresisModel hysteresis_;
};

/**
 * @brief Bouncing ball energy tracker
 *
 * Tracks energy loss during bouncing for validation
 */
class BouncingBallTracker {
public:
    BouncingBallTracker() : bounce_count_(0) {}

    /**
     * @brief Record bounce height
     */
    void recordBounce(double height, double velocity) {
        bounce_heights_.push_back(height);
        bounce_velocities_.push_back(velocity);
        bounce_count_++;
    }

    /**
     * @brief Compute coefficient of restitution from consecutive bounces
     */
    double computeRestitutionCoefficient() const {
        if (bounce_heights_.size() < 2) {
            return 1.0;  // No data
        }

        // e = sqrt(h_n+1 / h_n)
        double e_sum = 0.0;
        int count = 0;

        for (size_t i = 1; i < bounce_heights_.size(); ++i) {
            if (bounce_heights_[i-1] > 1e-6) {
                double e = std::sqrt(bounce_heights_[i] / bounce_heights_[i-1]);
                e_sum += e;
                count++;
            }
        }

        return count > 0 ? e_sum / count : 1.0;
    }

    /**
     * @brief Compute energy loss percentage
     */
    double computeEnergyLoss() const {
        double e = computeRestitutionCoefficient();
        return (1.0 - e*e) * 100.0;  // Percentage
    }

    int bounceCount() const { return bounce_count_; }

    void reset() {
        bounce_heights_.clear();
        bounce_velocities_.clear();
        bounce_count_ = 0;
    }

private:
    std::vector<double> bounce_heights_;
    std::vector<double> bounce_velocities_;
    int bounce_count_;
};

}  // namespace vde
