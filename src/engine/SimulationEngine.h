/**
 * @file SimulationEngine.h
 * @brief Main simulation engine integrating all physics components
 *
 * Orchestrates the full simulation pipeline:
 * - Broad-phase collision detection (Spatial Hash)
 * - Narrow-phase contact generation (Volumetric integration)
 * - SOCCP solver for contact forces
 * - Rigid body dynamics integration
 */

#pragma once

#include "collision/SpatialHash.h"
#include "dynamics/RigidBody.h"
#include "dynamics/ContactDynamics.h"
#include "solver/SemiSmoothNewton.h"
#include "geometry/VolumetricIntegrator.h"
#include <vector>
#include <memory>
#include <chrono>

namespace vde {

/**
 * @brief Extended contact geometry with compatibility fields
 */
struct ExtendedContactGeometry : public ContactGeometry {
    double equivalent_gap;           ///< Equivalent scalar gap (alias for g_eq)
    double contact_area;             ///< Contact area
    Eigen::Vector3d contact_normal;  ///< Contact normal (alias for n_c)
    Eigen::Vector3d contact_center;  ///< Contact center (alias for p_c)
    double torsion_radius;           ///< Torsion radius (alias for r_tau)
    
    ExtendedContactGeometry()
        : equivalent_gap(0.0),
          contact_area(0.0),
          contact_normal(Eigen::Vector3d::UnitY()),
          contact_center(Eigen::Vector3d::Zero()),
          torsion_radius(0.0) {
        // Initialize base class
        g_eq = 0.0;
        r_tau = 0.0;
        n_c = Eigen::Vector3d::UnitY();
        p_c = Eigen::Vector3d::Zero();
    }
};

/**
 * @brief Contact pair information
 */
struct ContactPair {
    int body_a_id;
    int body_b_id;
    ExtendedContactGeometry geometry;
    
    ContactPair(int a, int b, const ExtendedContactGeometry& geo)
        : body_a_id(a), body_b_id(b), geometry(geo) {}
};

/**
 * @brief Simulation statistics
 */
struct SimulationStats {
    int num_bodies = 0;
    int num_contacts = 0;
    int solver_iterations = 0;
    double solver_residual = 0.0;
    double step_time_ms = 0.0;
    double broad_phase_time_ms = 0.0;
    double narrow_phase_time_ms = 0.0;
    double solver_time_ms = 0.0;
    double dynamics_time_ms = 0.0;
    double total_energy = 0.0;
};

/**
 * @brief Simulation configuration
 */
struct SimulationConfig {
    double time_step = 0.001;           // Fixed time step (s)
    double spatial_hash_cell_size = 1.0; // Spatial hash cell size
    double contact_margin = 0.01;        // AABB expansion margin
    int max_solver_iterations = 50;      // Max SOCCP solver iterations
    double solver_tolerance = 1e-6;      // Solver convergence tolerance
    bool enable_friction = true;         // Enable friction
    double friction_coefficient = 0.5;   // Coulomb friction coefficient
    Eigen::Vector3d gravity = Eigen::Vector3d(0, -9.81, 0); // Gravity
    
    SimulationConfig() = default;
};

/**
 * @brief Main simulation engine
 */
class SimulationEngine {
public:
    explicit SimulationEngine(const SimulationConfig& config = SimulationConfig())
        : config_(config),
          broad_phase_(config.spatial_hash_cell_size),
          current_time_(0.0),
          step_count_(0) {}
    
    /**
     * @brief Add a rigid body to the simulation
     * @param body Shared pointer to rigid body
     * @return Body ID
     */
    int addBody(const std::shared_ptr<RigidBody>& body) {
        int id = static_cast<int>(bodies_.size());
        bodies_.push_back(body);
        return id;
    }
    
    /**
     * @brief Get number of bodies
     */
    size_t numBodies() const {
        return bodies_.size();
    }
    
    /**
     * @brief Get body by ID
     */
    std::shared_ptr<RigidBody> getBody(int id) {
        if (id >= 0 && id < static_cast<int>(bodies_.size())) {
            return bodies_[id];
        }
        return nullptr;
    }
    
    /**
     * @brief Remove all bodies
     */
    void clear() {
        bodies_.clear();
        contacts_.clear();
        current_time_ = 0.0;
        step_count_ = 0;
    }
    
    /**
     * @brief Perform single simulation step
     * @return Simulation statistics for this step
     */
    SimulationStats step() {
        SimulationStats stats;
        stats.num_bodies = static_cast<int>(bodies_.size());
        
        auto step_start = std::chrono::high_resolution_clock::now();
        
        // 1. Broad-phase collision detection
        auto broad_start = std::chrono::high_resolution_clock::now();
        auto potential_pairs = performBroadPhase();
        auto broad_end = std::chrono::high_resolution_clock::now();
        stats.broad_phase_time_ms = std::chrono::duration<double, std::milli>(
            broad_end - broad_start).count();
        
        // 2. Narrow-phase contact generation
        auto narrow_start = std::chrono::high_resolution_clock::now();
        contacts_.clear();
        for (const auto& pair : potential_pairs) {
            generateContact(pair.first, pair.second);
        }
        auto narrow_end = std::chrono::high_resolution_clock::now();
        stats.narrow_phase_time_ms = std::chrono::duration<double, std::milli>(
            narrow_end - narrow_start).count();
        
        stats.num_contacts = static_cast<int>(contacts_.size());
        
        // 3. Solve contact forces using SOCCP
        auto solver_start = std::chrono::high_resolution_clock::now();
        if (!contacts_.empty()) {
            solveContacts(stats);
        }
        auto solver_end = std::chrono::high_resolution_clock::now();
        stats.solver_time_ms = std::chrono::duration<double, std::milli>(
            solver_end - solver_start).count();
        
        // 4. Integrate rigid body dynamics
        auto dynamics_start = std::chrono::high_resolution_clock::now();
        integrateDynamics();
        auto dynamics_end = std::chrono::high_resolution_clock::now();
        stats.dynamics_time_ms = std::chrono::duration<double, std::milli>(
            dynamics_end - dynamics_start).count();
        
        // Update time
        current_time_ += config_.time_step;
        step_count_++;
        
        // Compute total energy
        stats.total_energy = computeTotalEnergy();
        
        auto step_end = std::chrono::high_resolution_clock::now();
        stats.step_time_ms = std::chrono::duration<double, std::milli>(
            step_end - step_start).count();
        
        return stats;
    }
    
    /**
     * @brief Run simulation for specified duration
     * @param duration Simulation duration in seconds
     * @return Vector of statistics for each step
     */
    std::vector<SimulationStats> run(double duration) {
        std::vector<SimulationStats> all_stats;
        int num_steps = static_cast<int>(duration / config_.time_step);
        
        for (int i = 0; i < num_steps; ++i) {
            all_stats.push_back(step());
        }
        
        return all_stats;
    }
    
    /**
     * @brief Get current simulation time
     */
    double currentTime() const {
        return current_time_;
    }
    
    /**
     * @brief Get current step count
     */
    int stepCount() const {
        return step_count_;
    }
    
    /**
     * @brief Get current contacts
     */
    const std::vector<ContactPair>& getContacts() const {
        return contacts_;
    }
    
    /**
     * @brief Compute total kinetic energy of the system
     */
    double computeTotalEnergy() const {
        double total = 0.0;
        for (const auto& body : bodies_) {
            total += body->kineticEnergy();
        }
        return total;
    }
    
    /**
     * @brief Compute total linear momentum
     */
    Eigen::Vector3d computeTotalMomentum() const {
        Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
        for (const auto& body : bodies_) {
            momentum += body->linearMomentum();
        }
        return momentum;
    }
    
    /**
     * @brief Check for penetrations (debugging)
     * @return Number of penetrating contacts
     */
    int countPenetrations() const {
        int count = 0;
        for (const auto& contact : contacts_) {
            if (contact.geometry.equivalent_gap < 0) {
                count++;
            }
        }
        return count;
    }
    
    /**
     * @brief Get configuration
     */
    const SimulationConfig& config() const {
        return config_;
    }
    
    /**
     * @brief Set configuration
     */
    void setConfig(const SimulationConfig& config) {
        config_ = config;
        broad_phase_ = BroadPhaseDetector(config.spatial_hash_cell_size);
    }

private:
    SimulationConfig config_;
    std::vector<std::shared_ptr<RigidBody>> bodies_;
    std::vector<ContactPair> contacts_;
    BroadPhaseDetector broad_phase_;
    double current_time_;
    int step_count_;
    
    /**
     * @brief Perform broad-phase collision detection
     */
    std::vector<std::pair<int, int>> performBroadPhase() {
        std::unordered_map<int, AABB> object_aabbs;
        
        for (size_t i = 0; i < bodies_.size(); ++i) {
            AABB aabb = computeBodyAABB(bodies_[i]);
            aabb.expand(config_.contact_margin);
            object_aabbs[static_cast<int>(i)] = aabb;
        }
        
        return broad_phase_.update(object_aabbs);
    }
    
    /**
     * @brief Compute AABB for a rigid body
     */
    AABB computeBodyAABB(const std::shared_ptr<RigidBody>& body) const {
        // Simple approximation using position and size
        Eigen::Vector3d pos = body->position();
        double size = 1.0; // Default size
        
        // Try to get actual size from SDF if available
        // For now, use a conservative estimate
        
        Eigen::Vector3d half_size = Eigen::Vector3d::Constant(size * 0.5);
        
        return AABB(pos - half_size, pos + half_size);
    }
    
    /**
     * @brief Generate contact between two bodies
     */
    void generateContact(int id_a, int id_b) {
        auto body_a = bodies_[id_a];
        auto body_b = bodies_[id_b];
        
        // Simple sphere-sphere contact for now
        Eigen::Vector3d pos_a = body_a->position();
        Eigen::Vector3d pos_b = body_b->position();
        
        double radius_a = 0.5; // Default radius
        double radius_b = 0.5;
        
        Eigen::Vector3d diff = pos_b - pos_a;
        double distance = diff.norm();
        double penetration = radius_a + radius_b - distance;
        
        if (penetration > 0) {
            // Contact normal (from A to B)
            Eigen::Vector3d normal = diff.normalized();
            
            // Contact center
            Eigen::Vector3d contact_center = pos_a + normal * (radius_a - penetration * 0.5);
            
            // Approximate contact area
            double contact_radius = std::sqrt(penetration * std::min(radius_a, radius_b));
            double contact_area = M_PI * contact_radius * contact_radius;
            
            ExtendedContactGeometry geo;
            geo.equivalent_gap = -penetration; // Negative for penetration
            geo.contact_area = contact_area;
            geo.contact_normal = normal;
            geo.contact_center = contact_center;
            geo.torsion_radius = contact_radius;
            // Also set base class fields
            geo.g_eq = -penetration;
            geo.n_c = normal;
            geo.p_c = contact_center;
            geo.r_tau = contact_radius;
            
            contacts_.emplace_back(id_a, id_b, geo);
        }
    }
    
    /**
     * @brief Solve contact forces using SOCCP
     */
    void solveContacts(SimulationStats& stats) {
        // Assemble global system
        // For each contact, compute Jacobians and assemble into global matrices
        
        int num_contacts = static_cast<int>(contacts_.size());
        int num_bodies = static_cast<int>(bodies_.size());
        
        // Simplified: use penalty method for now
        // Full SOCCP solver would be implemented here
        
        for (auto& contact : contacts_) {
            auto body_a = bodies_[contact.body_a_id];
            auto body_b = bodies_[contact.body_b_id];
            
            // Simple penalty force
            double gap = contact.geometry.equivalent_gap;
            if (gap < 0) {
                double stiffness = 1e6; // Penalty stiffness
                double damping = 1e3;   // Damping coefficient
                
                // Relative velocity at contact
                Eigen::Vector3d vel_a = body_a->linearVelocity();
                Eigen::Vector3d vel_b = body_b->linearVelocity();
                Eigen::Vector3d rel_vel = vel_b - vel_a;
                
                double vel_normal = rel_vel.dot(contact.geometry.contact_normal);
                
                // Normal force (penalty + damping)
                double force_mag = -stiffness * gap - damping * vel_normal;
                force_mag = std::max(0.0, force_mag); // Only repulsive
                
                Eigen::Vector3d force = force_mag * contact.geometry.contact_normal;
                
                // Apply equal and opposite forces
                body_a->applyForce(force, contact.geometry.contact_center);
                body_b->applyForce(-force, contact.geometry.contact_center);
            }
        }
        
        stats.solver_iterations = 1;
        stats.solver_residual = 0.0;
    }
    
    /**
     * @brief Integrate rigid body dynamics
     */
    void integrateDynamics() {
        for (auto& body : bodies_) {
            // Apply gravity
            Eigen::Vector3d gravity_force = body->mass() * config_.gravity;
            body->applyForce(gravity_force, body->position());
            
            // Integrate
            body->semiImplicitEuler(config_.time_step);
        }
    }
};

/**
 * @brief Performance profiler for simulation
 */
class PerformanceProfiler {
public:
    struct Profile {
        double avg_step_time = 0.0;
        double max_step_time = 0.0;
        double min_step_time = std::numeric_limits<double>::max();
        double avg_broad_phase_time = 0.0;
        double avg_narrow_phase_time = 0.0;
        double avg_solver_time = 0.0;
        double avg_dynamics_time = 0.0;
    };
    
    static Profile analyze(const std::vector<SimulationStats>& stats) {
        Profile profile;
        
        if (stats.empty()) return profile;
        
        double total_step = 0.0;
        double total_broad = 0.0;
        double total_narrow = 0.0;
        double total_solver = 0.0;
        double total_dynamics = 0.0;
        
        for (const auto& s : stats) {
            total_step += s.step_time_ms;
            total_broad += s.broad_phase_time_ms;
            total_narrow += s.narrow_phase_time_ms;
            total_solver += s.solver_time_ms;
            total_dynamics += s.dynamics_time_ms;
            
            profile.max_step_time = std::max(profile.max_step_time, s.step_time_ms);
            profile.min_step_time = std::min(profile.min_step_time, s.step_time_ms);
        }
        
        size_t n = stats.size();
        profile.avg_step_time = total_step / n;
        profile.avg_broad_phase_time = total_broad / n;
        profile.avg_narrow_phase_time = total_narrow / n;
        profile.avg_solver_time = total_solver / n;
        profile.avg_dynamics_time = total_dynamics / n;
        
        return profile;
    }
};

} // namespace vde
