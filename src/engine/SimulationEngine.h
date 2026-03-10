/**
 * @file SimulationEngine.h
 * @brief Main simulation engine integrating the paper's contact pipeline
 */

#pragma once

#include "collision/SpatialHash.h"
#include "dynamics/RigidBody.h"
#include "dynamics/ContactDynamics.h"
#include "geometry/AnalyticalSDF.h"
#include "geometry/VolumetricIntegrator.h"
#include "solver/SOCCPSolver.h"
#include <array>
#include <chrono>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

namespace vde {

/**
 * @brief Extended contact geometry with compatibility aliases
 */
struct ExtendedContactGeometry : public ContactGeometry {
    double equivalent_gap;
    double contact_area;
    Eigen::Vector3d contact_normal;
    Eigen::Vector3d contact_center;
    double torsion_radius;

    ExtendedContactGeometry()
        : equivalent_gap(0.0),
          contact_area(0.0),
          contact_normal(Eigen::Vector3d::UnitY()),
          contact_center(Eigen::Vector3d::Zero()),
          torsion_radius(0.0) {}

    explicit ExtendedContactGeometry(const ContactGeometry& geometry)
        : ContactGeometry(geometry),
          equivalent_gap(geometry.g_eq),
          contact_area(geometry.volume),
          contact_normal(geometry.n_c),
          contact_center(geometry.p_c),
          torsion_radius(geometry.r_tau) {}
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
    double time_step = 0.001;
    double spatial_hash_cell_size = 1.0;
    double contact_margin = 0.01;
    int max_solver_iterations = 50;
    double solver_tolerance = 1e-6;
    bool enable_friction = true;
    double friction_coefficient = 0.5;
    double baumgarte_gamma = 0.5;
    int contact_grid_resolution = 24;
    double contact_p_norm = 2.0;
    double torsion_regularization = 1e-4;
    Eigen::Vector3d gravity = Eigen::Vector3d(0, -9.81, 0);

    SimulationConfig() = default;
};

/**
 * @brief Registered local geometry for a body
 */
struct RegisteredShape {
    std::shared_ptr<SDF> local_sdf;
    AABB local_aabb;

    bool isValid() const {
        return static_cast<bool>(local_sdf) && local_aabb.isValid();
    }
};

/**
 * @brief Main simulation engine
 */
class SimulationEngine {
public:
    explicit SimulationEngine(const SimulationConfig& config = SimulationConfig())
        : config_(config),
          broad_phase_(config.spatial_hash_cell_size),
          integrator_(config.contact_grid_resolution, config.contact_p_norm),
          current_time_(0.0),
          step_count_(0) {}

    int addBody(const std::shared_ptr<RigidBody>& body) {
        const int id = static_cast<int>(bodies_.size());
        bodies_.push_back(body);
        shapes_.push_back(defaultShapeRegistration());
        return id;
    }

    int addBody(const std::shared_ptr<RigidBody>& body,
                const std::shared_ptr<SDF>& local_sdf,
                const AABB& local_aabb) {
        const int id = addBody(body);
        setBodyShape(id, local_sdf, local_aabb);
        return id;
    }

    void setBodyShape(int body_id,
                      const std::shared_ptr<SDF>& local_sdf,
                      const AABB& local_aabb) {
        if (body_id < 0 || body_id >= static_cast<int>(shapes_.size())) {
            return;
        }

        shapes_[body_id].local_sdf = local_sdf;
        shapes_[body_id].local_aabb = local_aabb;
    }

    size_t numBodies() const {
        return bodies_.size();
    }

    std::shared_ptr<RigidBody> getBody(int id) {
        if (id >= 0 && id < static_cast<int>(bodies_.size())) {
            return bodies_[id];
        }
        return nullptr;
    }

    void clear() {
        bodies_.clear();
        shapes_.clear();
        contacts_.clear();
        current_time_ = 0.0;
        step_count_ = 0;
    }

    SimulationStats step() {
        SimulationStats stats;
        stats.num_bodies = static_cast<int>(bodies_.size());

        auto step_start = std::chrono::high_resolution_clock::now();

        auto broad_start = std::chrono::high_resolution_clock::now();
        const auto potential_pairs = performBroadPhase();
        auto broad_end = std::chrono::high_resolution_clock::now();
        stats.broad_phase_time_ms =
            std::chrono::duration<double, std::milli>(broad_end - broad_start).count();

        auto narrow_start = std::chrono::high_resolution_clock::now();
        contacts_.clear();
        for (const auto& pair : potential_pairs) {
            generateContact(pair.first, pair.second);
        }
        auto narrow_end = std::chrono::high_resolution_clock::now();
        stats.narrow_phase_time_ms =
            std::chrono::duration<double, std::milli>(narrow_end - narrow_start).count();
        stats.num_contacts = static_cast<int>(contacts_.size());

        auto solver_start = std::chrono::high_resolution_clock::now();
        solveContacts(stats);
        auto solver_end = std::chrono::high_resolution_clock::now();
        stats.solver_time_ms =
            std::chrono::duration<double, std::milli>(solver_end - solver_start).count();

        auto dynamics_start = std::chrono::high_resolution_clock::now();
        integrateDynamics();
        auto dynamics_end = std::chrono::high_resolution_clock::now();
        stats.dynamics_time_ms =
            std::chrono::duration<double, std::milli>(dynamics_end - dynamics_start).count();

        current_time_ += config_.time_step;
        step_count_++;
        stats.total_energy = computeTotalEnergy();

        auto step_end = std::chrono::high_resolution_clock::now();
        stats.step_time_ms =
            std::chrono::duration<double, std::milli>(step_end - step_start).count();
        return stats;
    }

    std::vector<SimulationStats> run(double duration) {
        std::vector<SimulationStats> all_stats;
        const int num_steps = static_cast<int>(duration / config_.time_step);
        all_stats.reserve(num_steps);

        for (int i = 0; i < num_steps; ++i) {
            all_stats.push_back(step());
        }

        return all_stats;
    }

    double currentTime() const { return current_time_; }
    int stepCount() const { return step_count_; }
    const std::vector<ContactPair>& getContacts() const { return contacts_; }

    double computeTotalEnergy() const {
        double total = 0.0;
        for (const auto& body : bodies_) {
            total += body->kineticEnergy();
        }
        return total;
    }

    Eigen::Vector3d computeTotalMomentum() const {
        Eigen::Vector3d momentum = Eigen::Vector3d::Zero();
        for (const auto& body : bodies_) {
            momentum += body->linearMomentum();
        }
        return momentum;
    }

    int countPenetrations() const {
        int count = 0;
        for (const auto& contact : contacts_) {
            if (contact.geometry.equivalent_gap < 0.0) {
                count++;
            }
        }
        return count;
    }

    const SimulationConfig& config() const { return config_; }

    void setConfig(const SimulationConfig& config) {
        config_ = config;
        broad_phase_ = BroadPhaseDetector(config.spatial_hash_cell_size);
        integrator_.setGridResolution(config.contact_grid_resolution);
        integrator_.setPNorm(config.contact_p_norm);
    }

private:
    SimulationConfig config_;
    std::vector<std::shared_ptr<RigidBody>> bodies_;
    std::vector<RegisteredShape> shapes_;
    std::vector<ContactPair> contacts_;
    BroadPhaseDetector broad_phase_;
    VolumetricIntegrator integrator_;
    double current_time_;
    int step_count_;

    static RegisteredShape defaultShapeRegistration() {
        RegisteredShape shape;
        shape.local_sdf = std::make_shared<SphereSDF>(Eigen::Vector3d::Zero(), 0.5);
        shape.local_aabb = AABB(
            Eigen::Vector3d(-0.5, -0.5, -0.5),
            Eigen::Vector3d(0.5, 0.5, 0.5));
        return shape;
    }

    std::vector<std::pair<int, int>> performBroadPhase() const {
        std::unordered_map<int, AABB> object_aabbs;

        for (size_t i = 0; i < bodies_.size(); ++i) {
            AABB aabb = computeBodyAABB(static_cast<int>(i));
            aabb.expand(config_.contact_margin);
            object_aabbs[static_cast<int>(i)] = aabb;
        }

        BroadPhaseDetector broad_phase(config_.spatial_hash_cell_size);
        return broad_phase.update(object_aabbs);
    }

    AABB computeBodyAABB(int body_id) const {
        if (body_id < 0 || body_id >= static_cast<int>(bodies_.size())) {
            return AABB();
        }

        const RegisteredShape& shape = shapes_[body_id];
        if (!shape.isValid()) {
            const Eigen::Vector3d pos = bodies_[body_id]->position();
            return AABB(pos - Eigen::Vector3d::Constant(0.5), pos + Eigen::Vector3d::Constant(0.5));
        }

        const RigidBodyState& state = bodies_[body_id]->state();
        return transformLocalAABB(shape.local_aabb, state);
    }

    static AABB transformLocalAABB(const AABB& local_aabb,
                                   const RigidBodyState& state) {
        const Eigen::Vector3d& min = local_aabb.min;
        const Eigen::Vector3d& max = local_aabb.max;

        const std::array<Eigen::Vector3d, 8> local_corners = {{
            Eigen::Vector3d(min.x(), min.y(), min.z()),
            Eigen::Vector3d(min.x(), min.y(), max.z()),
            Eigen::Vector3d(min.x(), max.y(), min.z()),
            Eigen::Vector3d(min.x(), max.y(), max.z()),
            Eigen::Vector3d(max.x(), min.y(), min.z()),
            Eigen::Vector3d(max.x(), min.y(), max.z()),
            Eigen::Vector3d(max.x(), max.y(), min.z()),
            Eigen::Vector3d(max.x(), max.y(), max.z())
        }};

        AABB world_aabb;
        for (const auto& corner : local_corners) {
            world_aabb.expand(state.localToWorld(corner));
        }
        return world_aabb;
    }

    void generateContact(int id_a, int id_b) {
        auto body_a = bodies_[id_a];
        auto body_b = bodies_[id_b];

        if (body_a->isStatic() && body_b->isStatic()) {
            return;
        }

        const RegisteredShape& shape_a = shapes_[id_a];
        const RegisteredShape& shape_b = shapes_[id_b];
        if (!shape_a.isValid() || !shape_b.isValid()) {
            return;
        }

        const TransformedSDF sdf_a(
            shape_a.local_sdf, body_a->state().position, body_a->state().orientation);
        const TransformedSDF sdf_b(
            shape_b.local_sdf, body_b->state().position, body_b->state().orientation);

        const AABB aabb_a = computeBodyAABB(id_a);
        const AABB aabb_b = computeBodyAABB(id_b);

        const ContactGeometry geometry =
            integrator_.computeContactGeometry(sdf_a, sdf_b, aabb_a, aabb_b);
        if (geometry.g_eq >= 0.0 || geometry.volume <= 0.0) {
            return;
        }

        contacts_.emplace_back(id_a, id_b, ExtendedContactGeometry(geometry));
    }

    std::vector<SpatialVector> computeFreeVelocities() const {
        std::vector<SpatialVector> free_velocities(bodies_.size(), SpatialVector::Zero());

        for (size_t i = 0; i < bodies_.size(); ++i) {
            const auto& body = bodies_[i];
            if (body->isStatic()) {
                continue;
            }

            SpatialVector v_free = body->spatialVelocity();
            const SpatialVector wrench = body->externalWrench();
            const Eigen::Matrix<double, 6, 6> Minv = body->inverseMassMatrix();
            SpatialVector acceleration = Minv * wrench;
            acceleration.head<3>() += config_.gravity;
            v_free += config_.time_step * acceleration;
            free_velocities[i] = v_free;
        }

        return free_velocities;
    }

    void solveContacts(SimulationStats& stats) {
        const std::vector<SpatialVector> free_velocities = computeFreeVelocities();

        std::vector<int> active_body_ids;
        std::unordered_map<int, int> active_offsets;
        for (size_t i = 0; i < bodies_.size(); ++i) {
            if (!bodies_[i]->isStatic()) {
                const int offset = static_cast<int>(active_body_ids.size()) * 6;
                active_body_ids.push_back(static_cast<int>(i));
                active_offsets[static_cast<int>(i)] = offset;
            }
        }

        if (active_body_ids.empty()) {
            stats.solver_iterations = 0;
            stats.solver_residual = 0.0;
            return;
        }

        const int nv = static_cast<int>(active_body_ids.size()) * 6;
        Eigen::MatrixXd mass_matrix = Eigen::MatrixXd::Zero(nv, nv);
        Eigen::VectorXd global_free_velocity = Eigen::VectorXd::Zero(nv);

        for (size_t i = 0; i < active_body_ids.size(); ++i) {
            const int body_id = active_body_ids[i];
            const int offset = static_cast<int>(i) * 6;
            mass_matrix.block<6, 6>(offset, offset) = bodies_[body_id]->massMatrix();
            global_free_velocity.segment<6>(offset) = free_velocities[body_id];
        }

        Eigen::VectorXd solved_velocity = global_free_velocity;
        SOCCPSolution solution;

        if (!contacts_.empty()) {
            const int nc = static_cast<int>(contacts_.size());
            SOCCPProblem problem;
            problem.mass_matrix = mass_matrix;
            problem.free_velocity = global_free_velocity;
            problem.normal_jacobian = Eigen::MatrixXd::Zero(nc, nv);
            problem.tangential_jacobian = Eigen::MatrixXd::Zero(3 * nc, nv);
            problem.g_eq = Eigen::VectorXd::Zero(nc);
            problem.friction_coefficients = Eigen::VectorXd::Zero(nc);
            problem.baumgarte_gamma = Eigen::VectorXd::Zero(nc);
            problem.time_step = config_.time_step;
            problem.epsilon = 1e-8;

            for (int contact_id = 0; contact_id < nc; ++contact_id) {
                const ContactPair& pair = contacts_[contact_id];
                const auto& body_a = bodies_[pair.body_a_id];
                const auto& body_b = bodies_[pair.body_b_id];
                const RegisteredShape& shape_a = shapes_[pair.body_a_id];
                const RegisteredShape& shape_b = shapes_[pair.body_b_id];

                ContactConstraint constraint;
                constraint.computeJacobiansFromGeometry(
                    pair.geometry,
                    body_a->centerOfMassWorld(),
                    body_b->centerOfMassWorld(),
                    config_.torsion_regularization);

                AABB fixed_aabb_a = computeBodyAABB(pair.body_a_id);
                AABB fixed_aabb_b = computeBodyAABB(pair.body_b_id);
                const double fd_padding = std::max(1e-5, 10.0 * config_.torsion_regularization);
                fixed_aabb_a.expand(fd_padding);
                fixed_aabb_b.expand(fd_padding);

                constraint.J_n = computeIntegratedNormalJacobianFiniteDifference(
                    integrator_,
                    shape_a.local_sdf,
                    fixed_aabb_a,
                    body_a->state().position,
                    body_a->state().orientation,
                    shape_b.local_sdf,
                    fixed_aabb_b,
                    body_b->state().position,
                    body_b->state().orientation,
                    1e-5);
                constraint.J_f.row(0) = constraint.J_n;

                Eigen::RowVectorXd reduced_normal = Eigen::RowVectorXd::Zero(nv);
                Eigen::MatrixXd reduced_tangent = Eigen::MatrixXd::Zero(3, nv);

                if (!body_a->isStatic()) {
                    const int offset = active_offsets[pair.body_a_id];
                    reduced_normal.block(0, offset, 1, 6) = constraint.J_n.block<1, 6>(0, 0);
                    reduced_tangent.block(0, offset, 3, 6) = constraint.tangentialJacobian().block<3, 6>(0, 0);
                }

                if (!body_b->isStatic()) {
                    const int offset = active_offsets[pair.body_b_id];
                    reduced_normal.block(0, offset, 1, 6) = constraint.J_n.block<1, 6>(0, 6);
                    reduced_tangent.block(0, offset, 3, 6) = constraint.tangentialJacobian().block<3, 6>(0, 6);
                }

                problem.normal_jacobian.row(contact_id) = reduced_normal;
                problem.tangential_jacobian.block(3 * contact_id, 0, 3, nv) = reduced_tangent;
                problem.g_eq(contact_id) = pair.geometry.g_eq;
                problem.friction_coefficients(contact_id) =
                    config_.enable_friction ? config_.friction_coefficient : 0.0;
                problem.baumgarte_gamma(contact_id) = config_.baumgarte_gamma;
            }

            SOCCPSolver solver(
                config_.max_solver_iterations,
                config_.solver_tolerance,
                problem.epsilon);
            const bool converged = solver.solve(problem, solution);

            if (converged) {
                solved_velocity = solution.velocity;
            }

            stats.solver_iterations = solution.stats.iterations;
            stats.solver_residual =
                solution.residual.size() > 0
                    ? solution.residual.lpNorm<Eigen::Infinity>()
                    : (converged ? 0.0 : std::numeric_limits<double>::infinity());
        } else {
            stats.solver_iterations = 0;
            stats.solver_residual = 0.0;
        }

        for (size_t i = 0; i < active_body_ids.size(); ++i) {
            const int body_id = active_body_ids[i];
            const int offset = static_cast<int>(i) * 6;
            bodies_[body_id]->setSpatialVelocity(solved_velocity.segment<6>(offset));
            bodies_[body_id]->clearForces();
        }

        for (const auto& body : bodies_) {
            if (body->isStatic()) {
                body->clearForces();
            }
        }
    }

    void integrateDynamics() {
        for (auto& body : bodies_) {
            body->integrate(config_.time_step, Eigen::Vector3d::Zero());
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
        if (stats.empty()) {
            return profile;
        }

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

        const size_t n = stats.size();
        profile.avg_step_time = total_step / n;
        profile.avg_broad_phase_time = total_broad / n;
        profile.avg_narrow_phase_time = total_narrow / n;
        profile.avg_solver_time = total_solver / n;
        profile.avg_dynamics_time = total_dynamics / n;

        return profile;
    }
};

}  // namespace vde
