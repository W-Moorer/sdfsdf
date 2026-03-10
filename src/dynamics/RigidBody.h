/**
 * @file RigidBody.h
 * @brief Rigid body dynamics implementation
 *
 * Implements rigid body pose (position + orientation) and velocity updates
 * using semi-implicit Euler integration.
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace vde {

/**
 * @brief 6D spatial vector (3 linear + 3 angular)
 */
using SpatialVector = Eigen::Matrix<double, 6, 1>;

/**
 * @brief Rigid body state
 *
 * Contains pose (position + orientation) and velocities
 */
struct RigidBodyState {
    // Pose
    Eigen::Vector3d position;           // World position
    Eigen::Quaterniond orientation;     // World orientation

    // Velocities
    Eigen::Vector3d linear_velocity;    // World linear velocity
    Eigen::Vector3d angular_velocity;   // World angular velocity (exponential coords)

    RigidBodyState()
        : position(Eigen::Vector3d::Zero()),
          orientation(Eigen::Quaterniond::Identity()),
          linear_velocity(Eigen::Vector3d::Zero()),
          angular_velocity(Eigen::Vector3d::Zero()) {}

    /**
     * @brief Get rotation matrix
     */
    Eigen::Matrix3d rotationMatrix() const {
        return orientation.toRotationMatrix();
    }

    /**
     * @brief Transform point from local to world
     */
    Eigen::Vector3d localToWorld(const Eigen::Vector3d& local_point) const {
        return position + rotationMatrix() * local_point;
    }

    /**
     * @brief Transform point from world to local
     */
    Eigen::Vector3d worldToLocal(const Eigen::Vector3d& world_point) const {
        return rotationMatrix().transpose() * (world_point - position);
    }

    /**
     * @brief Transform vector from local to world
     */
    Eigen::Vector3d localVectorToWorld(const Eigen::Vector3d& local_vector) const {
        return rotationMatrix() * local_vector;
    }
};

/**
 * @brief Rigid body properties
 */
struct RigidBodyProperties {
    double mass;                        // Mass
    Eigen::Matrix3d inertia_local;      // Inertia tensor in body frame
    Eigen::Vector3d com_local;          // Center of mass in body frame

    RigidBodyProperties()
        : mass(1.0),
          inertia_local(Eigen::Matrix3d::Identity()),
          com_local(Eigen::Vector3d::Zero()) {}

    /**
     * @brief Create sphere properties
     */
    static RigidBodyProperties sphere(double mass, double radius) {
        RigidBodyProperties props;
        props.mass = mass;
        double I = 0.4 * mass * radius * radius;
        props.inertia_local = I * Eigen::Matrix3d::Identity();
        return props;
    }

    /**
     * @brief Create box properties
     */
    static RigidBodyProperties box(double mass,
                                    double width,
                                    double height,
                                    double depth) {
        RigidBodyProperties props;
        props.mass = mass;
        double Ix = (mass / 12.0) * (height * height + depth * depth);
        double Iy = (mass / 12.0) * (width * width + depth * depth);
        double Iz = (mass / 12.0) * (width * width + height * height);
        props.inertia_local.setZero();
        props.inertia_local(0, 0) = Ix;
        props.inertia_local(1, 1) = Iy;
        props.inertia_local(2, 2) = Iz;
        return props;
    }
};

/**
 * @brief Rigid body class
 */
class RigidBody {
public:
    RigidBody(const RigidBodyProperties& props = RigidBodyProperties())
        : properties_(props) {}

    /**
     * @brief Constructor with mass only (for simplified usage)
     */
    explicit RigidBody(double mass)
        : properties_(RigidBodyProperties()) {
        properties_.mass = mass;
    }

    /**
     * @brief Get current state
     */
    const RigidBodyState& state() const { return state_; }
    RigidBodyState& state() { return state_; }

    /**
     * @brief Get properties
     */
    const RigidBodyProperties& properties() const { return properties_; }

    /**
     * @brief Get mass matrix (6x6)
     */
    Eigen::Matrix<double, 6, 6> massMatrix() const {
        if (is_static_) {
            return Eigen::Matrix<double, 6, 6>::Zero();
        }
        Eigen::Matrix<double, 6, 6> M = Eigen::Matrix<double, 6, 6>::Zero();
        M.block<3, 3>(0, 0) = properties_.mass * Eigen::Matrix3d::Identity();
        M.block<3, 3>(3, 3) = inertiaTensorWorld();
        return M;
    }

    /**
     * @brief Get inverse mass matrix (6x6)
     */
    Eigen::Matrix<double, 6, 6> inverseMassMatrix() const {
        if (is_static_) {
            return Eigen::Matrix<double, 6, 6>::Zero();
        }
        Eigen::Matrix<double, 6, 6> Minv = Eigen::Matrix<double, 6, 6>::Zero();
        Minv.block<3, 3>(0, 0) = (1.0 / properties_.mass) * Eigen::Matrix3d::Identity();
        Minv.block<3, 3>(3, 3) = inertiaTensorWorld().inverse();
        return Minv;
    }

    /**
     * @brief Get inertia tensor in world frame
     */
    Eigen::Matrix3d inertiaTensorWorld() const {
        const Eigen::Matrix3d R = state_.rotationMatrix();
        return R * properties_.inertia_local * R.transpose();
    }

    /**
     * @brief Get center of mass in world frame
     */
    Eigen::Vector3d centerOfMassWorld() const {
        return state_.localToWorld(properties_.com_local);
    }

    /**
     * @brief Apply force at center of mass
     */
    void applyForce(const Eigen::Vector3d& force) {
        external_force_ += force;
    }

    /**
     * @brief Apply torque
     */
    void applyTorque(const Eigen::Vector3d& torque) {
        external_torque_ += torque;
    }

    /**
     * @brief Apply force at point
     */
    void applyForceAtPoint(const Eigen::Vector3d& force,
                           const Eigen::Vector3d& world_point) {
        external_force_ += force;
        Eigen::Vector3d r = world_point - centerOfMassWorld();
        external_torque_ += r.cross(force);
    }

    /**
     * @brief Apply impulse at center of mass
     */
    void applyImpulse(const Eigen::Vector3d& impulse) {
        state_.linear_velocity += impulse / properties_.mass;
    }

    /**
     * @brief Apply angular impulse
     */
    void applyAngularImpulse(const Eigen::Vector3d& angular_impulse) {
        Eigen::Matrix3d I_inv = inertiaTensorWorld().inverse();
        state_.angular_velocity += I_inv * angular_impulse;
    }

    /**
     * @brief Apply spatial impulse (6D)
     */
    void applySpatialImpulse(const SpatialVector& impulse) {
        applyImpulse(impulse.head<3>());
        applyAngularImpulse(impulse.tail<3>());
    }

    /**
     * @brief Clear external forces
     */
    void clearForces() {
        external_force_.setZero();
        external_torque_.setZero();
    }

    /**
     * @brief Get accumulated external force
     */
    const Eigen::Vector3d& externalForce() const {
        return external_force_;
    }

    /**
     * @brief Get accumulated external torque
     */
    const Eigen::Vector3d& externalTorque() const {
        return external_torque_;
    }

    /**
     * @brief Get accumulated external wrench [f; tau]
     */
    SpatialVector externalWrench() const {
        SpatialVector wrench;
        wrench.head<3>() = external_force_;
        wrench.tail<3>() = external_torque_;
        return wrench;
    }

    /**
     * @brief Semi-implicit Euler integration step
     *
     * @param dt Time step
     * @param gravity Gravity vector
     */
    void integrate(double dt, const Eigen::Vector3d& gravity = Eigen::Vector3d::Zero()) {
        if (is_static_) {
            clearForces();
            state_.linear_velocity.setZero();
            state_.angular_velocity.setZero();
            return;
        }

        // Add gravity
        external_force_ += properties_.mass * gravity;

        // Linear acceleration
        Eigen::Vector3d linear_accel = external_force_ / properties_.mass;

        // Angular acceleration: tau = I * alpha + omega x (I * omega)
        Eigen::Matrix3d I = inertiaTensorWorld();
        Eigen::Vector3d L = I * state_.angular_velocity;
        Eigen::Vector3d gyroscopic = state_.angular_velocity.cross(L);
        Eigen::Vector3d angular_accel = I.inverse() * (external_torque_ - gyroscopic);

        // Update velocities (implicit)
        state_.linear_velocity += linear_accel * dt;
        state_.angular_velocity += angular_accel * dt;

        // Update pose
        state_.position += state_.linear_velocity * dt;

        // Update orientation using exponential map
        Eigen::Vector3d omega_dt = state_.angular_velocity * dt;
        double angle = omega_dt.norm();
        if (angle > 1e-10) {
            Eigen::Vector3d axis = omega_dt / angle;
            Eigen::Quaterniond dq(Eigen::AngleAxisd(angle, axis));
            state_.orientation = (dq * state_.orientation).normalized();
        }

        // Clear forces for next step
        clearForces();
    }

    /**
     * @brief Get kinetic energy
     */
    double kineticEnergy() const {
        double linear_KE = 0.5 * properties_.mass * state_.linear_velocity.squaredNorm();
        Eigen::Matrix3d I = inertiaTensorWorld();
        double angular_KE = 0.5 * state_.angular_velocity.dot(I * state_.angular_velocity);
        return linear_KE + angular_KE;
    }

    /**
     * @brief Get potential energy
     */
    double potentialEnergy(const Eigen::Vector3d& gravity) const {
        return -properties_.mass * gravity.dot(centerOfMassWorld());
    }

    /**
     * @brief Get total energy
     */
    double totalEnergy(const Eigen::Vector3d& gravity) const {
        return kineticEnergy() + potentialEnergy(gravity);
    }

    /**
     * @brief Get position
     */
    Eigen::Vector3d position() const {
        return state_.position;
    }

    /**
     * @brief Set position
     */
    void setPosition(const Eigen::Vector3d& pos) {
        state_.position = pos;
    }

    /**
     * @brief Get linear velocity
     */
    Eigen::Vector3d linearVelocity() const {
        return state_.linear_velocity;
    }

    /**
     * @brief Set linear velocity
     */
    void setLinearVelocity(const Eigen::Vector3d& vel) {
        state_.linear_velocity = vel;
    }

    /**
     * @brief Get angular velocity
     */
    Eigen::Vector3d angularVelocity() const {
        return state_.angular_velocity;
    }

    /**
     * @brief Set angular velocity
     */
    void setAngularVelocity(const Eigen::Vector3d& vel) {
        state_.angular_velocity = vel;
    }

    /**
     * @brief Get full spatial velocity [v; omega]
     */
    SpatialVector spatialVelocity() const {
        SpatialVector v;
        v.head<3>() = state_.linear_velocity;
        v.tail<3>() = state_.angular_velocity;
        return v;
    }

    /**
     * @brief Set full spatial velocity [v; omega]
     */
    void setSpatialVelocity(const SpatialVector& vel) {
        state_.linear_velocity = vel.head<3>();
        state_.angular_velocity = vel.tail<3>();
    }

    /**
     * @brief Get linear momentum
     */
    Eigen::Vector3d linearMomentum() const {
        return properties_.mass * state_.linear_velocity;
    }

    /**
     * @brief Get mass
     */
    double mass() const {
        return properties_.mass;
    }

    bool isStatic() const {
        return is_static_;
    }

    void setStatic(bool is_static) {
        is_static_ = is_static;
        if (is_static_) {
            state_.linear_velocity.setZero();
            state_.angular_velocity.setZero();
            clearForces();
        }
    }

    /**
     * @brief Apply force at point (wrapper for applyForceAtPoint)
     */
    void applyForce(const Eigen::Vector3d& force, const Eigen::Vector3d& world_point) {
        applyForceAtPoint(force, world_point);
    }

    /**
     * @brief Semi-implicit Euler integration (simplified interface)
     */
    void semiImplicitEuler(double dt) {
        integrate(dt, Eigen::Vector3d::Zero());
    }

private:
    RigidBodyState state_;
    RigidBodyProperties properties_;
    bool is_static_ = false;

    Eigen::Vector3d external_force_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d external_torque_ = Eigen::Vector3d::Zero();
};

}  // namespace vde
