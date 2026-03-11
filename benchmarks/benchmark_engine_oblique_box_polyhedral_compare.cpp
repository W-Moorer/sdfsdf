#include "BenchmarkCommon.h"
#include "engine/SimulationEngine.h"
#include <array>
#include <cmath>
#include <iostream>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

constexpr double kPi = 3.14159265358979323846;

struct SceneConfig {
    double dt = 0.01;
    double duration = 2.0;
    double box_width = 0.80;
    double box_height = 0.18;
    double box_depth = 0.62;
    double ground_width = 8.0;
    double ground_height = 1.5;
    double ground_depth = 8.0;
    double friction_coefficient = 0.42;
    double preload_penetration = 0.035;
    Eigen::Vector3d initial_position = Eigen::Vector3d(0.0, 0.5 * box_height - preload_penetration, 0.0);
    Eigen::Vector3d initial_linear_velocity = Eigen::Vector3d(0.24, 0.0, -0.24);
    Eigen::Vector3d initial_angular_velocity = Eigen::Vector3d::Zero();
};

RigidBodyProperties makeBoxProps(double mass,
                                 double width,
                                 double height,
                                 double depth) {
    return RigidBodyProperties::box(mass, width, height, depth);
}

double tiltAngleDegrees(const RigidBody& body) {
    const Eigen::Vector3d world_up =
        body.state().orientation * Eigen::Vector3d::UnitY();
    const double cosine =
        std::clamp(world_up.dot(Eigen::Vector3d::UnitY()), -1.0, 1.0);
    return std::acos(cosine) * 180.0 / kPi;
}

double spinRateY(const RigidBody& body) {
    return std::abs(body.angularVelocity().y());
}

double horizontalSlip(const RigidBody& body, const Eigen::Vector2d& initial_xz) {
    const Eigen::Vector2d current_xz(body.position().x(), body.position().z());
    return (current_xz - initial_xz).norm();
}

void initializeDynamicBody(RigidBody& body, const SceneConfig& scene) {
    body.setPosition(scene.initial_position);
    body.setLinearVelocity(scene.initial_linear_velocity);
    body.setAngularVelocity(scene.initial_angular_velocity);
    body.state().orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(17.0 * kPi / 180.0, Eigen::Vector3d::UnitY()));
}

int populateEngineScene(SimulationEngine& engine,
                        const std::shared_ptr<SDF>& box_sdf,
                        const AABB& box_aabb,
                        const std::shared_ptr<SDF>& ground_sdf,
                        const AABB& ground_aabb,
                        const SceneConfig& scene) {
    auto box = std::make_shared<RigidBody>(
        makeBoxProps(2.4, scene.box_width, scene.box_height, scene.box_depth));
    initializeDynamicBody(*box, scene);
    auto ground = std::make_shared<RigidBody>(
        makeBoxProps(10.0, scene.ground_width, scene.ground_height, scene.ground_depth));
    ground->setStatic(true);
    ground->setPosition(Eigen::Vector3d(0.0, -0.5 * scene.ground_height, 0.0));

    const int box_id = engine.addBody(box, box_sdf, box_aabb);
    engine.addBody(ground, ground_sdf, ground_aabb);
    return box_id;
}

}  // namespace

int main() {
    const SceneConfig scene;
    const Eigen::Vector3d box_half(
        0.5 * scene.box_width,
        0.5 * scene.box_height,
        0.5 * scene.box_depth);
    const Eigen::Vector3d ground_half(
        0.5 * scene.ground_width,
        0.5 * scene.ground_height,
        0.5 * scene.ground_depth);

    SimulationConfig config;
    config.time_step = scene.dt;
    config.enable_friction = true;
    config.enable_torsional_friction = false;
    config.friction_coefficient = scene.friction_coefficient;
    config.contact_grid_resolution = 40;
    config.contact_p_norm = 2.0;
    config.baumgarte_gamma = 0.0;
    config.max_solver_iterations = 180;
    config.solver_tolerance = 1e-8;
    config.polyhedral_friction_directions = 16;

    auto box_sdf = std::make_shared<BoxSDF>(-box_half, box_half);
    auto ground_sdf = std::make_shared<BoxSDF>(-ground_half, ground_half);
    const AABB box_aabb(-box_half, box_half);
    const AABB ground_aabb(-ground_half, ground_half);

    SimulationConfig soccp_config = config;
    soccp_config.solver_mode = ContactSolverMode::SOCCP;
    SimulationEngine soccp_engine(soccp_config);

    SimulationConfig poly_config = config;
    poly_config.solver_mode = ContactSolverMode::PolyhedralFriction;
    SimulationEngine poly_engine(poly_config);

    const int box_id_soccp =
        populateEngineScene(soccp_engine, box_sdf, box_aabb, ground_sdf, ground_aabb, scene);
    const int box_id_poly =
        populateEngineScene(poly_engine, box_sdf, box_aabb, ground_sdf, ground_aabb, scene);

    const int num_steps = static_cast<int>(scene.duration / scene.dt);
    const Eigen::Vector2d initial_xz(scene.initial_position.x(), scene.initial_position.z());
    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);

    double peak_penetration_soccp = 0.0;
    double peak_penetration_poly = 0.0;
    double peak_residual_soccp = 0.0;
    double peak_residual_poly = 0.0;
    double peak_scaled_soccp = 0.0;
    double peak_scaled_poly = 0.0;
    double peak_comp_soccp = 0.0;
    double peak_comp_poly = 0.0;
    double max_iterations_soccp = 0.0;
    double max_iterations_poly = 0.0;
    double max_slip_diff = 0.0;
    double max_spin_diff = 0.0;
    double max_tilt_diff = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        const SimulationStats stats_soccp = soccp_engine.step();
        const SimulationStats stats_poly = poly_engine.step();

        const auto body_soccp = soccp_engine.getBody(box_id_soccp);
        const auto body_poly = poly_engine.getBody(box_id_poly);

        const double penetration_soccp = maxPenetrationFromContacts(soccp_engine.getContacts());
        const double penetration_poly = maxPenetrationFromContacts(poly_engine.getContacts());
        const double slip_soccp = horizontalSlip(*body_soccp, initial_xz);
        const double slip_poly = horizontalSlip(*body_poly, initial_xz);
        const double spin_soccp = spinRateY(*body_soccp);
        const double spin_poly = spinRateY(*body_poly);
        const double tilt_soccp = tiltAngleDegrees(*body_soccp);
        const double tilt_poly = tiltAngleDegrees(*body_poly);

        peak_penetration_soccp = std::max(peak_penetration_soccp, penetration_soccp);
        peak_penetration_poly = std::max(peak_penetration_poly, penetration_poly);
        peak_residual_soccp = std::max(peak_residual_soccp, stats_soccp.solver_residual);
        peak_residual_poly = std::max(peak_residual_poly, stats_poly.solver_residual);
        peak_scaled_soccp = std::max(peak_scaled_soccp, stats_soccp.solver_scaled_residual);
        peak_scaled_poly = std::max(peak_scaled_poly, stats_poly.solver_scaled_residual);
        peak_comp_soccp = std::max(peak_comp_soccp, stats_soccp.solver_complementarity_violation);
        peak_comp_poly = std::max(peak_comp_poly, stats_poly.solver_complementarity_violation);
        max_iterations_soccp = std::max(max_iterations_soccp, static_cast<double>(stats_soccp.solver_iterations));
        max_iterations_poly = std::max(max_iterations_poly, static_cast<double>(stats_poly.solver_iterations));
        max_slip_diff = std::max(max_slip_diff, std::abs(slip_soccp - slip_poly));
        max_spin_diff = std::max(max_spin_diff, std::abs(spin_soccp - spin_poly));
        max_tilt_diff = std::max(max_tilt_diff, std::abs(tilt_soccp - tilt_poly));

        rows.push_back(vectorToRow({
            soccp_engine.currentTime(),
            penetration_soccp,
            penetration_poly,
            stats_soccp.solver_residual,
            stats_poly.solver_residual,
            stats_soccp.solver_scaled_residual,
            stats_poly.solver_scaled_residual,
            stats_soccp.solver_complementarity_violation,
            stats_poly.solver_complementarity_violation,
            static_cast<double>(stats_soccp.solver_iterations),
            static_cast<double>(stats_poly.solver_iterations),
            spin_soccp,
            spin_poly,
            slip_soccp,
            slip_poly,
            tilt_soccp,
            tilt_poly,
            static_cast<double>(soccp_engine.getContacts().size()),
            static_cast<double>(poly_engine.getContacts().size())
        }));
    }

    const auto body_soccp_final = soccp_engine.getBody(box_id_soccp);
    const auto body_poly_final = poly_engine.getBody(box_id_poly);
    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_oblique_box_polyhedral_compare.csv",
        {
            "time",
            "penetration_soccp",
            "penetration_polyhedral",
            "residual_soccp",
            "residual_polyhedral",
            "scaled_residual_soccp",
            "scaled_residual_polyhedral",
            "complementarity_soccp",
            "complementarity_polyhedral",
            "iterations_soccp",
            "iterations_polyhedral",
            "spin_soccp",
            "spin_polyhedral",
            "slip_soccp",
            "slip_polyhedral",
            "tilt_soccp",
            "tilt_polyhedral",
            "contacts_soccp",
            "contacts_polyhedral"
        },
        rows);

    writeCsv(
        dir / "benchmark_engine_oblique_box_polyhedral_compare_summary.csv",
        {
            "peak_penetration_soccp",
            "peak_penetration_polyhedral",
            "peak_residual_soccp",
            "peak_residual_polyhedral",
            "peak_scaled_soccp",
            "peak_scaled_polyhedral",
            "peak_complementarity_soccp",
            "peak_complementarity_polyhedral",
            "max_iterations_soccp",
            "max_iterations_polyhedral",
            "final_spin_soccp",
            "final_spin_polyhedral",
            "final_slip_soccp",
            "final_slip_polyhedral",
            "final_tilt_soccp",
            "final_tilt_polyhedral",
            "max_slip_diff",
            "max_spin_diff",
            "max_tilt_diff"
        },
        {vectorToRow({
            peak_penetration_soccp,
            peak_penetration_poly,
            peak_residual_soccp,
            peak_residual_poly,
            peak_scaled_soccp,
            peak_scaled_poly,
            peak_comp_soccp,
            peak_comp_poly,
            max_iterations_soccp,
            max_iterations_poly,
            spinRateY(*body_soccp_final),
            spinRateY(*body_poly_final),
            horizontalSlip(*body_soccp_final, initial_xz),
            horizontalSlip(*body_poly_final, initial_xz),
            tiltAngleDegrees(*body_soccp_final),
            tiltAngleDegrees(*body_poly_final),
            max_slip_diff,
            max_spin_diff,
            max_tilt_diff
        })});

    std::cout << "benchmark_engine_oblique_box_polyhedral_compare.csv written to "
              << (dir / "benchmark_engine_oblique_box_polyhedral_compare.csv") << '\n';
    std::cout << "max_slip_diff=" << max_slip_diff
              << " max_spin_diff=" << max_spin_diff
              << " max_tilt_diff=" << max_tilt_diff << '\n';
    return 0;
}
