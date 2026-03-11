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
    Eigen::Vector3d initial_linear_velocity = Eigen::Vector3d(0.30, 0.0, -0.14);
    Eigen::Vector3d initial_angular_velocity = Eigen::Vector3d(0.0, 8.0, 0.0);
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
    config.enable_torsional_friction = true;
    config.friction_coefficient = scene.friction_coefficient;
    config.contact_grid_resolution = 40;
    config.contact_p_norm = 2.0;
    config.baumgarte_gamma = 0.0;
    config.max_solver_iterations = 180;
    config.solver_tolerance = 1e-8;

    SimulationEngine soccp_engine(config);
    SimulationConfig pgs_config = config;
    pgs_config.solver_mode = ContactSolverMode::FourDPGS;
    SimulationEngine four_d_pgs_engine(pgs_config);

    auto box_sdf = std::make_shared<BoxSDF>(-box_half, box_half);
    auto ground_sdf = std::make_shared<BoxSDF>(-ground_half, ground_half);
    const AABB box_aabb(-box_half, box_half);
    const AABB ground_aabb(-ground_half, ground_half);

    const int box_id_soccp =
        populateEngineScene(soccp_engine, box_sdf, box_aabb, ground_sdf, ground_aabb, scene);
    const int box_id_pgs =
        populateEngineScene(four_d_pgs_engine, box_sdf, box_aabb, ground_sdf, ground_aabb, scene);

    const Eigen::Vector2d initial_xz(scene.initial_position.x(), scene.initial_position.z());
    const int num_steps = static_cast<int>(scene.duration / scene.dt);
    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);

    double max_spin_diff = 0.0;
    double max_slip_diff = 0.0;
    double max_tilt_diff = 0.0;
    double max_penetration_diff = 0.0;
    double peak_penetration_soccp = 0.0;
    double peak_penetration_pgs = 0.0;
    double peak_raw_residual_soccp = 0.0;
    double peak_raw_residual_pgs = 0.0;
    double peak_scaled_residual_soccp = 0.0;
    double peak_scaled_residual_pgs = 0.0;
    double peak_complementarity_soccp = 0.0;
    double peak_complementarity_pgs = 0.0;
    double max_iterations_soccp = 0.0;
    double max_iterations_pgs = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        const SimulationStats soccp_stats = soccp_engine.step();
        const SimulationStats pgs_stats = four_d_pgs_engine.step();

        const auto body_soccp = soccp_engine.getBody(box_id_soccp);
        const auto body_pgs = four_d_pgs_engine.getBody(box_id_pgs);

        const double penetration_soccp =
            maxPenetrationFromContacts(soccp_engine.getContacts());
        const double penetration_pgs =
            maxPenetrationFromContacts(four_d_pgs_engine.getContacts());
        const double spin_soccp = spinRateY(*body_soccp);
        const double spin_pgs = spinRateY(*body_pgs);
        const double slip_soccp = horizontalSlip(*body_soccp, initial_xz);
        const double slip_pgs = horizontalSlip(*body_pgs, initial_xz);
        const double tilt_soccp = tiltAngleDegrees(*body_soccp);
        const double tilt_pgs = tiltAngleDegrees(*body_pgs);

        max_spin_diff = std::max(max_spin_diff, std::abs(spin_soccp - spin_pgs));
        max_slip_diff = std::max(max_slip_diff, std::abs(slip_soccp - slip_pgs));
        max_tilt_diff = std::max(max_tilt_diff, std::abs(tilt_soccp - tilt_pgs));
        max_penetration_diff = std::max(
            max_penetration_diff,
            std::abs(penetration_soccp - penetration_pgs));
        peak_penetration_soccp = std::max(peak_penetration_soccp, penetration_soccp);
        peak_penetration_pgs = std::max(peak_penetration_pgs, penetration_pgs);
        peak_raw_residual_soccp = std::max(peak_raw_residual_soccp, soccp_stats.solver_residual);
        peak_raw_residual_pgs = std::max(peak_raw_residual_pgs, pgs_stats.solver_residual);
        peak_scaled_residual_soccp = std::max(
            peak_scaled_residual_soccp,
            soccp_stats.solver_scaled_residual);
        peak_scaled_residual_pgs = std::max(
            peak_scaled_residual_pgs,
            pgs_stats.solver_scaled_residual);
        peak_complementarity_soccp = std::max(
            peak_complementarity_soccp,
            soccp_stats.solver_complementarity_violation);
        peak_complementarity_pgs = std::max(
            peak_complementarity_pgs,
            pgs_stats.solver_complementarity_violation);
        max_iterations_soccp = std::max(
            max_iterations_soccp,
            static_cast<double>(soccp_stats.solver_iterations));
        max_iterations_pgs = std::max(
            max_iterations_pgs,
            static_cast<double>(pgs_stats.solver_iterations));

        rows.push_back(vectorToRow({
            soccp_engine.currentTime(),
            penetration_soccp,
            penetration_pgs,
            spin_soccp,
            spin_pgs,
            slip_soccp,
            slip_pgs,
            tilt_soccp,
            tilt_pgs,
            soccp_stats.solver_residual,
            pgs_stats.solver_residual,
            soccp_stats.solver_scaled_residual,
            pgs_stats.solver_scaled_residual,
            soccp_stats.solver_complementarity_violation,
            pgs_stats.solver_complementarity_violation,
            static_cast<double>(soccp_stats.solver_iterations),
            static_cast<double>(pgs_stats.solver_iterations),
            static_cast<double>(soccp_engine.getContacts().size()),
            static_cast<double>(four_d_pgs_engine.getContacts().size())
        }));
    }

    const auto final_soccp = soccp_engine.getBody(box_id_soccp);
    const auto final_pgs = four_d_pgs_engine.getBody(box_id_pgs);
    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_oblique_box_4d_reference.csv",
        {
            "time",
            "penetration_soccp",
            "penetration_4dpgs",
            "spin_soccp",
            "spin_4dpgs",
            "slip_soccp",
            "slip_4dpgs",
            "tilt_soccp",
            "tilt_4dpgs",
            "raw_residual_soccp",
            "raw_residual_4dpgs",
            "scaled_residual_soccp",
            "scaled_residual_4dpgs",
            "complementarity_soccp",
            "complementarity_4dpgs",
            "iterations_soccp",
            "iterations_4dpgs",
            "contacts_soccp",
            "contacts_4dpgs"
        },
        rows);
    writeCsv(
        dir / "benchmark_engine_oblique_box_4d_reference_summary.csv",
        {
            "max_spin_diff",
            "max_slip_diff",
            "max_tilt_diff",
            "max_penetration_diff",
            "peak_penetration_soccp",
            "peak_penetration_4dpgs",
            "peak_raw_residual_soccp",
            "peak_raw_residual_4dpgs",
            "peak_scaled_residual_soccp",
            "peak_scaled_residual_4dpgs",
            "peak_complementarity_soccp",
            "peak_complementarity_4dpgs",
            "max_iterations_soccp",
            "max_iterations_4dpgs",
            "final_spin_soccp",
            "final_spin_4dpgs",
            "final_slip_soccp",
            "final_slip_4dpgs",
            "final_tilt_soccp",
            "final_tilt_4dpgs"
        },
        {vectorToRow({
            max_spin_diff,
            max_slip_diff,
            max_tilt_diff,
            max_penetration_diff,
            peak_penetration_soccp,
            peak_penetration_pgs,
            peak_raw_residual_soccp,
            peak_raw_residual_pgs,
            peak_scaled_residual_soccp,
            peak_scaled_residual_pgs,
            peak_complementarity_soccp,
            peak_complementarity_pgs,
            max_iterations_soccp,
            max_iterations_pgs,
            spinRateY(*final_soccp),
            spinRateY(*final_pgs),
            horizontalSlip(*final_soccp, initial_xz),
            horizontalSlip(*final_pgs, initial_xz),
            tiltAngleDegrees(*final_soccp),
            tiltAngleDegrees(*final_pgs)
        })});

    std::cout << "benchmark_engine_oblique_box_4d_reference.csv written to "
              << (dir / "benchmark_engine_oblique_box_4d_reference.csv") << '\n';
    std::cout << "max_spin_diff=" << max_spin_diff
              << " max_slip_diff=" << max_slip_diff
              << " max_penetration_diff=" << max_penetration_diff << '\n';
    return 0;
}
