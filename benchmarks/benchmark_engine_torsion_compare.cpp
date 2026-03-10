#include "BenchmarkCommon.h"
#include "engine/SimulationEngine.h"
#include <iostream>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

RigidBodyProperties cylinderProperties(double mass, double radius, double height) {
    RigidBodyProperties props;
    props.mass = mass;
    const double i_axis = 0.5 * mass * radius * radius;
    const double i_radial = (mass / 12.0) * (3.0 * radius * radius + height * height);
    props.inertia_local.setZero();
    props.inertia_local(0, 0) = i_radial;
    props.inertia_local(1, 1) = i_axis;
    props.inertia_local(2, 2) = i_radial;
    return props;
}

struct EngineCase {
    SimulationEngine engine;
    std::shared_ptr<RigidBody> cylinder;

    explicit EngineCase(bool torsional_friction)
        : engine([torsional_friction]() {
              SimulationConfig config;
              config.time_step = 0.01;
              config.enable_friction = true;
              config.enable_torsional_friction = torsional_friction;
              config.friction_coefficient = 0.35;
              config.contact_grid_resolution = 64;
              config.contact_p_norm = 2.0;
              config.baumgarte_gamma = 0.10;
              config.max_solver_iterations = 180;
              config.solver_tolerance = 1e-8;
              config.torsion_regularization = 1e-4;
              return config;
          }()) {
        constexpr double radius = 0.45;
        constexpr double height = 0.24;
        constexpr double mass = 3.0;
        constexpr double preload_penetration = 0.05;

        cylinder = std::make_shared<RigidBody>(cylinderProperties(mass, radius, height));
        cylinder->setPosition(Eigen::Vector3d(0.0, 0.5 * height - preload_penetration, 0.0));
        cylinder->setAngularVelocity(Eigen::Vector3d(0.0, 10.0, 0.0));

        auto ground = std::make_shared<RigidBody>();
        ground->setStatic(true);

        auto cylinder_sdf = std::make_shared<CylinderSDF>(radius, height);
        auto plane_sdf = std::make_shared<HalfSpaceSDF>(
            Eigen::Vector3d(0.0, 1.0, 0.0),
            Eigen::Vector3d::Zero());

        const AABB cylinder_aabb(
            Eigen::Vector3d(-radius, -0.5 * height, -radius),
            Eigen::Vector3d(radius, 0.5 * height, radius));
        const AABB plane_aabb =
            VolumetricIntegrator::halfSpaceAABB(*plane_sdf, Eigen::Vector3d::Zero(), 3.0);

        engine.addBody(cylinder, cylinder_sdf, cylinder_aabb);
        engine.addBody(ground, plane_sdf, plane_aabb);
    }
};

double firstBelowThreshold(const std::vector<double>& time,
                           const std::vector<double>& values,
                           double threshold) {
    for (size_t i = 0; i < time.size(); ++i) {
        if (values[i] <= threshold) {
            return time[i];
        }
    }
    return time.empty() ? 0.0 : time.back();
}

}  // namespace

int main() {
    constexpr double duration = 1.2;
    constexpr double omega_threshold = 0.5;

    EngineCase case_4d(true);
    EngineCase case_3d(false);

    const int num_steps = static_cast<int>(duration / case_4d.engine.config().time_step);

    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_steps);

    std::vector<double> time_history;
    std::vector<double> omega_4d_history;
    std::vector<double> omega_3d_history;

    double peak_residual_4d = 0.0;
    double peak_residual_3d = 0.0;
    double peak_penetration_4d = 0.0;
    double peak_penetration_3d = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        const SimulationStats stats_4d = case_4d.engine.step();
        const SimulationStats stats_3d = case_3d.engine.step();

        const double time = case_4d.engine.currentTime();
        const double omega_4d = std::abs(case_4d.cylinder->angularVelocity().y());
        const double omega_3d = std::abs(case_3d.cylinder->angularVelocity().y());
        const double penetration_4d =
            maxPenetrationFromContacts(case_4d.engine.getContacts());
        const double penetration_3d =
            maxPenetrationFromContacts(case_3d.engine.getContacts());

        time_history.push_back(time);
        omega_4d_history.push_back(omega_4d);
        omega_3d_history.push_back(omega_3d);

        peak_residual_4d = std::max(peak_residual_4d, stats_4d.solver_residual);
        peak_residual_3d = std::max(peak_residual_3d, stats_3d.solver_residual);
        peak_penetration_4d = std::max(peak_penetration_4d, penetration_4d);
        peak_penetration_3d = std::max(peak_penetration_3d, penetration_3d);

        rows.push_back(vectorToRow({
            time,
            omega_4d,
            omega_3d,
            stats_4d.solver_residual,
            stats_3d.solver_residual,
            penetration_4d,
            penetration_3d
        }));
    }

    const double stop_time_4d =
        firstBelowThreshold(time_history, omega_4d_history, omega_threshold);
    const double stop_time_3d =
        firstBelowThreshold(time_history, omega_3d_history, omega_threshold);
    const double final_omega_4d =
        omega_4d_history.empty() ? 0.0 : omega_4d_history.back();
    const double final_omega_3d =
        omega_3d_history.empty() ? 0.0 : omega_3d_history.back();
    const double spin_decay_ratio =
        (final_omega_3d - final_omega_4d) / std::max(final_omega_3d, 1e-12);

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_engine_torsion_compare.csv",
        {
            "time",
            "omega_4d",
            "omega_3d",
            "residual_4d",
            "residual_3d",
            "penetration_4d",
            "penetration_3d"
        },
        rows);
    writeCsv(
        dir / "benchmark_engine_torsion_compare_summary.csv",
        {
            "stop_time_4d",
            "stop_time_3d",
            "final_omega_4d",
            "final_omega_3d",
            "spin_decay_ratio",
            "peak_residual_4d",
            "peak_residual_3d",
            "peak_penetration_4d",
            "peak_penetration_3d"
        },
        {vectorToRow({
            stop_time_4d,
            stop_time_3d,
            final_omega_4d,
            final_omega_3d,
            spin_decay_ratio,
            peak_residual_4d,
            peak_residual_3d,
            peak_penetration_4d,
            peak_penetration_3d
        })});

    std::cout << "benchmark_engine_torsion_compare.csv written to "
              << (dir / "benchmark_engine_torsion_compare.csv") << '\n';
    std::cout << "final_omega_4d=" << final_omega_4d
              << " final_omega_3d=" << final_omega_3d
              << " spin_decay_ratio=" << spin_decay_ratio << '\n';
    return 0;
}
