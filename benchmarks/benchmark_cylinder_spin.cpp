#include "BenchmarkCommon.h"
#include <cmath>
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

}  // namespace

int main() {
    constexpr double radius = 0.45;
    constexpr double height = 0.24;
    constexpr double mass = 3.0;
    constexpr double friction = 0.35;
    constexpr double omega0 = 10.0;
    constexpr double preload_penetration = 0.05;
    constexpr double duration = 1.2;
    constexpr int num_samples = 121;

    const Eigen::Vector3d cylinder_center(0.0, 0.5 * height - preload_penetration, 0.0);
    auto cylinder_sdf = std::make_shared<CylinderSDF>(radius, height);
    auto plane_sdf = std::make_shared<HalfSpaceSDF>(
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d::Zero());

    const AABB cylinder_aabb(
        Eigen::Vector3d(-radius, cylinder_center.y() - 0.5 * height, -radius),
        Eigen::Vector3d(radius, cylinder_center.y() + 0.5 * height, radius));
    const AABB plane_aabb =
        VolumetricIntegrator::halfSpaceAABB(*plane_sdf, Eigen::Vector3d::Zero(), 3.0);

    TransformedSDF world_cylinder(
        cylinder_sdf,
        cylinder_center,
        Eigen::Quaterniond::Identity());
    TransformedSDF world_plane(
        plane_sdf,
        Eigen::Vector3d::Zero(),
        Eigen::Quaterniond::Identity());

    VolumetricIntegrator integrator(64, 2.0);
    const ContactGeometry geometry =
        integrator.computeContactGeometry(world_cylinder, world_plane, cylinder_aabb, plane_aabb);

    const double r_eff_model = geometry.r_tau;
    const double r_eff_theory = 2.0 * radius / 3.0;
    const double i_axis = cylinderProperties(mass, radius, height).inertia_local(1, 1);

    const double alpha_model = friction * mass * 9.81 * r_eff_model / i_axis;
    const double alpha_theory = friction * mass * 9.81 * r_eff_theory / i_axis;
    const double t_stop_model = omega0 / std::max(alpha_model, 1e-12);
    const double t_stop_theory = omega0 / std::max(alpha_theory, 1e-12);
    const double relative_radius_error =
        std::abs(r_eff_model - r_eff_theory) / std::max(r_eff_theory, 1e-12);
    const double relative_stop_error =
        std::abs(t_stop_model - t_stop_theory) / std::max(t_stop_theory, 1e-12);

    std::vector<std::vector<std::string>> rows;
    rows.reserve(num_samples);
    for (int sample = 0; sample < num_samples; ++sample) {
        const double alpha = static_cast<double>(sample) / static_cast<double>(num_samples - 1);
        const double time = duration * alpha;
        const double omega_model = std::max(0.0, omega0 - alpha_model * time);
        const double omega_theory = std::max(0.0, omega0 - alpha_theory * time);
        const double omega_point = omega0;

        rows.push_back(vectorToRow({
            time,
            omega_model,
            omega_theory,
            omega_point
        }));
    }

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_cylinder_spin.csv",
        {"time", "omega_model", "omega_theory", "omega_point"},
        rows);
    writeCsv(
        dir / "benchmark_cylinder_spin_summary.csv",
        {"r_eff_model", "r_eff_theory", "relative_radius_error", "t_stop_model", "t_stop_theory", "relative_stop_error", "final_omega_point"},
        {vectorToRow({
            r_eff_model,
            r_eff_theory,
            relative_radius_error,
            t_stop_model,
            t_stop_theory,
            relative_stop_error,
            omega0
        })});

    std::cout << "benchmark_cylinder_spin.csv written to " << (dir / "benchmark_cylinder_spin.csv") << '\n';
    std::cout << "r_eff_model=" << r_eff_model
              << " r_eff_theory=" << r_eff_theory
              << " relative_radius_error=" << relative_radius_error << '\n';
    std::cout << "t_stop_model=" << t_stop_model
              << " t_stop_theory=" << t_stop_theory
              << " relative_stop_error=" << relative_stop_error << '\n';
    return 0;
}
