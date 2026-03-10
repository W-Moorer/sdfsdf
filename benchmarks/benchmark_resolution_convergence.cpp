#include "BenchmarkCommon.h"
#include <Eigen/Geometry>
#include <array>
#include <iostream>
#include <limits>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

constexpr double kPi = 3.14159265358979323846;

std::array<Eigen::Vector3d, 8> localCorners(double half_extent) {
    return {{
        {-half_extent, -half_extent, -half_extent},
        {-half_extent, -half_extent, half_extent},
        {-half_extent, half_extent, -half_extent},
        {-half_extent, half_extent, half_extent},
        {half_extent, -half_extent, -half_extent},
        {half_extent, -half_extent, half_extent},
        {half_extent, half_extent, -half_extent},
        {half_extent, half_extent, half_extent},
    }};
}

AABB computeWorldAABB(double half_extent,
                      const Eigen::Vector3d& center,
                      const Eigen::Quaterniond& orientation) {
    AABB aabb;
    for (const auto& corner : localCorners(half_extent)) {
        aabb.expand(center + orientation * corner);
    }
    return aabb;
}

double lowestVertexOffset(double half_extent, const Eigen::Quaterniond& orientation) {
    double min_y = std::numeric_limits<double>::max();
    for (const auto& corner : localCorners(half_extent)) {
        min_y = std::min(min_y, (orientation * corner).y());
    }
    return min_y;
}

ContactGeometry computeSharpGeometry(int resolution,
                                     double p_norm,
                                     double half_extent,
                                     const Eigen::Quaterniond& orientation,
                                     double tip_gap) {
    auto sharp_sdf = std::make_shared<BoxSDF>(
        Eigen::Vector3d::Constant(-half_extent),
        Eigen::Vector3d::Constant(half_extent));
    auto plane_sdf = std::make_shared<HalfSpaceSDF>(
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d::Zero());

    const double corner_offset = lowestVertexOffset(half_extent, orientation);
    const Eigen::Vector3d center(0.0, tip_gap - corner_offset, 0.0);

    TransformedSDF sharp_world(sharp_sdf, center, orientation);
    TransformedSDF plane_world(
        plane_sdf,
        Eigen::Vector3d::Zero(),
        Eigen::Quaterniond::Identity());

    VolumetricIntegrator integrator(resolution, p_norm);
    return integrator.computeContactGeometry(
        sharp_world,
        plane_world,
        computeWorldAABB(half_extent, center, orientation),
        VolumetricIntegrator::halfSpaceAABB(*plane_sdf, Eigen::Vector3d::Zero(), 3.0));
}

}  // namespace

int main() {
    constexpr double half_extent = 0.14;
    constexpr double tip_gap = -0.08;
    const Eigen::Quaterniond orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(kPi / 4.0, Eigen::Vector3d::UnitZ())) *
        Eigen::Quaterniond(Eigen::AngleAxisd(kPi / 4.0, Eigen::Vector3d::UnitX()));

    const std::vector<int> resolutions = {16, 24, 32, 48, 64, 96, 128};
    const std::vector<double> p_values = {2.0, 6.0};

    std::vector<std::vector<std::string>> rows;
    rows.reserve(resolutions.size() * p_values.size());

    double error_p2_at_64 = 0.0;
    double error_p6_at_64 = 0.0;

    for (double p_norm : p_values) {
        const ContactGeometry reference = computeSharpGeometry(
            resolutions.back(),
            p_norm,
            half_extent,
            orientation,
            tip_gap);

        for (int resolution : resolutions) {
            const ContactGeometry geometry = computeSharpGeometry(
                resolution,
                p_norm,
                half_extent,
                orientation,
                tip_gap);

            const double gap_error =
                std::abs(geometry.g_eq - reference.g_eq) / std::max(std::abs(reference.g_eq), 1e-12);
            const double rtau_error =
                std::abs(geometry.r_tau - reference.r_tau) / std::max(std::abs(reference.r_tau), 1e-12);

            if (resolution == 64 && p_norm == 2.0) {
                error_p2_at_64 = gap_error;
            }
            if (resolution == 64 && p_norm == 6.0) {
                error_p6_at_64 = gap_error;
            }

            rows.push_back(vectorToRow({
                static_cast<double>(resolution),
                p_norm,
                geometry.g_eq,
                geometry.r_tau,
                gap_error,
                rtau_error
            }));
        }
    }

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_resolution_convergence.csv",
        {"resolution", "p_norm", "g_eq", "r_tau", "relative_gap_error", "relative_rtau_error"},
        rows);
    writeCsv(
        dir / "benchmark_resolution_convergence_summary.csv",
        {"gap_error_p2_at_64", "gap_error_p6_at_64", "reference_resolution"},
        {vectorToRow({
            error_p2_at_64,
            error_p6_at_64,
            static_cast<double>(resolutions.back())
        })});

    std::cout << "benchmark_resolution_convergence.csv written to "
              << (dir / "benchmark_resolution_convergence.csv") << '\n';
    std::cout << "gap_error_p2_at_64=" << error_p2_at_64
              << " gap_error_p6_at_64=" << error_p6_at_64 << '\n';
    return 0;
}
