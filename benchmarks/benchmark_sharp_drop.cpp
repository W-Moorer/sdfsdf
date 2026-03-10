#include "BenchmarkCommon.h"
#include <Eigen/Geometry>
#include <array>
#include <iostream>
#include <memory>

namespace {

using namespace vde;
using namespace vde::benchmarks;

constexpr double kPi = 3.14159265358979323846;

struct SharpDropSummary {
    double p_norm = 0.0;
    double activation_time = 0.0;
    double max_equivalent_penetration = 0.0;
    double final_equivalent_penetration = 0.0;
};

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

SharpDropSummary runCase(double p_norm,
                         const std::shared_ptr<SDF>& sharp_sdf,
                         double half_extent,
                         const Eigen::Quaterniond& orientation,
                         const std::shared_ptr<SDF>& plane_sdf,
                         const AABB& plane_aabb,
                         std::vector<std::vector<std::string>>& rows) {
    constexpr double start_tip_gap = 0.08;
    constexpr double end_tip_gap = -0.08;
    constexpr int num_samples = 121;
    constexpr double activation_threshold = 1e-4;

    VolumetricIntegrator integrator(64, p_norm);

    SharpDropSummary summary;
    summary.p_norm = p_norm;
    summary.activation_time = 1.0;

    const double corner_offset = lowestVertexOffset(half_extent, orientation);

    for (int sample = 0; sample < num_samples; ++sample) {
        const double alpha = static_cast<double>(sample) / static_cast<double>(num_samples - 1);
        const double time = 0.3 * alpha;
        const double tip_gap = start_tip_gap + alpha * (end_tip_gap - start_tip_gap);
        const double center_y = tip_gap - corner_offset;
        const Eigen::Vector3d center(0.0, center_y, 0.0);

        TransformedSDF sharp_world(sharp_sdf, center, orientation);
        TransformedSDF plane_world(
            plane_sdf,
            Eigen::Vector3d::Zero(),
            Eigen::Quaterniond::Identity());

        const ContactGeometry geometry = integrator.computeContactGeometry(
            sharp_world,
            plane_world,
            computeWorldAABB(half_extent, center, orientation),
            plane_aabb);
        const double penetration = std::max(0.0, -geometry.g_eq);

        if (penetration >= activation_threshold && summary.activation_time >= 1.0) {
            summary.activation_time = time;
        }

        summary.max_equivalent_penetration =
            std::max(summary.max_equivalent_penetration, penetration);
        summary.final_equivalent_penetration = penetration;

        rows.push_back({
            formatDouble(p_norm),
            formatDouble(time),
            formatDouble(tip_gap),
            formatDouble(geometry.g_eq),
            formatDouble(penetration)
        });
    }

    return summary;
}

}  // namespace

int main() {
    constexpr double half_extent = 0.14;
    const Eigen::Quaterniond orientation =
        Eigen::Quaterniond(Eigen::AngleAxisd(kPi / 4.0, Eigen::Vector3d::UnitZ())) *
        Eigen::Quaterniond(Eigen::AngleAxisd(kPi / 4.0, Eigen::Vector3d::UnitX()));

    auto sharp_sdf = std::make_shared<BoxSDF>(
        Eigen::Vector3d::Constant(-half_extent),
        Eigen::Vector3d::Constant(half_extent));
    auto plane_sdf = std::make_shared<HalfSpaceSDF>(
        Eigen::Vector3d(0.0, 1.0, 0.0),
        Eigen::Vector3d::Zero());
    const AABB plane_aabb =
        VolumetricIntegrator::halfSpaceAABB(*plane_sdf, Eigen::Vector3d::Zero(), 3.0);

    std::vector<std::vector<std::string>> rows;
    rows.reserve(3 * 121);

    std::vector<SharpDropSummary> summaries;
    for (double p_norm : {2.0, 4.0, 6.0}) {
        summaries.push_back(runCase(
            p_norm,
            sharp_sdf,
            half_extent,
            orientation,
            plane_sdf,
            plane_aabb,
            rows));
    }

    std::vector<std::vector<std::string>> summary_rows;
    summary_rows.reserve(summaries.size());
    for (const auto& summary : summaries) {
        summary_rows.push_back(vectorToRow({
            summary.p_norm,
            summary.activation_time,
            summary.max_equivalent_penetration,
            summary.final_equivalent_penetration
        }));
    }

    const auto dir = resultsDirectory();
    writeCsv(
        dir / "benchmark_sharp_drop.csv",
        {"p_norm", "time", "tip_gap", "g_eq", "equivalent_penetration"},
        rows);
    writeCsv(
        dir / "benchmark_sharp_drop_summary.csv",
        {"p_norm", "activation_time", "max_equivalent_penetration", "final_equivalent_penetration"},
        summary_rows);

    std::cout << "benchmark_sharp_drop.csv written to " << (dir / "benchmark_sharp_drop.csv") << '\n';
    for (const auto& summary : summaries) {
        std::cout << "p=" << summary.p_norm
                  << " activation_time=" << summary.activation_time
                  << " max_penetration=" << summary.max_equivalent_penetration
                  << " final_penetration=" << summary.final_equivalent_penetration << '\n';
    }
    return 0;
}
