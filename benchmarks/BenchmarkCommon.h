#pragma once

#include "engine/SimulationEngine.h"
#include <Eigen/Core>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vde::benchmarks {

inline std::filesystem::path resultsDirectory() {
#ifdef VDE_RESULTS_DIR
    const std::filesystem::path dir(VDE_RESULTS_DIR);
#else
    const std::filesystem::path dir("results");
#endif
    std::filesystem::create_directories(dir);
    return dir;
}

inline std::string formatDouble(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(10) << value;
    return stream.str();
}

inline void writeCsv(const std::filesystem::path& path,
                     const std::vector<std::string>& header,
                     const std::vector<std::vector<std::string>>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open CSV file: " + path.string());
    }

    for (size_t i = 0; i < header.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << header[i];
    }
    out << '\n';

    for (const auto& row : rows) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) {
                out << ',';
            }
            out << row[i];
        }
        out << '\n';
    }
}

inline double maxPenetrationFromContacts(const std::vector<vde::ContactPair>& contacts) {
    double max_penetration = 0.0;
    for (const auto& contact : contacts) {
        max_penetration = std::max(max_penetration, -contact.geometry.g_eq);
    }
    return max_penetration;
}

inline double minEquivalentGap(const std::vector<vde::ContactPair>& contacts) {
    double min_gap = 0.0;
    bool found = false;
    for (const auto& contact : contacts) {
        if (!found) {
            min_gap = contact.geometry.g_eq;
            found = true;
        } else {
            min_gap = std::min(min_gap, contact.geometry.g_eq);
        }
    }
    return found ? min_gap : 0.0;
}

inline std::vector<std::string> vectorToRow(const std::vector<double>& values) {
    std::vector<std::string> row;
    row.reserve(values.size());
    for (double value : values) {
        row.push_back(formatDouble(value));
    }
    return row;
}

}  // namespace vde::benchmarks
