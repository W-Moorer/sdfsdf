#pragma once

#include "AnalyticalSDF.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vde {

struct TriangleMesh {
    std::vector<Eigen::Vector3d> vertices;
    std::vector<Eigen::Vector3i> faces;

    bool isValid() const {
        if (vertices.empty() || faces.empty()) {
            return false;
        }
        for (const auto& face : faces) {
            for (int i = 0; i < 3; ++i) {
                if (face(i) < 0 || face(i) >= static_cast<int>(vertices.size())) {
                    return false;
                }
            }
        }
        return true;
    }

    static TriangleMesh makeBox(const Eigen::Vector3d& min_corner,
                                const Eigen::Vector3d& max_corner) {
        if ((max_corner - min_corner).minCoeff() <= 0.0) {
            throw std::invalid_argument("Box dimensions must be positive");
        }

        TriangleMesh mesh;
        mesh.vertices = {
            {min_corner.x(), min_corner.y(), min_corner.z()},
            {max_corner.x(), min_corner.y(), min_corner.z()},
            {max_corner.x(), min_corner.y(), max_corner.z()},
            {min_corner.x(), min_corner.y(), max_corner.z()},
            {min_corner.x(), max_corner.y(), min_corner.z()},
            {max_corner.x(), max_corner.y(), min_corner.z()},
            {max_corner.x(), max_corner.y(), max_corner.z()},
            {min_corner.x(), max_corner.y(), max_corner.z()},
        };

        mesh.faces = {
            {0, 1, 2}, {0, 2, 3},
            {4, 6, 5}, {4, 7, 6},
            {0, 4, 5}, {0, 5, 1},
            {1, 5, 6}, {1, 6, 2},
            {3, 2, 6}, {3, 6, 7},
            {0, 3, 7}, {0, 7, 4},
        };
        return mesh;
    }
};

class MeshSDF : public SDF {
public:
    explicit MeshSDF(TriangleMesh mesh)
        : mesh_(std::move(mesh)) {
        if (!mesh_.isValid()) {
            throw std::invalid_argument("MeshSDF requires a valid closed triangle mesh");
        }
        computeBounds();
    }

    double phi(const Eigen::Vector3d& x) const override {
        const QueryResult query = querySurface(x);
        const double unsigned_distance = query.distance;
        return isInside(x) ? -unsigned_distance : unsigned_distance;
    }

    Eigen::Vector3d gradient(const Eigen::Vector3d& x) const override {
        double out_phi = 0.0;
        Eigen::Vector3d out_grad = Eigen::Vector3d::UnitY();
        phiAndGradient(x, out_phi, out_grad);
        return out_grad;
    }

    void phiAndGradient(const Eigen::Vector3d& x,
                        double& out_phi,
                        Eigen::Vector3d& out_grad) const override {
        const QueryResult query = querySurface(x);
        const bool inside = isInside(x);
        out_phi = inside ? -query.distance : query.distance;
        if (query.distance < 1e-12) {
            out_grad = query.normal;
            return;
        }

        const Eigen::Vector3d diff = x - query.closest_point;
        const Eigen::Vector3d unsigned_grad = diff / query.distance;
        out_grad = inside ? -unsigned_grad : unsigned_grad;
    }

    const TriangleMesh& mesh() const { return mesh_; }
    const Eigen::Vector3d& minCorner() const { return min_corner_; }
    const Eigen::Vector3d& maxCorner() const { return max_corner_; }

private:
    TriangleMesh mesh_;
    Eigen::Vector3d min_corner_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d max_corner_ = Eigen::Vector3d::Zero();

    void computeBounds() {
        min_corner_ = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
        max_corner_ = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
        for (const auto& vertex : mesh_.vertices) {
            min_corner_ = min_corner_.cwiseMin(vertex);
            max_corner_ = max_corner_.cwiseMax(vertex);
        }
    }

    struct QueryResult {
        double distance = std::numeric_limits<double>::infinity();
        Eigen::Vector3d closest_point = Eigen::Vector3d::Zero();
        Eigen::Vector3d normal = Eigen::Vector3d::UnitY();
    };

    QueryResult querySurface(const Eigen::Vector3d& x) const {
        QueryResult best;
        for (const auto& face : mesh_.faces) {
            const Eigen::Vector3d& a = mesh_.vertices[face(0)];
            const Eigen::Vector3d& b = mesh_.vertices[face(1)];
            const Eigen::Vector3d& c = mesh_.vertices[face(2)];
            const Eigen::Vector3d closest = closestPointOnTriangle(x, a, b, c);
            const double distance = (x - closest).norm();
            if (distance < best.distance) {
                best.distance = distance;
                best.closest_point = closest;
                const Eigen::Vector3d face_normal = (b - a).cross(c - a);
                if (face_normal.norm() > 1e-12) {
                    best.normal = face_normal.normalized();
                }
            }
        }
        return best;
    }

    bool isInside(const Eigen::Vector3d& x) const {
        if ((x.array() < min_corner_.array() - 1e-10).any() ||
            (x.array() > max_corner_.array() + 1e-10).any()) {
            return false;
        }
        double winding_sum = 0.0;
        for (const auto& face : mesh_.faces) {
            const Eigen::Vector3d& a = mesh_.vertices[face(0)];
            const Eigen::Vector3d& b = mesh_.vertices[face(1)];
            const Eigen::Vector3d& c = mesh_.vertices[face(2)];
            winding_sum += solidAngleAtPoint(x, a, b, c);
        }
        constexpr double two_pi = 6.28318530717958647692;
        return std::abs(winding_sum) > two_pi;
    }

    static Eigen::Vector3d closestPointOnTriangle(const Eigen::Vector3d& p,
                                                  const Eigen::Vector3d& a,
                                                  const Eigen::Vector3d& b,
                                                  const Eigen::Vector3d& c) {
        const Eigen::Vector3d ab = b - a;
        const Eigen::Vector3d ac = c - a;
        const Eigen::Vector3d ap = p - a;

        const double d1 = ab.dot(ap);
        const double d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0) {
            return a;
        }

        const Eigen::Vector3d bp = p - b;
        const double d3 = ab.dot(bp);
        const double d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3) {
            return b;
        }

        const double vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            const double v = d1 / (d1 - d3);
            return a + v * ab;
        }

        const Eigen::Vector3d cp = p - c;
        const double d5 = ab.dot(cp);
        const double d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) {
            return c;
        }

        const double vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            const double w = d2 / (d2 - d6);
            return a + w * ac;
        }

        const double va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            const Eigen::Vector3d bc = c - b;
            const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return b + w * bc;
        }

        const double denom = 1.0 / (va + vb + vc);
        const double v = vb * denom;
        const double w = vc * denom;
        return a + ab * v + ac * w;
    }

    static double solidAngleAtPoint(const Eigen::Vector3d& p,
                                    const Eigen::Vector3d& a,
                                    const Eigen::Vector3d& b,
                                    const Eigen::Vector3d& c) {
        const Eigen::Vector3d ra = a - p;
        const Eigen::Vector3d rb = b - p;
        const Eigen::Vector3d rc = c - p;

        const double la = ra.norm();
        const double lb = rb.norm();
        const double lc = rc.norm();
        if (la < 1e-12 || lb < 1e-12 || lc < 1e-12) {
            return 6.28318530717958647692;
        }

        const double numerator = ra.dot(rb.cross(rc));
        const double denominator =
            la * lb * lc +
            la * rb.dot(rc) +
            lb * rc.dot(ra) +
            lc * ra.dot(rb);

        return 2.0 * std::atan2(numerator, denominator);
    }
};

}  // namespace vde
