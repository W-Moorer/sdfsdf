/**
 * @file SpatialHash.h
 * @brief Spatial hash grid for broad-phase collision detection
 *
 * Implements uniform grid-based spatial hashing to accelerate
 * collision detection between rigid bodies.
 */

#pragma once

#include <Eigen/Core>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <algorithm>

namespace vde {

/**
 * @brief 3D integer grid cell coordinates
 */
struct GridCell {
    int x, y, z;
    
    GridCell(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    bool operator==(const GridCell& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

/**
 * @brief Hash function for GridCell
 */
struct GridCellHash {
    std::size_t operator()(const GridCell& cell) const {
        // Spatial hash using large prime numbers
        const int64_t p1 = 73856093;
        const int64_t p2 = 19349663;
        const int64_t p3 = 83492791;
        
        return static_cast<std::size_t>(
            (static_cast<int64_t>(cell.x) * p1) ^
            (static_cast<int64_t>(cell.y) * p2) ^
            (static_cast<int64_t>(cell.z) * p3)
        );
    }
};

/**
 * @brief AABB (Axis-Aligned Bounding Box)
 */
struct AABB {
    Eigen::Vector3d min;
    Eigen::Vector3d max;
    
    AABB() 
        : min(Eigen::Vector3d::Constant(std::numeric_limits<double>::max())),
          max(Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest())) {}
    
    AABB(const Eigen::Vector3d& min_, const Eigen::Vector3d& max_)
        : min(min_), max(max_) {}
    
    /**
     * @brief Expand AABB to include a point
     */
    void expand(const Eigen::Vector3d& point) {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }
    
    /**
     * @brief Expand AABB by a margin
     */
    void expand(double margin) {
        min -= Eigen::Vector3d::Constant(margin);
        max += Eigen::Vector3d::Constant(margin);
    }
    
    /**
     * @brief Check intersection with another AABB
     */
    bool intersects(const AABB& other) const {
        return (min.array() <= other.max.array()).all() &&
               (max.array() >= other.min.array()).all();
    }
    
    /**
     * @brief Get center of AABB
     */
    Eigen::Vector3d center() const {
        return (min + max) * 0.5;
    }
    
    /**
     * @brief Get size of AABB
     */
    Eigen::Vector3d size() const {
        return max - min;
    }
    
    /**
     * @brief Check if AABB is valid
     */
    bool isValid() const {
        return (min.array() <= max.array()).all();
    }

    /**
     * @brief Check if AABB is empty (for compatibility with VolumetricIntegrator)
     */
    bool isEmpty() const {
        return (max - min).minCoeff() <= 0;
    }

    /**
     * @brief Compute intersection of two AABBs
     */
    AABB intersection(const AABB& other) const {
        Eigen::Vector3d new_min = min.cwiseMax(other.min);
        Eigen::Vector3d new_max = max.cwiseMin(other.max);

        // If no intersection, return empty bounding box
        if ((new_max - new_min).minCoeff() < 0) {
            return AABB(new_min, new_min);  // Empty bounding box
        }

        return AABB(new_min, new_max);
    }

    /**
     * @brief Compute bounding box volume
     */
    double volume() const {
        Eigen::Vector3d s = size();
        return s.x() * s.y() * s.z();
    }
};

/**
 * @brief Spatial hash grid for broad-phase collision detection
 *
 * Uses uniform grid to partition space and quickly find
 * potentially colliding object pairs.
 */
class SpatialHash {
public:
    /**
     * @brief Constructor with cell size
     * @param cell_size Size of each grid cell
     */
    explicit SpatialHash(double cell_size = 1.0)
        : cell_size_(cell_size) {}
    
    /**
     * @brief Clear all stored objects
     */
    void clear() {
        grid_.clear();
        object_cells_.clear();
    }
    
    /**
     * @brief Insert an object with its AABB
     * @param object_id Unique object identifier
     * @param aabb Object's axis-aligned bounding box
     */
    void insert(int object_id, const AABB& aabb) {
        // Get all cells covered by this AABB
        std::vector<GridCell> cells = getCoveredCells(aabb);
        
        // Store cells for this object
        object_cells_[object_id] = cells;
        
        // Insert into grid
        for (const auto& cell : cells) {
            grid_[cell].push_back(object_id);
        }
    }
    
    /**
     * @brief Remove an object from the grid
     * @param object_id Object to remove
     */
    void remove(int object_id) {
        auto it = object_cells_.find(object_id);
        if (it == object_cells_.end()) return;
        
        // Remove from all cells
        for (const auto& cell : it->second) {
            auto& cell_objects = grid_[cell];
            cell_objects.erase(
                std::remove(cell_objects.begin(), cell_objects.end(), object_id),
                cell_objects.end()
            );
        }
        
        object_cells_.erase(it);
    }
    
    /**
     * @brief Update an object's position
     * @param object_id Object identifier
     * @param aabb New AABB
     */
    void update(int object_id, const AABB& aabb) {
        remove(object_id);
        insert(object_id, aabb);
    }
    
    /**
     * @brief Find all potential collision pairs
     * @return Vector of object ID pairs that might collide
     */
    std::vector<std::pair<int, int>> findPotentialCollisions() const {
        std::unordered_set<uint64_t> pair_set;
        std::vector<std::pair<int, int>> pairs;
        
        for (const auto& [cell, objects] : grid_) {
            // Check all pairs in this cell
            for (size_t i = 0; i < objects.size(); ++i) {
                for (size_t j = i + 1; j < objects.size(); ++j) {
                    int id1 = objects[i];
                    int id2 = objects[j];
                    
                    // Create unique pair key
                    uint64_t key = createPairKey(id1, id2);
                    
                    if (pair_set.insert(key).second) {
                        // New pair
                        if (id1 < id2) {
                            pairs.emplace_back(id1, id2);
                        } else {
                            pairs.emplace_back(id2, id1);
                        }
                    }
                }
            }
        }
        
        return pairs;
    }
    
    /**
     * @brief Find potential collisions for a specific object
     * @param object_id Object to query
     * @param aabb Object's AABB
     * @return Vector of potentially colliding object IDs
     */
    std::vector<int> findPotentialCollisions(int object_id, const AABB& aabb) const {
        std::unordered_set<int> neighbors;
        std::vector<GridCell> cells = getCoveredCells(aabb);
        
        for (const auto& cell : cells) {
            auto it = grid_.find(cell);
            if (it != grid_.end()) {
                for (int other_id : it->second) {
                    if (other_id != object_id) {
                        neighbors.insert(other_id);
                    }
                }
            }
        }
        
        return std::vector<int>(neighbors.begin(), neighbors.end());
    }
    
    /**
     * @brief Get number of cells in the grid
     */
    size_t numCells() const {
        return grid_.size();
    }
    
    /**
     * @brief Get total number of object-cell entries
     */
    size_t numEntries() const {
        size_t count = 0;
        for (const auto& [cell, objects] : grid_) {
            count += objects.size();
        }
        return count;
    }
    
    /**
     * @brief Set cell size
     */
    void setCellSize(double cell_size) {
        cell_size_ = cell_size;
        clear();
    }
    
    /**
     * @brief Get cell size
     */
    double cellSize() const {
        return cell_size_;
    }

private:
    double cell_size_;
    std::unordered_map<GridCell, std::vector<int>, GridCellHash> grid_;
    std::unordered_map<int, std::vector<GridCell>> object_cells_;
    
    /**
     * @brief Convert world position to grid cell
     */
    GridCell worldToCell(const Eigen::Vector3d& pos) const {
        return GridCell(
            static_cast<int>(std::floor(pos.x() / cell_size_)),
            static_cast<int>(std::floor(pos.y() / cell_size_)),
            static_cast<int>(std::floor(pos.z() / cell_size_))
        );
    }
    
    /**
     * @brief Get all cells covered by an AABB
     */
    std::vector<GridCell> getCoveredCells(const AABB& aabb) const {
        std::vector<GridCell> cells;
        
        GridCell min_cell = worldToCell(aabb.min);
        GridCell max_cell = worldToCell(aabb.max);
        
        for (int x = min_cell.x; x <= max_cell.x; ++x) {
            for (int y = min_cell.y; y <= max_cell.y; ++y) {
                for (int z = min_cell.z; z <= max_cell.z; ++z) {
                    cells.emplace_back(x, y, z);
                }
            }
        }
        
        return cells;
    }
    
    /**
     * @brief Create unique key for an object pair
     */
    static uint64_t createPairKey(int id1, int id2) {
        // Combine two 32-bit IDs into one 64-bit key
        uint32_t u1 = static_cast<uint32_t>(id1);
        uint32_t u2 = static_cast<uint32_t>(id2);
        return (static_cast<uint64_t>(std::min(u1, u2)) << 32) | 
                static_cast<uint64_t>(std::max(u1, u2));
    }
};

/**
 * @brief Broad-phase collision detector using spatial hash
 */
class BroadPhaseDetector {
public:
    explicit BroadPhaseDetector(double cell_size = 1.0)
        : spatial_hash_(cell_size) {}
    
    /**
     * @brief Update object positions and find collision pairs
     * @param object_aabbs Map of object ID to AABB
     * @return Vector of potentially colliding pairs
     */
    std::vector<std::pair<int, int>> update(
        const std::unordered_map<int, AABB>& object_aabbs) {
        
        spatial_hash_.clear();
        
        // Insert all objects
        for (const auto& [id, aabb] : object_aabbs) {
            spatial_hash_.insert(id, aabb);
        }
        
        // Find potential collisions
        return spatial_hash_.findPotentialCollisions();
    }
    
    /**
     * @brief Get spatial hash statistics
     */
    struct Stats {
        size_t num_cells;
        size_t num_entries;
        double average_objects_per_cell;
    };
    
    Stats getStats() const {
        Stats stats;
        stats.num_cells = spatial_hash_.numCells();
        stats.num_entries = spatial_hash_.numEntries();
        stats.average_objects_per_cell = 
            stats.num_cells > 0 ? 
            static_cast<double>(stats.num_entries) / stats.num_cells : 0.0;
        return stats;
    }

private:
    SpatialHash spatial_hash_;
};

} // namespace vde
