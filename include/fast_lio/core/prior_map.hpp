// Immutable prior-map asset.
//
// Owns everything derived from one PCD map file: the downsampled matching
// cloud, its static kd-tree, a PointXYZ view for the Open3D relocalizer and a
// coarser cloud for /Laser_map display. Built once (possibly off the hot path)
// and then shared as std::shared_ptr<const PriorMap>, so swapping maps is an
// atomic pointer replacement and the old map is destroyed outside any lock.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>

#include <fast_lio/common/common_lib.h>

namespace fast_lio
{

class PriorMap
{
public:
  struct Info
  {
    std::string path;
    size_t raw_points = 0;      // points in the PCD file
    size_t map_points = 0;      // after map_leaf downsampling (matching resolution)
    size_t publish_points = 0;  // after publish_leaf downsampling (display)
    Eigen::Vector3f min{0.f, 0.f, 0.f};  // bounding box of the matching cloud
    Eigen::Vector3f max{0.f, 0.f, 0.f};
  };

  // Loads and prepares a prior map from a PCD file:
  //   map_leaf     [m] voxel the matching cloud is downsampled to
  //   publish_leaf [m] voxel for the /Laser_map display cloud
  // Throws std::runtime_error when the file cannot be read or is too sparse.
  PriorMap(const std::string & pcd_path, double map_leaf, double publish_leaf);

  // k nearest map points around pt (k is NUM_MATCH_POINTS at most call sites).
  // Thread-safe for concurrent queries (const FLANN search).
  int nearestSearch(
    const PointType & pt, int k, PointVector & points, std::vector<float> & sq_dists) const;

  const PointCloudXYZI & cloud() const { return *cloud_; }
  // Coarser cloud for RViz display on /Laser_map.
  const PointCloudXYZI & publishCloud() const { return *cloud_publish_; }
  // PointXYZ view of the matching cloud, fed once to the Open3D relocalizer.
  const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloudXyz() const { return cloud_xyz_; }
  const Info & info() const { return info_; }

private:
  Info info_;
  PointCloudXYZI::Ptr cloud_;
  PointCloudXYZI::Ptr cloud_publish_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_;
  std::unique_ptr<pcl::KdTreeFLANN<PointType>> tree_;
};

using PriorMapPtr = std::shared_ptr<const PriorMap>;

}  // namespace fast_lio
