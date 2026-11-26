// Prior-map asset implementation. See prior_map.hpp.

#include <fast_lio/core/prior_map.hpp>

#include <cmath>
#include <stdexcept>

#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

namespace
{

// Voxel-downsample a cloud, robust against very large maps: pcl::VoxelGrid
// silently refuses to filter when the voxel index space would overflow an int
// (large extent / fine leaf), so pre-coarsen in one pass when that would
// happen and apply the requested leaf on the result.
template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr voxel_downsample(
  const typename pcl::PointCloud<PointT>::Ptr & in, double leaf)
{
  using CloudT = pcl::PointCloud<PointT>;
  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D(*in, min_pt, max_pt);
  const Eigen::Vector3f ext = (max_pt - min_pt).head<3>();

  constexpr double kMaxCells = 1e9;  // stay well below INT_MAX cells
  const double cells =
    (ext.x() / leaf + 1.0) * (ext.y() / leaf + 1.0) * (ext.z() / leaf + 1.0);

  typename CloudT::Ptr input = in;
  pcl::VoxelGrid<PointT> voxel;
  if (cells > kMaxCells)
  {
    const double pre_leaf = leaf * std::cbrt(cells / kMaxCells);
    typename CloudT::Ptr pre(new CloudT());
    voxel.setLeafSize(
      static_cast<float>(pre_leaf), static_cast<float>(pre_leaf), static_cast<float>(pre_leaf));
    voxel.setInputCloud(input);
    voxel.filter(*pre);
    input = pre;
  }

  auto out = std::make_shared<CloudT>();
  voxel.setLeafSize(
    static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
  voxel.setInputCloud(input);
  voxel.filter(*out);
  return out;
}

}  // namespace

namespace fast_lio
{

PriorMap::PriorMap(const std::string & pcd_path, double map_leaf, double publish_leaf)
{
  info_.path = pcd_path;

  // Load as plain PointXYZ: map PCDs usually carry no intensity/normal fields
  // (loading them into PointXYZINormal spams "Failed to find match for field"
  // warnings and wastes memory).
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw(new pcl::PointCloud<pcl::PointXYZ>());
  if (pcl::io::loadPCDFile(pcd_path, *raw) < 0)
  {
    throw std::runtime_error("failed to load map file: " + pcd_path);
  }
  info_.raw_points = raw->size();
  if (info_.raw_points < 10)
  {
    throw std::runtime_error("map has too few points: " + pcd_path);
  }

  // Full-resolution PointXYZ view for the Open3D relocalizer (it pre-downsamples
  // internally at its own map_voxel_size, exactly as when it was fed the raw
  // loaded map before).
  cloud_xyz_ = raw;

  // Matching cloud: downsampled exactly once, in the map frame, with no pose
  // transform applied (the old load_map() double-downsampled and pushed the
  // map through point_body_to_world with whatever state the filter happened
  // to hold).
  const auto ds_xyz = voxel_downsample<pcl::PointXYZ>(raw, map_leaf);
  cloud_.reset(new PointCloudXYZI());
  cloud_->reserve(ds_xyz->size());
  for (const auto & p : ds_xyz->points)
  {
    PointType q;
    q.x = p.x;
    q.y = p.y;
    q.z = p.z;
    q.intensity = 0.0f;
    q.normal_x = q.normal_y = q.normal_z = 0.0f;
    q.curvature = 0.0f;
    cloud_->push_back(q);
  }
  if (cloud_->size() < 10)
  {
    throw std::runtime_error("map too sparse after downsampling: " + pcd_path);
  }
  info_.map_points = cloud_->size();
  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D(*cloud_, min_pt, max_pt);
  info_.min = min_pt.head<3>();
  info_.max = max_pt.head<3>();

  // Display cloud for /Laser_map.
  const auto pub_xyz = voxel_downsample<pcl::PointXYZ>(raw, publish_leaf);
  cloud_publish_.reset(new PointCloudXYZI());
  cloud_publish_->reserve(pub_xyz->size());
  for (const auto & p : pub_xyz->points)
  {
    PointType q;
    q.x = p.x;
    q.y = p.y;
    q.z = p.z;
    cloud_publish_->push_back(q);
  }
  info_.publish_points = cloud_publish_->size();

  // Static kd-tree over the matching cloud.
  tree_.reset(new pcl::KdTreeFLANN<PointType>());
  tree_->setInputCloud(cloud_);
}

int PriorMap::nearestSearch(
  const PointType & pt, int k, PointVector & points, std::vector<float> & sq_dists) const
{
  // PCL 1.12 only exposes the index-returning overload; gather the points from
  // the (immutable) cloud ourselves. The index buffer is thread_local so the
  // OpenMP parallel matching loop does not reallocate it every call.
  thread_local std::vector<int> indices;
  indices.clear();
  sq_dists.clear();
  const int found = tree_->nearestKSearch(pt, static_cast<unsigned int>(k), indices, sq_dists);
  points.clear();
  points.reserve(indices.size());
  for (const int idx : indices)
  {
    points.push_back(cloud_->points[idx]);
  }
  return found;
}

}  // namespace fast_lio
