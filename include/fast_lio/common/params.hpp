// All tunable parameters of the localization node and algorithm core, loaded
// by the node from the yaml config in LocalizationNode::load_params().

#pragma once

#include <string>
#include <vector>

#include <fast_lio/relocalization/open3d_params.hpp>

namespace fast_lio
{

struct Params
{
  // topics
  std::string lidar_topic = "/livox/lidar";
  std::string imu_topic = "/livox/imu";
  // frames
  std::string map_frame = "map";
  std::string base_frame = "base_link";
  std::string lidar_frame = "livox_frame";
  // time sync
  bool time_sync_en = false;
  double time_offset_lidar_to_imu = 0.0;
  // preprocess
  double blind = 0.5;
  // EKF
  double gyr_cov = 0.1;
  double acc_cov = 0.1;
  double b_gyr_cov = 0.0001;
  double b_acc_cov = 0.0001;
  double filter_size_surf = 0.5;
  int max_iteration = 4;
  bool extrinsic_est_en = true;
  std::vector<double> extrin_T{0.0, 0.0, 0.0};
  std::vector<double> extrin_R{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  // publish
  bool path_en = true;
  bool effect_map_en = false;
  bool map_en = false;
  bool scan_publish_en = true;
  bool scan_bodyframe_pub_en = true;
  // logging
  bool debug_time_usage = false;  // per-scan timing breakdown printed each iteration
  // localization
  double leaf_size = 0.05;         // [m] voxel the loaded map is downsampled to (matching resolution)
  double map_publish_leaf = 0.2;   // [m] voxel for the /Laser_map display cloud
  std::string map_path;
  double init_z = 1.2;             // base_link height in the map for an initial pose
  double tf_lookup_timeout = 5.0;  // [s] to wait for the lidar->base static TF
  std::string initial_pose_topic = "/initialpose";
  // Open3D relocalization: refine the /initialpose guess by matching the
  // current scan against the loaded map before seeding the EKF.
  bool relocalization_en = true;
  Open3DParams open3d;  // FPFH+RANSAC + multiscale-ICP tuning (localization.relocalization.*)
  // Localization failure detection: the pose must stay close to the loaded map
  // (map support) and move at a physically plausible speed, otherwise the
  // localization is declared lost and the map->base TF is suppressed until a
  // new /initialpose re-seeds the filter.
  bool failure_detect_en = true;
  double max_pose_to_map_dist = 3.0;  // [m] pose farther than this from the nearest map point
  double max_pose_speed = 3.0;        // [m/s] scan-to-scan pose displacement rate above this
  // Post-relocalization fly-away guard: right after a /initialpose re-seeds the
  // EKF the robot is expected to be stationary, so the pose must stay near the
  // re-seeded position for a short verification window. If it leaves, the
  // relocalization itself is judged failed and localization is declared lost.
  // Both the straight-line drift from the re-seeded position and the TOTAL path
  // the pose travels during the window are bounded: net drift misses an
  // oscillating pose (or one that moves away and comes back), which still ends
  // near the seed while having traveled a long path.
  bool relocalization_verify_en = true;          // enable the post-relocalization fly-away check
  double relocalization_verify_window = 2.0;     // [s] verification window after the re-seed
  double relocalization_verify_max_drift = 1.0;  // [m] max allowed straight-line drift from the re-seeded position
  double relocalization_verify_max_path = 2.0;   // [m] max total path the pose may travel during the window
};

}  // namespace fast_lio
