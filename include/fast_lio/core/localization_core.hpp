// FAST-LIO localization algorithm core, separated from the ROS2 node.
//
// Pure algorithm: IMU integration and scan undistortion, the iterated-EKF
// update against a static prior map, and the localization health check. It
// performs no file IO and no ROS communication; the prior map is injected
// through set_map() as an immutable shared asset, so a map swap is a single
// atomic pointer replacement.

#pragma once

#include <memory>
#include <vector>

#include <rclcpp/logger.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <pcl/filters/voxel_grid.h>

#include <fast_lio/common/common_lib.h>
#include <fast_lio/common/params.hpp>
#include <fast_lio/common/use-ikfom.hpp>
#include <fast_lio/core/health_report.hpp>
#include <fast_lio/core/prior_map.hpp>
#include <fast_lio/imu/imu_processing.hpp>

namespace fast_lio
{

// Per-scan wall-clock stamps captured inside process_scan(). All fields are
// absolute omp_get_wtime() stamps except solve_H_time (a duration written by
// the EKF update). The node computes the [debug-timing] deltas from them.
struct ScanTiming
{
  double t_imu_end = 0.0;       // after imu_process_->Process
  double t1 = 0.0;              // after surface downsampling
  double t2 = 0.0;              // after match-index prep
  double t_update_start = 0.0;  // before the iterated update
  double t_update_end = 0.0;    // after the iterated update
  double solve_H_time = 0.0;    // iterated-update output: time spent building H
};

class LocalizationCore
{
public:
  explicit LocalizationCore(const Params & params, const rclcpp::Logger & logger);
  ~LocalizationCore();  // clears the esekfom trampoline pointer

  // ---------------------------------------------------------------------------
  // Map
  // ---------------------------------------------------------------------------
  // Inject (or atomically replace) the prior map. The core only reads from it.
  void set_map(PriorMapPtr map) { map_ = std::move(map); }
  bool has_map() const { return map_ != nullptr; }

  // ---------------------------------------------------------------------------
  // Algorithm entry points (driven by the node's timer_callback)
  // ---------------------------------------------------------------------------
  // (Re-)initialize the IMU process and the iterated-EKF for a fresh
  // /initialpose seed.
  void init_filter();
  // Run one scan through the EKF pipeline (IMU integration, downsampling,
  // iterated update). Returns false when the scan produced no publishable
  // update (no map, empty features, too few points).
  bool process_scan(const MeasureGroup & meas, ScanTiming & timing);
  // Seed the filter with a refined T_map_base.
  void set_initial_localization(const Eigen::Matrix4d & T_map_base);
  // Re-arm the post-relocalization verification state after a re-seed.
  void rearm_reinit(const Eigen::Vector3d & seed_pos, double lidar_beg_time);
  void set_first_lidar_time(double t);
  // Map-support + motion sanity check on the current pose. On failure it sets
  // localization_failed_ and the returned report says why.
  HealthReport localization_health_check(double lidar_beg_time);

  // ---------------------------------------------------------------------------
  // Point transforms (public for the node's publish helpers)
  // ---------------------------------------------------------------------------
  void point_body_to_world(PointType const * const pi, PointType * const po);
  void RGBpointBodyToWorld(PointType const * const pi, PointType * const po);
  void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po);
  template <typename T>
  void set_posestamp(T & out)
  {
    out.pose.position.x = state_point_.pos(0);
    out.pose.position.y = state_point_.pos(1);
    out.pose.position.z = state_point_.pos(2);
    out.pose.orientation.x = geoQuat_.x;
    out.pose.orientation.y = geoQuat_.y;
    out.pose.orientation.z = geoQuat_.z;
    out.pose.orientation.w = geoQuat_.w;
  }

  // ---------------------------------------------------------------------------
  // State accessors for publishing / diagnostics
  // ---------------------------------------------------------------------------
  const Eigen::Matrix<double, 23, 23> & get_P() const { return kf_->get_P(); }
  const PointCloudXYZI::Ptr & feats_undistort() const { return feats_undistort_; }
  const PointCloudXYZI::Ptr & feats_down_body() const { return feats_down_body_; }
  const PointCloudXYZI::Ptr & laser_cloud_ori() const { return laserCloudOri_; }
  int effct_feat_num() const { return effct_feat_num_; }
  int feats_down_size() const { return feats_down_size_; }
  double res_mean_last() const { return res_mean_last_; }
  bool is_initialized() const { return kf_ != nullptr; }
  bool localization_failed() const { return localization_failed_; }
  // True while the post-relocalization verification window is still running.
  bool verifying() const { return relocalized_verify_; }
  void set_localization_failed(bool failed) { localization_failed_ = failed; }

private:
  // esekfom stores the measurement model as a raw C function pointer, so this
  // is a static trampoline to the private h_share_model_impl (reached through
  // the file-local core pointer in localization_core.cpp).
  static void h_share_model(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data);
  void h_share_model_impl(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data);

  Params params_;
  rclcpp::Logger logger_;
  int num_max_iterations_ = 0;

  PriorMapPtr map_;  // immutable prior map; only read from
  std::shared_ptr<ImuProcess> imu_process_;

  // ---------------------------------------------------------------------------
  // EKF / state
  // ---------------------------------------------------------------------------
  std::unique_ptr<esekfom::esekf<state_ikfom, 12, input_ikfom>> kf_;
  state_ikfom state_point_;
  vect3 pos_lid_;
  double epsi_[23] = {0.001};
  V3D euler_cur_;
  geometry_msgs::msg::Quaternion geoQuat_;
  double first_lidar_time_ = 0.0;

  // ---------------------------------------------------------------------------
  // Point clouds and match buffers
  // ---------------------------------------------------------------------------
  PointCloudXYZI::Ptr feats_undistort_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr feats_down_body_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr feats_down_world_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr normvec_{new PointCloudXYZI(100000, 1)};
  PointCloudXYZI::Ptr laserCloudOri_{new PointCloudXYZI(100000, 1)};
  PointCloudXYZI::Ptr corr_normvect_{new PointCloudXYZI(100000, 1)};

  pcl::VoxelGrid<PointType> downSizeFilterSurf_;

  int feats_down_size_ = 0;
  int effct_feat_num_ = 0;
  bool point_selected_surf_[100000];
  float res_last_[100000];
  std::vector<std::vector<int>> pointSearchInd_surf_;
  std::vector<PointVector> Nearest_Points_;

  // ---------------------------------------------------------------------------
  // Localization failure-detection state
  // ---------------------------------------------------------------------------
  bool localization_failed_ = false;
  Eigen::Vector3d last_good_pos_ = Eigen::Vector3d::Zero();  // last healthy pose position [map]
  double last_good_time_ = 0.0;   // lidar_beg_time of the last healthy update
  bool last_good_valid_ = false;  // false until the first healthy update
  // Post-relocalization verification state: while relocalized_verify_ is true the
  // pose must stay within relocalization_verify_max_drift of relocalized_pos_ and
  // travel less than relocalization_verify_max_path in total (accumulated
  // scan-to-scan displacement in relocalized_path_) until relocalized_time_ +
  // relocalization_verify_window (see localization_health_check()). Re-armed by
  // rearm_reinit() every time a /initialpose is consumed.
  bool relocalized_verify_ = false;
  Eigen::Vector3d relocalized_pos_ = Eigen::Vector3d::Zero();
  double relocalized_time_ = 0.0;
  double relocalized_path_ = 0.0;  // total pose path traveled since the re-seed

  // ---------------------------------------------------------------------------
  // Per-scan timing (for the debug_time_usage printout and diagnostics)
  // ---------------------------------------------------------------------------
  double match_time_ = 0.0, solve_time_ = 0.0;
  double res_mean_last_ = 0.05, total_residual_ = 0.0;
};

}  // namespace fast_lio
