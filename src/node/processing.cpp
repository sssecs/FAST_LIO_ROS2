// Per-scan orchestration: the 100 Hz timer that synchronizes the sensor
// buffers, consumes /initialpose seeds (Open3D relocalization + EKF re-init),
// runs the EKF update and drives the localization business state machine.

#include <fast_lio/node/localization_node.hpp>

#include <omp.h>

namespace
{

// Convert a FAST_LIO scan (PointXYZINormal, base_link frame) to the plain
// PointXYZ cloud the Open3D relocalizer expects.
pcl::PointCloud<pcl::PointXYZ>::Ptr to_point_xyz(const PointCloudXYZI::Ptr & cloud)
{
  auto out = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
  out->reserve(cloud->size());
  for (const auto & p : cloud->points)
  {
    out->push_back(pcl::PointXYZ(p.x, p.y, p.z));
  }
  return out;
}

// Build the initial guess T_map_base from an RViz "2D Pose Estimate" message.
// The click is 2D: z is forced to the configured init_z height.
Eigen::Matrix4d initial_pose_to_matrix(
  const geometry_msgs::msg::PoseWithCovarianceStamped & pose, double init_z)
{
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  const auto & p = pose.pose.pose.position;
  const auto & q = pose.pose.pose.orientation;
  Eigen::Quaterniond quat(q.w, q.x, q.y, q.z);
  T.block<3, 3>(0, 0) = quat.toRotationMatrix();
  T(0, 3) = p.x;
  T(1, 3) = p.y;
  T(2, 3) = init_z;
  return T;
}

}  // namespace

namespace fast_lio
{

void LocalizationNode::timer_callback()
{
  if (!sync_packages(measures_))
  {
    return;
  }

  // First synchronized scan after a (re-)initial pose: (re-)initialize the
  // filter with the pose received on /initialpose.
  if (initial_pose_set_.load())
  {
    set_state(LocalizationState::kRelocalizing);
    core_->init_filter();

    // Initial guess: T_map_base built from the RViz /initialpose click
    // (a 2D click — z is forced to the configured init_z height).
    Eigen::Matrix4d T_map_base = initial_pose_to_matrix(init_pose_, params_.init_z);

    // Refine the guess by matching the current scan against the loaded map.
    // Only a trustworthy match may seed the EKF: the raw RViz click is a 2D
    // approximation that can land in a local minimum and later fly away, so a
    // rejected match is treated as a failure (stay stopped, wait for a better
    // initial pose) instead of a fallback that seeds the raw guess.
    bool match_ok = false;
    if (localizer_ready_ && !measures_.lidar->empty())
    {
      const auto scan_xyz = to_point_xyz(measures_.lidar);
      const auto res = localizer_.align(scan_xyz, T_map_base);
      if (res.success)
      {
        RCLCPP_INFO(
          this->get_logger(),
          "Relocalization accepted: fitness %.3f, correction %.2f m / %.2f rad "
          "(%.0f ms, scan %zu pts).",
          res.fitness, res.translation_delta, res.rotation_delta, res.total_ms,
          res.scan_points);
        T_map_base = res.transform;
        match_ok = true;
      }
      else
      {
        RCLCPP_WARN(
          this->get_logger(),
          "Relocalization rejected (fitness %.3f, correction %.2f m / %.2f rad, "
          "scan %zu pts); not seeding the EKF.",
          res.fitness, res.translation_delta, res.rotation_delta, res.scan_points);
      }
    }
    else if (!localizer_ready_)
    {
      RCLCPP_WARN(
        this->get_logger(), "Relocalizer not ready (no map fed to it); using the RViz initial pose.");
    }

    // The map was loaded but the match was rejected: the click does not line up
    // with the map. Do NOT seed the EKF with the raw guess — keep localization
    // stopped (TF suppressed) until a new /initialpose re-arms this branch.
    if (!match_ok && localizer_ready_)
    {
      core_->set_localization_failed(true);
      initial_pose_set_.store(false);
      set_state(LocalizationState::kLost);
      RCLCPP_WARN(
        this->get_logger(),
        "Relocalization did not produce a trusted pose; localization stays "
        "stopped until a new %s is received.",
        params_.initial_pose_topic.c_str());
      return;
    }

    core_->set_initial_localization(T_map_base);
    core_->rearm_reinit(T_map_base.block<3, 1>(0, 3), measures_.lidar_beg_time);
    core_->set_first_lidar_time(measures_.lidar_beg_time);

    initial_pose_set_.store(false);
    set_state(core_->verifying() ? LocalizationState::kVerifying : LocalizationState::kTracking);
    return;
  }

  // Do nothing until the operator's first /initialpose: without a pose the
  // filter has never been seeded (kf_ is null), so there is no valid pose to
  // publish. Hold here (no EKF, no TF) until one arrives.
  if (!core_->is_initialized())
  {
    RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Waiting for initial pose on %s before starting localization.",
      params_.initial_pose_topic.c_str());
    return;
  }

  // Localization was declared lost on an earlier scan (pose drifted off the
  // map or moved faster than a plausible robot): hold the pose and suppress
  // the map->base TF until a new /initialpose re-seeds the filter.
  if (core_->localization_failed())
  {
    RCLCPP_WARN_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "Localization lost (pose off the map or moving too fast); map->base TF "
      "suppressed. Send a new %s to re-seed.", params_.initial_pose_topic.c_str());
    return;
  }

  double t0 = omp_get_wtime();

  /*** run one scan through the algorithm core ***/
  ScanTiming timing;
  if (!core_->process_scan(measures_, timing))
  {
    return;
  }

  // Detect localization failure on the updated pose: if it drifted away from
  // the map support or moved faster than a plausible robot, suppress the
  // TF/odometry for this scan (the hold at the top of this callback keeps it
  // suppressed until a new /initialpose).
  const auto report = core_->localization_health_check(measures_.lidar_beg_time);
  if (!report.ok)
  {
    RCLCPP_WARN(
      this->get_logger(),
      "Localization FAILED: %s. map->base TF suppressed until a new initial pose.",
      report.describe().c_str());
    set_state(LocalizationState::kLost);
    return;
  }

  // The post-relocalization verification window has passed without incident.
  if (state_.load() == LocalizationState::kVerifying && !core_->verifying())
  {
    set_state(LocalizationState::kTracking);
  }

  /******* Publish odometry *******/
  publish_odometry();

  double t3 = omp_get_wtime();

  /******* Publish points *******/
  if (params_.path_en) publish_path();
  if (params_.scan_publish_en) publish_frame_world();
  if (params_.scan_publish_en && params_.scan_bodyframe_pub_en) publish_frame_body();
  if (params_.effect_map_en) publish_effect_world();

  /*** Debug: per-step time usage for this scan ***/
  if (params_.debug_time_usage)
  {
    double t6 = omp_get_wtime();
    RCLCPP_INFO(
      this->get_logger(),
      "[debug-timing] lidar_t=%.3f n_pts=%d | imu+undistort:%.2f downsample:%.2f "
      "prep:%.2f | eskf:%.2f | check+odom:%.2f publish:%.2f | total:%.2f ms",
      measures_.lidar_beg_time, core_->feats_down_size(),
      (timing.t_imu_end - t0) * 1e3, (timing.t1 - timing.t_imu_end) * 1e3,
      (timing.t2 - timing.t1) * 1e3, (timing.t_update_end - timing.t_update_start) * 1e3,
      (t3 - timing.t_update_end) * 1e3, (t6 - t3) * 1e3, (t6 - t0) * 1e3);
  }
}

}  // namespace fast_lio
