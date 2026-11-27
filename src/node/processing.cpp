// Per-scan orchestration: the 100 Hz timer that synchronizes the sensor
// buffers, consumes /initialpose seeds (Open3D relocalization + EKF re-init),
// runs the EKF update and drives the localization business state machine.

#include <fast_lio/node/localization_node.hpp>

#include <cstdio>

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
  // Serializes all core_/localizer_ access against map switches and lifecycle
  // transitions (see the concurrency note in localization_node.hpp).
  std::lock_guard<std::mutex> pipeline_lock(pipeline_mtx_);

  if (!sync_packages(measures_))
  {
    return;
  }

  // First synchronized scan after a (re-)initial pose: (re-)initialize the
  // filter with the pose received on /initialpose or the ~/relocalize service.
  if (initial_pose_set_.load())
  {
    // If a relocalize service call queued this seed, remember its sequence
    // number so the outcome below is handed back to that blocked caller.
    uint64_t reloc_seq = 0;
    {
      std::lock_guard<std::mutex> lock(reloc_mtx_);
      if (reloc_waiter_)
      {
        reloc_seq = reloc_req_seq_;
      }
    }

    // Initial guess: T_map_base built from the queued pose (a 2D click or a
    // service request — z is forced to the configured init_z height).
    geometry_msgs::msg::PoseWithCovarianceStamped init_pose;
    {
      std::lock_guard<std::mutex> lock(init_pose_mtx_);
      init_pose = init_pose_;
    }
    Eigen::Matrix4d T_map_base = initial_pose_to_matrix(init_pose, init_z_);
    const bool seamless = seamless_seed_.exchange(false);

    // Outcome of this re-init, reported to the pending relocalize service
    // call (if any). complete_pending_reloc() is a no-op for seq 0, so the
    // topic and map-switch paths are unaffected.
    fast_lio::srv::Relocalize::Response reloc_res;
    auto finish_reloc = [&](bool success, const std::string & message) {
      reloc_res.success = success;
      reloc_res.message = message;
      complete_pending_reloc(reloc_seq, reloc_res);
    };

    // Seamless map switch with a live filter: the cross-map pose is trusted,
    // so only the pose is replaced — the converged IMU state (velocity,
    // biases, gravity) and the sensor stream continue uninterrupted, allowing
    // the robot to keep MOVING across a map boundary. The stationary fly-away
    // guard is skipped for the same reason.
    if (seamless && core_->is_initialized())
    {
      core_->reseed_pose(T_map_base);
      core_->rearm_reinit(T_map_base.block<3, 1>(0, 3), measures_.lidar_beg_time, false);
      initial_pose_set_.store(false);
      set_state(LocalizationState::kTracking);
      RCLCPP_INFO(
        get_logger(),
        "Seamless map switch: pose reseeded directly (IMU state kept, no scan matching).");
      reloc_res.refined_pose = matrix_to_pose(T_map_base);
      finish_reloc(true, "seamless pose seed (no scan matching)");
      return;
    }

    set_state(LocalizationState::kRelocalizing);
    core_->init_filter();

    // Refine the guess by matching the current scan against the loaded map.
    // Only a trustworthy match may seed the EKF: the raw RViz click is a 2D
    // approximation that can land in a local minimum and later fly away, so a
    // rejected match is treated as a failure (stay stopped, wait for a better
    // initial pose) instead of a fallback that seeds the raw guess.
    //
    // Exception: a seamless seed (filter not yet initialized, e.g. right after
    // the first activation) is still trusted and skips the scan matching.
    bool match_ok = false;
    fast_lio::AlignResult align_res;  // valid only when align_attempted
    bool align_attempted = false;
    if (seamless)
    {
      RCLCPP_INFO(
        get_logger(),
        "Seamless seed before filter init: trusting the provided pose, skipping "
        "scan-matching relocalization.");
      match_ok = true;
    }
    else if (localizer_ready_ && localizer_ && !measures_.lidar->empty())
    {
      const auto scan_xyz = to_point_xyz(measures_.lidar);
      align_res = localizer_->align(scan_xyz, T_map_base);
      align_attempted = true;
      {
        std::lock_guard<std::mutex> lock(telemetry_.mtx);
        telemetry_.d.reloc_attempted = true;
        telemetry_.d.reloc_accepted = align_res.success;
        telemetry_.d.reloc_fitness = align_res.fitness;
        telemetry_.d.reloc_rmse = align_res.rmse;
        telemetry_.d.reloc_trans_delta = align_res.translation_delta;
        telemetry_.d.reloc_rot_delta = align_res.rotation_delta;
        telemetry_.d.reloc_ms = align_res.total_ms;
      }
      if (align_res.success)
      {
        RCLCPP_INFO(
          this->get_logger(),
          "Relocalization accepted: fitness %.3f, correction %.2f m / %.2f rad "
          "(%.0f ms, scan %zu pts).",
          align_res.fitness, align_res.translation_delta, align_res.rotation_delta, align_res.total_ms,
          align_res.scan_points);
        T_map_base = align_res.transform;
        match_ok = true;
      }
      else
      {
        RCLCPP_WARN(
          this->get_logger(),
          "Relocalization rejected (fitness %.3f, correction %.2f m / %.2f rad, "
          "scan %zu pts); not seeding the EKF.",
          align_res.fitness, align_res.translation_delta, align_res.rotation_delta, align_res.scan_points);
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
      char buf[160];
      snprintf(
        buf, sizeof(buf), "relocalization rejected (fitness %.3f, correction %.2f m / %.2f rad)",
        align_res.fitness, align_res.translation_delta, align_res.rotation_delta);
      RCLCPP_WARN(
        this->get_logger(),
        "Relocalization did not produce a trusted pose; localization stays "
        "stopped until a new %s is received.",
        params_.initial_pose_topic.c_str());
      reloc_res.refined_pose = matrix_to_pose(T_map_base);  // the raw guess
      if (align_attempted)
      {
        reloc_res.fitness = align_res.fitness;
        reloc_res.rmse = align_res.rmse;
        reloc_res.translation_correction = align_res.translation_delta;
        reloc_res.rotation_correction = align_res.rotation_delta;
      }
      finish_reloc(false, buf);
      return;
    }

    core_->set_initial_localization(T_map_base);
    core_->rearm_reinit(T_map_base.block<3, 1>(0, 3), measures_.lidar_beg_time, !seamless);
    core_->set_first_lidar_time(measures_.lidar_beg_time);

    initial_pose_set_.store(false);
    set_state(core_->verifying() ? LocalizationState::kVerifying : LocalizationState::kTracking);

    reloc_res.refined_pose = matrix_to_pose(T_map_base);
    if (align_attempted)
    {
      reloc_res.fitness = align_res.fitness;
      reloc_res.rmse = align_res.rmse;
      reloc_res.translation_correction = align_res.translation_delta;
      reloc_res.rotation_correction = align_res.rotation_delta;
      char buf[160];
      snprintf(
        buf, sizeof(buf), "localized (fitness %.3f, correction %.2f m / %.2f rad)",
        align_res.fitness, align_res.translation_delta, align_res.rotation_delta);
      finish_reloc(true, buf);
    }
    else if (seamless)
    {
      finish_reloc(true, "seeded with trusted seamless pose (no scan matching)");
    }
    else
    {
      finish_reloc(true, "seeded with raw guess (relocalizer unavailable)");
    }
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
    {
      std::lock_guard<std::mutex> lock(telemetry_.mtx);
      telemetry_.d.scan_valid = true;
      telemetry_.d.health = report;
    }
    RCLCPP_WARN(
      get_logger(),
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

  /*** Diagnostics snapshot for the /diagnostics tasks ***/
  {
    std::lock_guard<std::mutex> lock(telemetry_.mtx);
    telemetry_.d.scan_valid = true;
    telemetry_.d.health = report;
    telemetry_.d.feats_down = core_->feats_down_size();
    telemetry_.d.effct_feat_num = core_->effct_feat_num();
    telemetry_.d.res_mean = core_->res_mean_last();
    telemetry_.d.scan_ms = (omp_get_wtime() - t0) * 1e3;
  }

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
