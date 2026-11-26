// /diagnostics tasks of the localization node.
//
// Four tasks, all reading the lock-light Telemetry snapshot (plus immutable
// config / map info) so the 1 Hz diagnostic timer never contends with the
// EKF or the Open3D relocalizer:
//   Sensor Input          lidar/IMU rates, queue depth, timestamp regressions
//   Localization State    lifecycle state + business state + current map
//   Tracking Quality      matched points, residual, map distance, scan time
//   Relocalization & Map  last /initialpose match result and map info

#include <fast_lio/node/localization_node.hpp>

#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <lifecycle_msgs/msg/state.hpp>

namespace fast_lio
{

using DiagnosticStatus = diagnostic_msgs::msg::DiagnosticStatus;

void LocalizationNode::setup_diagnostics()
{
  updater_ = std::make_unique<diagnostic_updater::Updater>(this, 1.0);
  updater_->setHardwareID("fastlio_localization");
  updater_->add(
    "Sensor Input", std::bind(&LocalizationNode::fill_sensor_diagnostics, this, std::placeholders::_1));
  updater_->add(
    "Localization State", std::bind(&LocalizationNode::fill_state_diagnostics, this, std::placeholders::_1));
  updater_->add(
    "Tracking Quality", std::bind(&LocalizationNode::fill_tracking_diagnostics, this, std::placeholders::_1));
  updater_->add(
    "Relocalization & Map", std::bind(&LocalizationNode::fill_relocation_diagnostics, this, std::placeholders::_1));
}

void LocalizationNode::fill_sensor_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  const auto t = telemetry_.snapshot();

  const bool active = get_current_state().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE;
  stat.add("lidar_hz", t.lidar_hz);
  stat.add("imu_hz", t.imu_hz);
  stat.add("lidar_queue", static_cast<int>(t.lidar_queue));
  stat.add("imu_queue", static_cast<int>(t.imu_queue));
  stat.add("lidar_frames_total", static_cast<int>(t.lidar_frames));
  stat.add("imu_frames_total", static_cast<int>(t.imu_frames));
  stat.add("lidar_time_regressions", static_cast<int>(t.lidar_regressions));

  if (!active)
  {
    stat.summary(DiagnosticStatus::OK, "inactive (no subscriptions)");
    return;
  }

  const double now = steady_now_sec();
  const double lidar_age = t.last_lidar_wall > 0.0 ? now - t.last_lidar_wall : 1e9;
  const double imu_age = t.last_imu_wall > 0.0 ? now - t.last_imu_wall : 1e9;
  stat.add("lidar_age_sec", lidar_age);
  stat.add("imu_age_sec", imu_age);

  if (lidar_age > 1.0 || imu_age > 1.0)
  {
    stat.summary(DiagnosticStatus::ERROR, "sensor data stalled (>1s without frames)");
  }
  else if (t.lidar_hz < 5.0 || t.imu_hz < 50.0 || t.lidar_queue > 30)
  {
    stat.summary(DiagnosticStatus::WARN, "low sensor rate or deep lidar queue");
  }
  else
  {
    stat.summary(DiagnosticStatus::OK, "sensor input nominal");
  }
}

void LocalizationNode::fill_state_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  const auto loc_state = state_.load();
  const auto map = std::atomic_load(&map_);
  stat.add("lifecycle_state", get_current_state().label());
  stat.add("localization_state", toString(loc_state));
  stat.add("map_path", map ? map->info().path : std::string("(none)"));
  stat.add("map_points", map ? static_cast<int>(map->info().map_points) : 0);

  if (loc_state == LocalizationState::kLost)
  {
    stat.summary(DiagnosticStatus::ERROR, "localization LOST; waiting for a new initial pose");
  }
  else if (loc_state == LocalizationState::kTracking)
  {
    stat.summary(DiagnosticStatus::OK, "tracking");
  }
  else
  {
    stat.summary(DiagnosticStatus::WARN, std::string("not tracking: ") + toString(loc_state));
  }
}

void LocalizationNode::fill_tracking_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  const auto t = telemetry_.snapshot();

  if (!t.scan_valid)
  {
    stat.summary(DiagnosticStatus::OK, "no scan processed yet");
    return;
  }

  stat.add("scan_points_downsampled", t.feats_down);
  stat.add("matched_points", t.effct_feat_num);
  stat.add("residual_mean_m", t.res_mean);
  stat.add("nearest_map_dist_m", t.health.nearest_dist);
  stat.add("pose_speed_mps", t.health.speed);
  stat.add("scan_proc_ms", t.scan_ms);

  if (!t.health.ok)
  {
    stat.summary(DiagnosticStatus::ERROR, "health check failed: " + t.health.describe());
  }
  else if (t.effct_feat_num < 50 || t.scan_ms > 80.0)
  {
    stat.summary(DiagnosticStatus::WARN, "few matched points or slow scan processing");
  }
  else
  {
    stat.summary(DiagnosticStatus::OK, "tracking quality nominal");
  }
}

void LocalizationNode::fill_relocation_diagnostics(diagnostic_updater::DiagnosticStatusWrapper & stat)
{
  const auto t = telemetry_.snapshot();
  const auto map = std::atomic_load(&map_);

  stat.add("map_path", map ? map->info().path : std::string("(none)"));
  stat.add("map_raw_points", map ? static_cast<int>(map->info().raw_points) : 0);
  stat.add("map_matching_points", map ? static_cast<int>(map->info().map_points) : 0);

  if (!t.reloc_attempted)
  {
    stat.summary(DiagnosticStatus::OK, "no relocalization attempted yet");
    return;
  }

  stat.add("last_accepted", t.reloc_accepted);
  stat.add("fitness", t.reloc_fitness);
  stat.add("rmse_m", t.reloc_rmse);
  stat.add("correction_m", t.reloc_trans_delta);
  stat.add("correction_rad", t.reloc_rot_delta);
  stat.add("align_ms", t.reloc_ms);

  if (t.reloc_accepted)
  {
    stat.summary(DiagnosticStatus::OK, "last relocalization accepted");
  }
  else
  {
    stat.summary(DiagnosticStatus::WARN, "last relocalization REJECTED");
  }
}

}  // namespace fast_lio
