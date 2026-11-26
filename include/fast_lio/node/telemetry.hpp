// Lightweight telemetry snapshots feeding the /diagnostics tasks.
//
// The sensor callbacks, the per-scan loop and the relocalization branch write
// into this struct; the diagnostic tasks (1 Hz) only copy it. All accesses are
// short scalar reads/writes under one small mutex — the diagnostic callback
// never touches the EKF, the kd-tree or the Open3D localizer.

#pragma once

#include <cstdint>
#include <mutex>

#include <fast_lio/core/health_report.hpp>

namespace fast_lio
{

// Plain-data payload, assignable as a whole.
struct TelemetryData
{
  // ---- sensor input (written by the lidar/IMU callbacks) ----
  uint64_t lidar_frames = 0;
  uint64_t imu_frames = 0;
  uint64_t lidar_regressions = 0;  // lidar timestamp jump-backs (buffer resets)
  double lidar_hz = 0.0;           // exponential moving average
  double imu_hz = 0.0;
  double last_lidar_wall = 0.0;    // steady-clock seconds of the last frame
  double last_imu_wall = 0.0;
  size_t lidar_queue = 0;          // buffered scans waiting for processing
  size_t imu_queue = 0;

  // ---- tracking quality (written after each processed scan) ----
  bool scan_valid = false;  // false until the first processed scan
  HealthReport health;
  int feats_down = 0;      // downsampled scan points fed to the update
  int effct_feat_num = 0;  // points actually matched against the map
  double res_mean = 0.0;   // mean point-to-plane residual [m]
  double scan_ms = 0.0;    // wall time of the whole per-scan pipeline

  // ---- relocalization (written when an /initialpose is consumed) ----
  bool reloc_attempted = false;
  bool reloc_accepted = false;
  double reloc_fitness = 0.0;
  double reloc_rmse = 0.0;
  double reloc_trans_delta = 0.0;  // [m]
  double reloc_rot_delta = 0.0;    // [rad]
  double reloc_ms = 0.0;
};

struct Telemetry
{
  std::mutex mtx;
  TelemetryData d;

  // Locked whole-snapshot copy for the diagnostic tasks.
  TelemetryData snapshot()
  {
    std::lock_guard<std::mutex> lock(mtx);
    return d;
  }

  void reset()
  {
    std::lock_guard<std::mutex> lock(mtx);
    d = TelemetryData{};
  }
};

}  // namespace fast_lio
