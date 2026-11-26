// Structured result of the per-scan localization health check.
//
// The core produces one HealthReport per processed scan; both the log lines
// and (later) the /diagnostics tasks consume the same struct, so there is a
// single source of truth for "why is localization unhealthy".

#pragma once

#include <sstream>
#include <string>

namespace fast_lio
{

struct HealthReport
{
  enum class Reason
  {
    kOk,
    kNonFinitePose,  // EKF diverged: pose contains NaN/inf
    kVerifyDrift,    // post-relocalization pose left the seed neighborhood
    kOffMap,         // pose farther than max_pose_to_map_dist from the nearest map point
    kSpeedJump,      // scan-to-scan displacement rate above max_pose_speed
  };

  bool ok = true;
  Reason reason = Reason::kOk;
  double nearest_dist = 0.0;  // [m] pose to the nearest map point
  double speed = 0.0;         // [m/s] pose displacement rate since the last healthy update
  double verify_drift = 0.0;  // [m] straight-line drift from the re-seeded position
  double verify_path = 0.0;   // [m] total path traveled since the re-seed

  // Human-readable failure description for log lines.
  std::string describe() const
  {
    std::ostringstream os;
    switch (reason)
    {
      case Reason::kOk:
        os << "healthy: nearest map point " << nearest_dist << " m, speed " << speed << " m/s";
        break;
      case Reason::kNonFinitePose:
        os << "pose is non-finite (filter diverged)";
        break;
      case Reason::kVerifyDrift:
        os << "relocalization drifted " << verify_drift << " m from the re-seeded position / traveled "
           << verify_path << " m within the verification window";
        break;
      case Reason::kOffMap:
        os << "nearest map point " << nearest_dist << " m exceeds the map-support bound";
        break;
      case Reason::kSpeedJump:
        os << "pose speed " << speed << " m/s exceeds the plausibility bound";
        break;
    }
    return os.str();
  }
};

}  // namespace fast_lio
