// Runtime localization business state (the inner state machine).
//
// This is orthogonal to the ROS lifecycle state (which manages infrastructure):
// it describes what the localization pipeline is doing while the node is
// running. Transitions are driven by the node:
//
//   kNoMap        no prior map loaded; waiting for a map
//   kUnlocalized  map ready; waiting for /initialpose
//   kRelocalizing an /initialpose is being consumed (Open3D match + EKF seed)
//   kVerifying    re-seeded; inside the post-relocalization fly-away window
//   kTracking     normal tracking; publishing odometry and map->base TF
//   kLost         health check failed; TF suppressed, waiting for /initialpose

#pragma once

namespace fast_lio
{

enum class LocalizationState
{
  kNoMap,
  kUnlocalized,
  kRelocalizing,
  kVerifying,
  kTracking,
  kLost,
};

inline const char * toString(LocalizationState state)
{
  switch (state)
  {
    case LocalizationState::kNoMap: return "NO_MAP";
    case LocalizationState::kUnlocalized: return "UNLOCALIZED";
    case LocalizationState::kRelocalizing: return "RELOCALIZING";
    case LocalizationState::kVerifying: return "VERIFYING";
    case LocalizationState::kTracking: return "TRACKING";
    case LocalizationState::kLost: return "LOST";
  }
  return "UNKNOWN";
}

}  // namespace fast_lio
