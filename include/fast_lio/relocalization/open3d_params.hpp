// Tunable Open3D registration parameters (mirror the yaml
// localization.relocalization.* block). Kept in a standalone header free of
// Open3D/PCL includes so the parameter struct can be shared without pulling
// the heavy registration dependencies into every translation unit.

#pragma once

#include <vector>

namespace fast_lio {

struct Open3DParams {
  // ---- Coarse (FPFH + RANSAC) + fine (multiscale ICP) registration --------
  double voxel_size          = 0.2;  // voxel used to downsample each SCAN for FPFH & ICP [m]
  double map_voxel_size      = 0.2;  // voxel used to downsample the LOADED MAP target [m].
                                     // Independent of voxel_size: raise it to build the map
                                     // target faster / with less memory.
  std::vector<double> scale  = {1.0, 4.0, 6.0};  // multiscale ICP voxel ladder (× voxel_size, coarse first)
  int    icp_method          = 1;    // 0 = point-to-point, 1 = point-to-plane, 2 = GICP
  int    icp_iteration       = 30;   // iterations per ICP level
  bool   use_fpfh            = true; // run coarse FPFH+RANSAC before ICP
  int    ransac_max_iteration = 100000; // coarse RANSAC budget. With a good initial guess a few
                                        // hundred/thousand suffice; raise if guesses are bad.
  bool   mutual_filter       = true; // FPFH correspondence mutual filtering
  bool   statistical_filter  = true; // RemoveStatisticalOutliers on scan+map
  int    filter_neighbors    = 50;   //   statistical filter neighbors
  double filter_std_ratio    = 3.0;  //   statistical filter std ratio
  // Crop the map to a cube (±crop_radius m) around the initial guess before
  // building the registration target. Huge speedup on large maps: FPFH, RANSAC
  // and ICP only see the local region. 0 disables the crop.
  double crop_radius         = 60.0;

  // ---- Validity thresholds (decide whether the initial guess was good) ----
  double min_fitness            = 0.5;   // min overlap ratio of the final transform (0..1)
  double max_translation_delta  = 5.0;   // max correction the pipeline may apply [m]
  double max_rotation_delta     = 0.785; // max correction the pipeline may apply [rad]
};

}  // namespace fast_lio
