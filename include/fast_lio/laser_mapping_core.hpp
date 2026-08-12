// FAST-LIO algorithm core, separated from the ROS2 node.
//
// Owns everything that makes up the iterated-EKF LIO algorithm: the ikd-Tree
// map and its incremental maintenance, scan-to-map surface matching, the EKF
// measurement model, IMU integration and the localization health checks. The
// node (LaserMappingNode) drives it one scan at a time and reads the result
// back for publishing; it holds no algorithm state itself.

#pragma once

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/logger.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <pcl/filters/voxel_grid.h>

#include <common_lib.h>
#include <use-ikfom.hpp>
#include <ikd-Tree/ikd_Tree.h>
#include <fast_lio/IMU_Processing.hpp>
#include <fast_lio/open3d_localizer.hpp>

namespace fast_lio
{

// Per-scan wall-clock stamps captured inside process_scan(). All fields are
// absolute omp_get_wtime() stamps except solve_H_time (a duration written by
// the EKF update). The node computes the [debug-timing] deltas from them.
struct ScanTiming
{
  double t_imu_end = 0.0;       // after imu_process_->Process
  double t_fov_end = 0.0;       // after lasermap_fov_segment
  double t1 = 0.0;              // after surface downsampling
  double t2 = 0.0;              // after match-index prep
  double t_update_start = 0.0;  // before the iterated update
  double t_update_end = 0.0;    // after the iterated update
  double solve_H_time = 0.0;    // iterated-update output: time spent building H
};

// Maximum number of processed lidar scans kept for the time log.
constexpr int kMaxN = 720000;

class LaserMappingCore
{
public:
  // ---------------------------------------------------------------------------
  // Parameters (loaded by the node from the yaml in its load_params())
  // ---------------------------------------------------------------------------
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
    // mapping / EKF
    double gyr_cov = 0.1;
    double acc_cov = 0.1;
    double b_gyr_cov = 0.0001;
    double b_acc_cov = 0.0001;
    double filter_size_surf = 0.5;
    double filter_size_map = 0.5;
    double cube_side_length = 200.0;
    float det_range = 300.0f;
    int max_iteration = 4;
    bool extrinsic_est_en = true;
    std::vector<double> extrin_T{0.0, 0.0, 0.0};
    std::vector<double> extrin_R{1., 0., 0., 0., 1., 0., 0., 0., 1.};
    // publish
    bool path_en = true;
    bool effect_map_en = false;
    bool map_en = false;
    bool scan_publish_en = true;
    bool dense_publish_en = true;
    bool scan_bodyframe_pub_en = true;
    // pcd save (only used when map incremental is enabled)
    bool pcd_save_en = false;
    std::string map_file_path = "./test.pcd";
    // logging
    bool runtime_pos_log_enable = false;
    bool debug_time_usage = false;  // per-scan timing breakdown printed each iteration
    // localization
    bool enable_localization = false;
    bool enable_map_incremental = true;
    double leaf_size = 0.05;
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
    bool relocalization_verify_en = true;           // enable the post-relocalization fly-away check
    double relocalization_verify_window = 2.0;      // [s] verification window after the re-seed
    double relocalization_verify_max_drift = 1.0;   // [m] max allowed straight-line drift from the re-seeded position
    double relocalization_verify_max_path = 2.0;    // [m] max total path the pose may travel during the window
  };

  explicit LaserMappingCore(const Params & params, const rclcpp::Logger & logger);
  ~LaserMappingCore() = default;

  // ---------------------------------------------------------------------------
  // Algorithm entry points (driven by the node's timer_callback)
  // ---------------------------------------------------------------------------
  // (Re-)initialize the IMU process and the iterated-EKF for a fresh
  // /initialpose seed.
  void init_filter();
  // Run one scan through the full EKF pipeline (IMU integration, FOV
  // segmentation, downsampling, kd-tree init, iterated update). Returns false
  // when the scan produced no publishable update (empty features, map kd-tree
  // built for the first time, or too few points); the node returns without
  // publishing on false.
  bool process_scan(const MeasureGroup & meas, ScanTiming & timing);
  // Add the matched feature points to the map kd-tree (call after odometry).
  void map_incremental();
  void load_map();
  void set_initial_localization(const Eigen::Matrix4d & T_map_base);
  // Re-arm the post-relocalization verification state after a re-seed.
  void rearm_reinit(const Eigen::Vector3d & seed_pos, double lidar_beg_time);
  void set_first_lidar_time(double t);
  // Map-support + motion sanity check on the current pose; on failure it sets
  // localization_failed_ and returns false.
  bool localization_health_check(double lidar_beg_time);
  // Runtime performance log for one scan (moving averages, mat_out.txt, csv).
  void log_scan_runtime(
    const MeasureGroup & meas, double t0, double t3, double t5, const ScanTiming & timing);
  // Close log files, save the accumulated map and write the time-log csv.
  void on_shutdown();
  // Record the preprocess time for scan #idx in the s_plot11_ column.
  void record_preprocess_time(int scan_count, double seconds);
  void set_localization_failed(bool failed) { localization_failed_ = failed; }

  // ---------------------------------------------------------------------------
  // Point transforms (public for the node's publish helpers)
  // ---------------------------------------------------------------------------
  void point_body_to_world(PointType const * const pi, PointType * const po);
  template <typename T>
  void point_body_to_world(const Matrix<T, 3, 1> & pi, Matrix<T, 3, 1> & po)
  {
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) + state_point_.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
  }
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
  // State accessors for publishing
  // ---------------------------------------------------------------------------
  const Eigen::Matrix<double, 23, 23> & get_P() const { return kf_->get_P(); }
  const PointCloudXYZI::Ptr & feats_undistort() const { return feats_undistort_; }
  const PointCloudXYZI::Ptr & feats_down_body() const { return feats_down_body_; }
  const PointCloudXYZI::Ptr & laser_cloud_ori() const { return laserCloudOri_; }
  const PointCloudXYZI::Ptr & pcl_wait_pub() const { return pcl_wait_pub_; }
  const PointCloudXYZI::Ptr & map_raw() const { return map_raw_; }
  int effct_feat_num() const { return effct_feat_num_; }
  int feats_down_size() const { return feats_down_size_; }
  bool is_initialized() const { return kf_ != nullptr; }
  bool localization_failed() const { return localization_failed_; }
  // Accumulate the current scan into the map-to-publish cloud, downsample it,
  // and return the cloud for the /Laser_map publisher.
  PointCloudXYZI::Ptr map_to_publish();

private:
  // esekfom stores the measurement model as a raw C function pointer, so this
  // is a static trampoline to the private h_share_model_impl (reached through
  // the file-local core pointer in laser_mapping_core.cpp).
  static void h_share_model(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data);
  void h_share_model_impl(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data);
  void lasermap_fov_segment();
  void points_cache_collect();
  void dump_lio_state_to_log(FILE * fp, double lidar_beg_time);

  Params params_;
  rclcpp::Logger logger_;
  int num_max_iterations_ = 0;

  std::shared_ptr<ImuProcess> imu_process_;

  // ---------------------------------------------------------------------------
  // EKF / state
  // ---------------------------------------------------------------------------
  std::unique_ptr<esekfom::esekf<state_ikfom, 12, input_ikfom>> kf_;
  state_ikfom state_point_;
  vect3 pos_lid_;
  double epsi_[23] = {0.001};
  V3D euler_cur_;
  V3D Lidar_T_wrt_IMU_{Zero3d};
  M3D Lidar_R_wrt_IMU_{Eye3d};
  geometry_msgs::msg::Quaternion geoQuat_;
  double first_lidar_time_ = 0.0;
  bool flg_EKF_inited_ = false;

  // ---------------------------------------------------------------------------
  // Point clouds, filters and map kd-tree
  // ---------------------------------------------------------------------------
  PointCloudXYZI::Ptr feats_undistort_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr feats_down_body_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr feats_down_world_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr normvec_{new PointCloudXYZI(100000, 1)};
  PointCloudXYZI::Ptr laserCloudOri_{new PointCloudXYZI(100000, 1)};
  PointCloudXYZI::Ptr corr_normvect_{new PointCloudXYZI(100000, 1)};
  PointCloudXYZI::Ptr pcl_wait_pub_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr pcl_wait_save_{new PointCloudXYZI()};
  PointCloudXYZI::Ptr map_raw_;  // full-resolution loaded map (node feeds the relocalizer)
  V3F XAxisPoint_body_{LIDAR_SP_LEN, 0.0, 0.0};
  V3F XAxisPoint_world_{LIDAR_SP_LEN, 0.0, 0.0};

  pcl::VoxelGrid<PointType> downSizeFilterSurf_;
  pcl::VoxelGrid<PointType> downSizeFilterMap_;
  KD_TREE<PointType> ikdtree_;

  int feats_down_size_ = 0;
  int effct_feat_num_ = 0;
  bool point_selected_surf_[100000];
  float res_last_[100000];
  std::vector<std::vector<int>> pointSearchInd_surf_;
  std::vector<BoxPointType> cub_needrm_;
  std::vector<PointVector> Nearest_Points_;
  BoxPointType LocalMap_Points_;
  bool Localmap_Initialized_ = false;

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
  // Debug logging / timing
  // ---------------------------------------------------------------------------
  double kdtree_incremental_time_ = 0.0, kdtree_search_time_ = 0.0, kdtree_delete_time_ = 0.0;
  double match_time_ = 0.0, solve_time_ = 0.0, solve_const_H_time_ = 0.0;
  int kdtree_size_st_ = 0, kdtree_size_end_ = 0, add_point_size_ = 0, kdtree_delete_counter_ = 0;
  double res_mean_last_ = 0.05, total_residual_ = 0.0;
  double aver_time_consu_ = 0.0, aver_time_icp_ = 0.0, aver_time_match_ = 0.0,
         aver_time_incre_ = 0.0, aver_time_solve_ = 0.0, aver_time_const_H_time_ = 0.0;
  int frame_num_ = 0;
  int time_log_counter_ = 0;
  double T1_[kMaxN], s_plot_[kMaxN], s_plot2_[kMaxN], s_plot3_[kMaxN], s_plot4_[kMaxN],
         s_plot5_[kMaxN], s_plot6_[kMaxN], s_plot7_[kMaxN], s_plot8_[kMaxN], s_plot9_[kMaxN],
         s_plot10_[kMaxN], s_plot11_[kMaxN];

  FILE * fp_ = nullptr;
  std::ofstream fout_pre_;
  std::ofstream fout_out_;
};

}  // namespace fast_lio
