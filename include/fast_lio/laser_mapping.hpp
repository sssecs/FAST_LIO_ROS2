// Mid-360 localization node for FAST-LIO.
//
// This node runs the FAST-LIO iterated-EKF against a pre-built point-cloud map
// (loaded once at startup) instead of building the map online. The lidar input
// is a Livox Mid-360 streamed as sensor_msgs/PointCloud2 and the sensor data
// (lidar + IMU) is transformed from the lidar frame to the robot base frame
// through a static TF that must be available at startup.
//
// Localization flow:
//   1. load a downsampled PCD map into the ikd-Tree,
//   2. wait for an initial pose on /initialpose (RViz "2D Pose Estimate"),
//   3. run the standard FAST-LIO update against the (static) map.

#pragma once

#include <atomic>
#include <cstdio>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <common_lib.h>
#include <use-ikfom.hpp>
#include <ikd-Tree/ikd_Tree.h>
#include <fast_lio/IMU_Processing.hpp>
#include <fast_lio/preprocess.hpp>
#include <fast_lio/open3d_localizer.hpp>

namespace fast_lio
{

// Maximum number of processed lidar scans kept for the time log.
constexpr int kMaxN = 720000;

class LaserMappingNode : public rclcpp::Node
{
public:
  explicit LaserMappingNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~LaserMappingNode() override;

private:
  // ---------------------------------------------------------------------------
  // Parameters (loaded from the yaml in load_params())
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

  void load_params();

  Params params_;
  int num_max_iterations_ = 0;

  MeasureGroup measures_;

  // ---------------------------------------------------------------------------
  // Sensor buffers and synchronization
  // ---------------------------------------------------------------------------
  std::mutex mtx_buffer_;
  std::deque<double> time_buffer_;
  std::deque<PointCloudXYZI::Ptr> lidar_buffer_;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer_;
  bool lidar_pushed_ = false;
  bool is_first_lidar_ = true;
  double last_timestamp_lidar_ = 0.0;
  double last_timestamp_imu_ = -1.0;
  double timediff_lidar_wrt_imu_ = 0.0;
  bool timediff_set_flg_ = false;
  double lidar_mean_scantime_ = 0.0;
  int scan_num_ = 0;
  int scan_count_ = 0;
  int publish_count_ = 0;
  double lidar_end_time_ = 0.0;
  double first_lidar_time_ = 0.0;
  bool flg_EKF_inited_ = false;

  bool sync_packages(MeasureGroup & meas);

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

  // esekfom stores the measurement model as a raw C function pointer, so this
  // is a static trampoline to the private h_share_model_impl (reached through
  // the file-local node pointer in laser_mapping.cpp).
  static void h_share_model(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data);
  void h_share_model_impl(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data);

  // Body -> world / IMU frame point transforms (use the current state_point_).
  void point_body_to_world(PointType const * pi, PointType * po);
  template <typename T>
  void point_body_to_world(const Matrix<T, 3, 1> & pi, Matrix<T, 3, 1> & po);
  void RGBpointBodyToWorld(PointType const * pi, PointType * po);
  void RGBpointBodyLidarToIMU(PointType const * pi, PointType * po);
  template <typename T>
  void set_posestamp(T & out);
  void points_cache_collect();

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

  void lasermap_fov_segment();
  void map_incremental();
  void load_map();
  // Seed the EKF with an initial pose expressed as T_map_base (homogeneous 4x4).
  void set_initial_localization(const Eigen::Matrix4d & T_map_base);

  // ---------------------------------------------------------------------------
  // Preprocessing / IMU
  // ---------------------------------------------------------------------------
  std::shared_ptr<Preprocess> preprocess_;
  std::shared_ptr<ImuProcess> imu_process_;

  // ---------------------------------------------------------------------------
  // Messages
  // ---------------------------------------------------------------------------
  nav_msgs::msg::Path path_;
  nav_msgs::msg::Odometry odomAftMapped_;
  geometry_msgs::msg::PoseStamped msg_body_pose_;

  // ---------------------------------------------------------------------------
  // ROS handles
  // ---------------------------------------------------------------------------
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_full_body_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_effect_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_laser_cloud_map_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_init_pose_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr map_pub_timer_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  geometry_msgs::msg::TransformStamped tf_lidar_to_base_;

  // ---------------------------------------------------------------------------
  // Localization state
  // ---------------------------------------------------------------------------
  // True when a fresh /initialpose is waiting to be consumed by the re-init
  // branch in timer_callback(). In localization mode it starts false so the node
  // does nothing until the operator's first /initialpose; the constructor sets
  // it true for mapping mode (self-start from the origin on the first scan).
  std::atomic<bool> initial_pose_set_{false};
  geometry_msgs::msg::PoseWithCovarianceStamped init_pose_;
  // Open3D registration core; refines the /initialpose guess against the map.
  Open3DLocalizer localizer_;
  bool localizer_ready_ = false;  // set once the loaded map has been fed to localizer_
  // Localization failure detection state: while localization_failed_ is true the
  // map->base TF is suppressed and the filter holds its pose until a new
  // /initialpose re-seeds it (see localization_health_check()).
  bool localization_failed_ = false;
  Eigen::Vector3d last_good_pos_ = Eigen::Vector3d::Zero();  // last healthy pose position [map]
  double last_good_time_ = 0.0;   // measures_.lidar_beg_time of the last healthy update
  bool last_good_valid_ = false;  // false until the first healthy update
  // Post-relocalization verification state: while relocalized_verify_ is true the
  // pose must stay within relocalization_verify_max_drift of relocalized_pos_ and
  // travel less than relocalization_verify_max_path in total (accumulated
  // scan-to-scan displacement in relocalized_path_) until relocalized_time_ +
  // relocalization_verify_window (see localization_health_check()). Re-armed by
  // the re-init branch every time a /initialpose is consumed.
  bool relocalized_verify_ = false;
  Eigen::Vector3d relocalized_pos_ = Eigen::Vector3d::Zero();
  double relocalized_time_ = 0.0;
  double relocalized_path_ = 0.0;  // total pose path traveled since the re-seed

  // ---------------------------------------------------------------------------
  // Callbacks
  // ---------------------------------------------------------------------------
  void timer_callback();
  void map_publish_callback();
  void map_save_callback(
    const std_srvs::srv::Trigger::Request::SharedPtr req,
    std_srvs::srv::Trigger::Response::SharedPtr res);
  void initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg);
  void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg);
  void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr & msg_in);

  // ---------------------------------------------------------------------------
  // Publishing helpers
  // ---------------------------------------------------------------------------
  void publish_odometry();
  void publish_path();
  void publish_frame_world();
  void publish_frame_body();
  void publish_effect_world();
  void publish_map();
  void save_to_pcd();
  void dump_lio_state_to_log(FILE * fp);

  // Checks the current EKF pose for map support (distance to the nearest map
  // point) and physically plausible scan-to-scan motion; on failure it sets
  // localization_failed_ and returns false, on success it refreshes the
  // last-good pose reference used by the motion check.
  bool localization_health_check();

  // ---------------------------------------------------------------------------
  // Debug logging
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
