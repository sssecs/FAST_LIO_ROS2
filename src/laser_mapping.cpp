// Mid-360 localization node for FAST-LIO (node logic only).
//
// This node loads a pre-built point-cloud map at startup and runs the FAST-LIO
// iterated-EKF against it once an initial pose is received on /initialpose. The
// algorithm itself lives in LaserMappingCore (laser_mapping_core.cpp); this file
// holds only the ROS2 node: parameters, TF, callbacks, buffer synchronization,
// per-scan orchestration and publishing.

#include <fast_lio/laser_mapping.hpp>

#include <functional>
#include <iostream>
#include <mutex>

#include <omp.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

// How many lidar scans between each full-cloud publish.
#define PUBFRAME_PERIOD (20)

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

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

LaserMappingNode::LaserMappingNode(const rclcpp::NodeOptions & options)
  : Node("laser_mapping", options)
{
  preprocess_ = std::make_shared<Preprocess>();

  load_params();

  init_pose_.pose.pose.position.x = 0.0;
  init_pose_.pose.pose.position.y = 0.0;
  init_pose_.pose.pose.position.z = 0.0;

  // Localization mode waits for the operator's first /initialpose before doing
  // anything; mapping mode self-starts from the origin on the first scan.
  initial_pose_set_.store(!params_.enable_localization);

  RCLCPP_INFO(
    this->get_logger(), "lidar topic: %s, imu topic: %s, frames: %s <- %s -> %s",
    params_.lidar_topic.c_str(), params_.imu_topic.c_str(), params_.map_frame.c_str(),
    params_.base_frame.c_str(), params_.lidar_frame.c_str());

  path_.header.stamp = this->get_clock()->now();
  path_.header.frame_id = params_.map_frame;

  // --- static lidar -> base TF -----------------------------------------------
  // Required to express the lidar scan and the IMU in the robot base frame.
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  try
  {
    tf_lidar_to_base_ = tf_buffer_->lookupTransform(
      params_.base_frame, params_.lidar_frame, tf2::TimePointZero,
      std::chrono::seconds(static_cast<int64_t>(params_.tf_lookup_timeout)));
    RCLCPP_INFO(this->get_logger(), "Static TF %s -> %s available.", params_.lidar_frame.c_str(), params_.base_frame.c_str());
  }
  catch (const tf2::TransformException & ex)
  {
    RCLCPP_ERROR(
      this->get_logger(), "Could not get transform %s -> %s: %s", params_.lidar_frame.c_str(),
      params_.base_frame.c_str(), ex.what());
    throw std::runtime_error("lidar -> base static TF not available at startup");
  }

  // --- algorithm core --------------------------------------------------------
  // Owns the IMU process, filters, map kd-tree and debug logs; constructed here
  // (after load_params) so it can seed the map kd-tree and open the log files.
  core_ = std::make_unique<LaserMappingCore>(params_, this->get_logger());

  // Feed the full-resolution map to the Open3D relocalizer (it pre-downsamples
  // internally at map_voxel_size), used to refine the initial pose guess.
  if (params_.relocalization_en && core_->map_raw())
  {
    const auto & raw = core_->map_raw();
    auto map_xyz = std::make_shared<Open3DLocalizer::Cloud>();
    map_xyz->reserve(raw->size());
    for (const auto & p : raw->points)
    {
      map_xyz->push_back(pcl::PointXYZ(p.x, p.y, p.z));
    }
    localizer_.setMap(map_xyz);
    localizer_ready_ = true;
    RCLCPP_INFO(this->get_logger(), "Open3D relocalizer map ready: %zu points.", localizer_.mapPoints());
  }

  // --- ROS graph -------------------------------------------------------------
  sub_pcl_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    params_.lidar_topic, rclcpp::SensorDataQoS(),
    [this](const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg) { this->standard_pcl_cbk(msg); });
  sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
    params_.imu_topic, 10,
    [this](const sensor_msgs::msg::Imu::ConstSharedPtr & msg) { this->imu_cbk(msg); });
  sub_init_pose_ = this->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
    params_.initial_pose_topic, 10,
    [this](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) { this->initial_pose_callback(msg); });

  pub_laser_cloud_full_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
  pub_laser_cloud_full_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
  pub_laser_cloud_effect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
  // transient_local so RViz (which subscribes after the map was published once
  // at startup) still receives the loaded map.
  pub_laser_cloud_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
    "/Laser_map", rclcpp::QoS(1).transient_local());
  pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
  pub_path_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

  // Publish the (loaded) map once so RViz can display it before the first pose.
  publish_map();

  auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
  timer_ = rclcpp::create_timer(
    this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

  auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
  map_pub_timer_ = rclcpp::create_timer(
    this, this->get_clock(), map_period_ms, std::bind(&LaserMappingNode::map_publish_callback, this));

  map_save_srv_ = this->create_service<std_srvs::srv::Trigger>(
    "map_save",
    std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(this->get_logger(), "Node init finished.");
}

LaserMappingNode::~LaserMappingNode()
{
  // Close the log files, save the accumulated map and write the time-log csv.
  core_->on_shutdown();
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

void LaserMappingNode::load_params()
{
  // topics
  this->declare_parameter<std::string>("common.lid_topic", "/livox/lidar");
  this->declare_parameter<std::string>("common.imu_topic", "/livox/imu");
  this->declare_parameter<bool>("common.time_sync_en", false);
  this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
  // frames
  this->declare_parameter<std::string>("frame.map_frame", "map");
  this->declare_parameter<std::string>("frame.base_frame", "base_link");
  this->declare_parameter<std::string>("frame.lidar_frame", "livox_frame");
  // preprocess
  this->declare_parameter<double>("preprocess.blind", 0.5);
  // mapping / EKF
  this->declare_parameter<double>("mapping.gyr_cov", 0.1);
  this->declare_parameter<double>("mapping.acc_cov", 0.1);
  this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
  this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
  this->declare_parameter<double>("filter_size_surf", 0.5);
  this->declare_parameter<double>("filter_size_map", 0.5);
  this->declare_parameter<double>("cube_side_length", 200.0);
  this->declare_parameter<float>("mapping.det_range", 300.0);
  this->declare_parameter<int>("max_iteration", 4);
  this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
  this->declare_parameter<std::vector<double>>("mapping.extrinsic_T", std::vector<double>());
  this->declare_parameter<std::vector<double>>("mapping.extrinsic_R", std::vector<double>());
  // publish
  this->declare_parameter<bool>("publish.path_en", true);
  this->declare_parameter<bool>("publish.effect_map_en", false);
  this->declare_parameter<bool>("publish.map_en", false);
  this->declare_parameter<bool>("publish.scan_publish_en", true);
  this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
  // pcd save
  this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
  this->declare_parameter<std::string>("map_file_path", "./test.pcd");
  // logging
  this->declare_parameter<bool>("runtime_pos_log_enable", false);
  this->declare_parameter<bool>("debug_time_usage", false);
  // localization
  this->declare_parameter<bool>("localization.enable_localization", false);
  this->declare_parameter<bool>("localization.enable_map_incremental", true);
  this->declare_parameter<double>("localization.leaf_size", 0.05);
  this->declare_parameter<std::string>("localization.map_path", "");
  this->declare_parameter<double>("localization.init_z", 1.2);
  this->declare_parameter<double>("localization.tf_lookup_timeout", 5.0);
  this->declare_parameter<std::string>("localization.initial_pose_topic", "/initialpose");
  // Open3D relocalization
  this->declare_parameter<bool>("localization.relocalization.enable", true);
  this->declare_parameter<double>("localization.relocalization.voxel_size", 0.2);
  this->declare_parameter<double>("localization.relocalization.map_voxel_size", 0.2);
  this->declare_parameter<std::vector<double>>("localization.relocalization.scale", std::vector<double>{1.0, 4.0, 6.0});
  this->declare_parameter<int>("localization.relocalization.icp_method", 1);
  this->declare_parameter<int>("localization.relocalization.icp_iteration", 30);
  this->declare_parameter<bool>("localization.relocalization.use_fpfh", true);
  this->declare_parameter<int>("localization.relocalization.ransac_max_iteration", 100000);
  this->declare_parameter<bool>("localization.relocalization.mutual_filter", true);
  this->declare_parameter<bool>("localization.relocalization.statistical_filter", true);
  this->declare_parameter<int>("localization.relocalization.filter_neighbors", 50);
  this->declare_parameter<double>("localization.relocalization.filter_std_ratio", 3.0);
  this->declare_parameter<double>("localization.relocalization.crop_radius", 60.0);
  this->declare_parameter<double>("localization.relocalization.min_fitness", 0.5);
  this->declare_parameter<double>("localization.relocalization.max_translation_delta", 5.0);
  this->declare_parameter<double>("localization.relocalization.max_rotation_delta", 0.785);
  // Localization failure detection (needs declare_parameter — the node is created
  // with default NodeOptions, so yaml overrides are not auto-declared and a bare
  // get_parameter would silently keep the struct defaults).
  this->declare_parameter<bool>("localization.failure_detect_en", true);
  this->declare_parameter<double>("localization.max_pose_to_map_dist", 3.0);
  this->declare_parameter<double>("localization.max_pose_speed", 3.0);
  // Post-relocalization fly-away guard.
  this->declare_parameter<bool>("localization.relocalization_verify_en", true);
  this->declare_parameter<double>("localization.relocalization_verify_window", 2.0);
  this->declare_parameter<double>("localization.relocalization_verify_max_drift", 1.0);
  this->declare_parameter<double>("localization.relocalization_verify_max_path", 2.0);

  // reads
  this->get_parameter("common.lid_topic", params_.lidar_topic);
  this->get_parameter("common.imu_topic", params_.imu_topic);
  this->get_parameter("common.time_sync_en", params_.time_sync_en);
  this->get_parameter("common.time_offset_lidar_to_imu", params_.time_offset_lidar_to_imu);
  this->get_parameter("frame.map_frame", params_.map_frame);
  this->get_parameter("frame.base_frame", params_.base_frame);
  this->get_parameter("frame.lidar_frame", params_.lidar_frame);
  this->get_parameter("preprocess.blind", params_.blind);
  this->get_parameter("mapping.gyr_cov", params_.gyr_cov);
  this->get_parameter("mapping.acc_cov", params_.acc_cov);
  this->get_parameter("mapping.b_gyr_cov", params_.b_gyr_cov);
  this->get_parameter("mapping.b_acc_cov", params_.b_acc_cov);
  this->get_parameter("filter_size_surf", params_.filter_size_surf);
  this->get_parameter("filter_size_map", params_.filter_size_map);
  this->get_parameter("cube_side_length", params_.cube_side_length);
  this->get_parameter("mapping.det_range", params_.det_range);
  this->get_parameter("max_iteration", params_.max_iteration);
  this->get_parameter("mapping.extrinsic_est_en", params_.extrinsic_est_en);
  this->get_parameter("mapping.extrinsic_T", params_.extrin_T);
  this->get_parameter("mapping.extrinsic_R", params_.extrin_R);
  this->get_parameter("publish.path_en", params_.path_en);
  this->get_parameter("publish.effect_map_en", params_.effect_map_en);
  this->get_parameter("publish.map_en", params_.map_en);
  this->get_parameter("publish.scan_publish_en", params_.scan_publish_en);
  this->get_parameter("publish.scan_bodyframe_pub_en", params_.scan_bodyframe_pub_en);
  this->get_parameter("pcd_save.pcd_save_en", params_.pcd_save_en);
  this->get_parameter("map_file_path", params_.map_file_path);
  this->get_parameter("runtime_pos_log_enable", params_.runtime_pos_log_enable);
  this->get_parameter("debug_time_usage", params_.debug_time_usage);
  this->get_parameter("localization.enable_localization", params_.enable_localization);
  this->get_parameter("localization.enable_map_incremental", params_.enable_map_incremental);
  this->get_parameter("localization.leaf_size", params_.leaf_size);
  this->get_parameter("localization.map_path", params_.map_path);
  this->get_parameter("localization.init_z", params_.init_z);
  this->get_parameter("localization.tf_lookup_timeout", params_.tf_lookup_timeout);
  this->get_parameter("localization.initial_pose_topic", params_.initial_pose_topic);
  this->get_parameter("localization.relocalization.enable", params_.relocalization_en);
  this->get_parameter("localization.relocalization.voxel_size", params_.open3d.voxel_size);
  this->get_parameter("localization.relocalization.map_voxel_size", params_.open3d.map_voxel_size);
  this->get_parameter("localization.relocalization.scale", params_.open3d.scale);
  this->get_parameter("localization.relocalization.icp_method", params_.open3d.icp_method);
  this->get_parameter("localization.relocalization.icp_iteration", params_.open3d.icp_iteration);
  this->get_parameter("localization.relocalization.use_fpfh", params_.open3d.use_fpfh);
  this->get_parameter("localization.relocalization.ransac_max_iteration", params_.open3d.ransac_max_iteration);
  this->get_parameter("localization.relocalization.mutual_filter", params_.open3d.mutual_filter);
  this->get_parameter("localization.relocalization.statistical_filter", params_.open3d.statistical_filter);
  this->get_parameter("localization.relocalization.filter_neighbors", params_.open3d.filter_neighbors);
  this->get_parameter("localization.relocalization.filter_std_ratio", params_.open3d.filter_std_ratio);
  this->get_parameter("localization.relocalization.crop_radius", params_.open3d.crop_radius);
  this->get_parameter("localization.relocalization.min_fitness", params_.open3d.min_fitness);
  this->get_parameter("localization.relocalization.max_translation_delta", params_.open3d.max_translation_delta);
  this->get_parameter("localization.relocalization.max_rotation_delta", params_.open3d.max_rotation_delta);
  this->get_parameter("localization.failure_detect_en", params_.failure_detect_en);
  this->get_parameter("localization.max_pose_to_map_dist", params_.max_pose_to_map_dist);
  this->get_parameter("localization.max_pose_speed", params_.max_pose_speed);
  this->get_parameter("localization.relocalization_verify_en", params_.relocalization_verify_en);
  this->get_parameter("localization.relocalization_verify_window", params_.relocalization_verify_window);
  this->get_parameter("localization.relocalization_verify_max_drift", params_.relocalization_verify_max_drift);
  this->get_parameter("localization.relocalization_verify_max_path", params_.relocalization_verify_max_path);

  preprocess_->blind = params_.blind;
  localizer_.setParams(params_.open3d);
}

// ---------------------------------------------------------------------------
// Sensor callbacks
// ---------------------------------------------------------------------------

void LaserMappingNode::standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(mtx_buffer_);
  scan_count_++;

  // Transform the scan from the lidar frame to the base frame.
  sensor_msgs::msg::PointCloud2 cloud_transformed;
  try
  {
    tf2::doTransform(*msg, cloud_transformed, tf_lidar_to_base_);
  }
  catch (const tf2::TransformException & ex)
  {
    RCLCPP_FATAL(this->get_logger(), "LiDAR transform failed: %s. Shutting down.", ex.what());
    rclcpp::shutdown();
    return;
  }

  double cur_time = get_time_sec(msg->header.stamp);
  double preprocess_start_time = omp_get_wtime();

  if (!is_first_lidar_ && cur_time < last_timestamp_lidar_)
  {
    std::cerr << "lidar loop back, clear buffer" << std::endl;
    lidar_buffer_.clear();
  }
  if (is_first_lidar_)
  {
    is_first_lidar_ = false;
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  preprocess_->process(cloud_transformed, ptr);

  lidar_buffer_.push_back(ptr);
  time_buffer_.push_back(cur_time);
  last_timestamp_lidar_ = cur_time;
  core_->record_preprocess_time(scan_count_, omp_get_wtime() - preprocess_start_time);
}

void LaserMappingNode::imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr & msg_in)
{
  publish_count_++;

  // Rotate the IMU measurement from the lidar frame to the base frame using the
  // cached static transform.
  sensor_msgs::msg::Imu imu_transformed = *msg_in;

  tf2::Quaternion q_tf;
  tf2::fromMsg(tf_lidar_to_base_.transform.rotation, q_tf);

  tf2::Quaternion q_imu;
  tf2::fromMsg(msg_in->orientation, q_imu);
  imu_transformed.orientation = tf2::toMsg(q_tf * q_imu);

  tf2::Vector3 gyro(
    msg_in->angular_velocity.x, msg_in->angular_velocity.y, msg_in->angular_velocity.z);
  gyro = tf2::quatRotate(q_tf, gyro);
  imu_transformed.angular_velocity.x = gyro.x();
  imu_transformed.angular_velocity.y = gyro.y();
  imu_transformed.angular_velocity.z = gyro.z();

  tf2::Vector3 accel(
    msg_in->linear_acceleration.x, msg_in->linear_acceleration.y, msg_in->linear_acceleration.z);
  accel = tf2::quatRotate(q_tf, accel);
  imu_transformed.linear_acceleration.x = accel.x();
  imu_transformed.linear_acceleration.y = accel.y();
  imu_transformed.linear_acceleration.z = accel.z();

  auto msg = std::make_shared<sensor_msgs::msg::Imu>(imu_transformed);
  msg->header.stamp =
    get_ros_time(get_time_sec(msg_in->header.stamp) - params_.time_offset_lidar_to_imu);

  if (std::abs(timediff_lidar_wrt_imu_) > 0.1 && params_.time_sync_en)
  {
    msg->header.stamp = rclcpp::Time(timediff_lidar_wrt_imu_ + get_time_sec(msg_in->header.stamp));
  }

  double timestamp = get_time_sec(msg->header.stamp);

  {
    std::lock_guard<std::mutex> lock(mtx_buffer_);
    if (timestamp < last_timestamp_imu_)
    {
      std::cerr << "IMU loop back, clear buffer" << std::endl;
      imu_buffer_.clear();
    }
    last_timestamp_imu_ = timestamp;
    imu_buffer_.push_back(msg);
  }
}

void LaserMappingNode::initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  init_pose_ = *msg;
  initial_pose_set_.store(true);
  RCLCPP_INFO(this->get_logger(), "Initial pose received from Rviz and applied.");
}

// ---------------------------------------------------------------------------
// Sensor synchronization and main processing loop
// ---------------------------------------------------------------------------

bool LaserMappingNode::sync_packages(MeasureGroup & meas)
{
  if (lidar_buffer_.empty() || imu_buffer_.empty())
  {
    return false;
  }

  /*** push a lidar scan ***/
  if (!lidar_pushed_)
  {
    meas.lidar = lidar_buffer_.front();
    meas.lidar_beg_time = time_buffer_.front();
    if (meas.lidar->points.size() <= 1)  // time too little
    {
      lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
      std::cerr << "Too few input point cloud!\n";
    }
    else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime_)
    {
      lidar_end_time_ = meas.lidar_beg_time + lidar_mean_scantime_;
    }
    else
    {
      scan_num_++;
      lidar_end_time_ = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
      lidar_mean_scantime_ +=
        (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime_) / scan_num_;
    }

    meas.lidar_end_time = lidar_end_time_;

    lidar_pushed_ = true;
  }

  if (last_timestamp_imu_ < lidar_end_time_)
  {
    return false;
  }

  /*** push imu data, and pop from imu buffer ***/
  double imu_time = get_time_sec(imu_buffer_.front()->header.stamp);
  meas.imu.clear();
  while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_))
  {
    imu_time = get_time_sec(imu_buffer_.front()->header.stamp);
    if (imu_time > lidar_end_time_) break;
    meas.imu.push_back(imu_buffer_.front());
    imu_buffer_.pop_front();
  }

  lidar_buffer_.pop_front();
  time_buffer_.pop_front();
  lidar_pushed_ = false;
  return true;
}

void LaserMappingNode::timer_callback()
{
  if (sync_packages(measures_))
  {
    // First synchronized scan after a (re-)initial pose: (re-)initialize the
    // filter with the pose received on /initialpose.
    if (initial_pose_set_.load())
    {
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
          this->get_logger(), "Relocalizer not ready (no map loaded); using the RViz initial pose.");
      }

      // The map was loaded but the match was rejected: the click does not line up
      // with the map. Do NOT seed the EKF with the raw guess — keep localization
      // stopped (TF suppressed) until a new /initialpose re-arms this branch.
      if (!match_ok && localizer_ready_)
      {
        core_->set_localization_failed(true);
        initial_pose_set_.store(false);
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
      return;
    }

    // Localization mode does nothing until the operator's first /initialpose:
    // without a pose the filter has never been seeded (kf_ is null), so there is
    // no valid pose to publish. Hold here (no EKF, no TF) until one arrives.
    if (params_.enable_localization && !core_->is_initialized())
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
    // the map support or moved faster than a plausible robot, set the failure
    // flag and suppress the TF/odometry for this scan (the hold at the top of
    // this callback keeps it suppressed until a new /initialpose).
    if (!core_->localization_health_check(measures_.lidar_beg_time))
    {
      return;
    }

    /******* Publish odometry *******/
    publish_odometry();

    double t3 = omp_get_wtime();

    /*** add the feature points to map kdtree ***/
    if (params_.enable_map_incremental)
    {
      core_->map_incremental();
    }

    double t5 = omp_get_wtime();

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
        "[debug-timing] lidar_t=%.3f n_pts=%d | imu+undistort:%.2f fov_segment:%.2f "
        "downsample:%.2f prep:%.2f | ikdtree+eskf:%.2f | check+odom:%.2f "
        "map_incr:%.2f | publish:%.2f | total:%.2f ms",
        measures_.lidar_beg_time, core_->feats_down_size(),
        (timing.t_imu_end - t0) * 1e3, (timing.t_fov_end - timing.t_imu_end) * 1e3,
        (timing.t1 - timing.t_fov_end) * 1e3, (timing.t2 - timing.t1) * 1e3,
        (timing.t_update_end - timing.t_update_start) * 1e3, (t3 - timing.t_update_end) * 1e3,
        (t5 - t3) * 1e3, (t6 - t5) * 1e3, (t6 - t0) * 1e3);
    }

    /*** Debug variables ***/
    if (params_.runtime_pos_log_enable)
    {
      core_->log_scan_runtime(measures_, t0, t3, t5, timing);
    }
  }
}

// ---------------------------------------------------------------------------
// Publishing helpers
// ---------------------------------------------------------------------------

void LaserMappingNode::publish_odometry()
{
  odomAftMapped_.header.frame_id = params_.map_frame;
  odomAftMapped_.child_frame_id = params_.base_frame;
  odomAftMapped_.header.stamp = get_ros_time(lidar_end_time_);
  core_->set_posestamp(odomAftMapped_.pose);
  pub_odom_->publish(odomAftMapped_);

  auto P = core_->get_P();
  for (int i = 0; i < 6; i++)
  {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped_.pose.covariance[i * 6 + 0] = P(k, 3);
    odomAftMapped_.pose.covariance[i * 6 + 1] = P(k, 4);
    odomAftMapped_.pose.covariance[i * 6 + 2] = P(k, 5);
    odomAftMapped_.pose.covariance[i * 6 + 3] = P(k, 0);
    odomAftMapped_.pose.covariance[i * 6 + 4] = P(k, 1);
    odomAftMapped_.pose.covariance[i * 6 + 5] = P(k, 2);
  }

  geometry_msgs::msg::TransformStamped trans;
  trans.header.frame_id = params_.map_frame;
  trans.header.stamp = odomAftMapped_.header.stamp;
  trans.child_frame_id = params_.base_frame;
  trans.transform.translation.x = odomAftMapped_.pose.pose.position.x;
  trans.transform.translation.y = odomAftMapped_.pose.pose.position.y;
  trans.transform.translation.z = odomAftMapped_.pose.pose.position.z;
  trans.transform.rotation.w = odomAftMapped_.pose.pose.orientation.w;
  trans.transform.rotation.x = odomAftMapped_.pose.pose.orientation.x;
  trans.transform.rotation.y = odomAftMapped_.pose.pose.orientation.y;
  trans.transform.rotation.z = odomAftMapped_.pose.pose.orientation.z;
  tf_broadcaster_->sendTransform(trans);
}

void LaserMappingNode::publish_path()
{
  core_->set_posestamp(msg_body_pose_);
  msg_body_pose_.header.stamp = get_ros_time(lidar_end_time_);  // ros::Time().fromSec(lidar_end_time);
  msg_body_pose_.header.frame_id = params_.map_frame;

  /*** if path is too large, the rviz will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0)
  {
    path_.poses.push_back(msg_body_pose_);
    pub_path_->publish(path_);
  }
}

void LaserMappingNode::publish_frame_world()
{
  if (params_.scan_publish_en)
  {
    PointCloudXYZI::Ptr laserCloudFullRes(core_->feats_down_body());
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
      core_->RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time_);
    laserCloudmsg.header.frame_id = params_.map_frame;
    pub_laser_cloud_full_->publish(laserCloudmsg);
    publish_count_ -= PUBFRAME_PERIOD;
  }
}

void LaserMappingNode::publish_frame_body()
{
  int size = core_->feats_undistort()->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
  {
    core_->RGBpointBodyLidarToIMU(
      &core_->feats_undistort()->points[i], &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time_);
  laserCloudmsg.header.frame_id = params_.base_frame;
  pub_laser_cloud_full_body_->publish(laserCloudmsg);
  publish_count_ -= PUBFRAME_PERIOD;
}

void LaserMappingNode::publish_effect_world()
{
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(core_->effct_feat_num(), 1));
  for (int i = 0; i < core_->effct_feat_num(); i++)
  {
    core_->RGBpointBodyToWorld(&core_->laser_cloud_ori()->points[i], &laserCloudWorld->points[i]);
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time_);
  laserCloudFullRes3.header.frame_id = params_.map_frame;
  pub_laser_cloud_effect_->publish(laserCloudFullRes3);
}

void LaserMappingNode::publish_map()
{
  // Accumulate the current scan into the map-to-publish cloud (kept in the
  // core), downsample it, and publish.
  PointCloudXYZI::Ptr laserCloudFullResMap = core_->map_to_publish();

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudFullResMap, laserCloudmsg);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time_);
  laserCloudmsg.header.frame_id = params_.map_frame;
  pub_laser_cloud_map_->publish(laserCloudmsg);
}

void LaserMappingNode::save_to_pcd()
{
  pcl::PCDWriter pcd_writer;
  pcd_writer.writeBinary(params_.map_file_path, *core_->pcl_wait_pub());
}

void LaserMappingNode::map_publish_callback()
{
  if (params_.map_en)
  {
    publish_map();
  }
}

void LaserMappingNode::map_save_callback(
  const std_srvs::srv::Trigger::Request::SharedPtr req,
  std_srvs::srv::Trigger::Response::SharedPtr res)
{
  RCLCPP_INFO(this->get_logger(), "Saving map to %s...", params_.map_file_path.c_str());
  if (params_.pcd_save_en)
  {
    save_to_pcd();
    res->success = true;
    res->message = "Map saved.";
  }
  else
  {
    res->success = false;
    res->message = "Map save disabled.";
  }
}

}  // namespace fast_lio
