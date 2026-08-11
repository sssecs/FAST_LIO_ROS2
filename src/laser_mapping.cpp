// Mid-360 localization node for FAST-LIO (implementation).
//
// This node loads a pre-built point-cloud map at startup and runs the FAST-LIO
// iterated-EKF against it once an initial pose is received on /initialpose.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#include <omp.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <fast_lio/laser_mapping.hpp>

// Length of the IMU-bias estimation window (s).
#define INIT_TIME (0.1)
// Measurement noise of a matched lidar point (m).
#define LASER_POINT_COV (0.001)
// How many lidar scans between each full-cloud publish.
#define PUBFRAME_PERIOD (20)

namespace
{
// FOV segment trigger threshold (relative to det_range).
constexpr float MOV_THRESHOLD = 1.5f;

// The esekfom library stores the measurement model as a raw function pointer,
// so the node instance is reached through this single file-local pointer.
fast_lio::LaserMappingNode * g_mapping_node = nullptr;
}  // namespace

namespace fast_lio
{

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

LaserMappingNode::LaserMappingNode(const rclcpp::NodeOptions & options)
  : Node("laser_mapping", options)
{
  g_mapping_node = this;

  preprocess_ = std::make_shared<Preprocess>();
  imu_process_ = std::make_shared<ImuProcess>();

  load_params();

  init_pose_.pose.pose.position.x = 0.0;
  init_pose_.pose.pose.position.y = 0.0;
  init_pose_.pose.pose.position.z = 0.0;

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

  // --- filters and map -------------------------------------------------------
  memset(point_selected_surf_, true, sizeof(point_selected_surf_));
  memset(res_last_, -1000.0f, sizeof(res_last_));
  downSizeFilterSurf_.setLeafSize(
    params_.filter_size_surf, params_.filter_size_surf, params_.filter_size_surf);
  downSizeFilterMap_.setLeafSize(
    params_.filter_size_map, params_.filter_size_map, params_.filter_size_map);

  if (params_.enable_localization)
  {
    load_map();
  }

  // --- debug logs ------------------------------------------------------------
  const std::string pos_log_dir = std::string(ROOT_DIR) + "/Log/pos_log.txt";
  fp_ = fopen(pos_log_dir.c_str(), "w");
  fout_pre_.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out_.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
  if (fout_pre_ && fout_out_)
  {
    std::cout << "~~~~" << ROOT_DIR << " file opened" << std::endl;
  }
  else
  {
    std::cout << "~~~~" << ROOT_DIR << " doesn't exist" << std::endl;
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
  fout_out_.close();
  fout_pre_.close();
  if (fp_)
  {
    fclose(fp_);
  }

  // Save the accumulated map (only populated when map incremental / pcd save is enabled).
  if (!pcl_wait_save_->empty() && params_.pcd_save_en)
  {
    const std::string file_name = "scans.pcd";
    const std::string all_points_dir = std::string(ROOT_DIR) + "PCD/" + file_name;
    pcl::PCDWriter pcd_writer;
    std::cout << "current scan saved to /PCD/" << file_name << std::endl;
    pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
  }

  // Write the runtime performance log.
  if (params_.runtime_pos_log_enable)
  {
    std::vector<double> t, s_vec, s_vec2, s_vec3, s_vec5;
    FILE * fp2 = nullptr;
    const std::string log_dir = std::string(ROOT_DIR) + "/Log/fast_lio_time_log.csv";
    fp2 = fopen(log_dir.c_str(), "w");
    fprintf(
      fp2,
      "time_stamp, total time, scan point size, incremental time, search time, delete size, delete "
      "time, tree size st, tree size end, add point size, preprocess time\n");
    for (int i = 0; i < time_log_counter_; i++)
    {
      fprintf(
        fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n", T1_[i], s_plot_[i],
        int(s_plot2_[i]), s_plot3_[i], s_plot4_[i], int(s_plot5_[i]), s_plot6_[i],
        int(s_plot7_[i]), int(s_plot8_[i]), int(s_plot10_[i]), s_plot11_[i]);
      t.push_back(T1_[i]);
      s_vec.push_back(s_plot9_[i]);
      s_vec2.push_back(s_plot3_[i] + s_plot6_[i]);
      s_vec3.push_back(s_plot4_[i]);
      s_vec5.push_back(s_plot_[i]);
    }
    fclose(fp2);
  }
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
  this->declare_parameter<bool>("publish.dense_publish_en", true);
  this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
  // pcd save
  this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
  this->declare_parameter<std::string>("map_file_path", "./test.pcd");
  // logging
  this->declare_parameter<bool>("runtime_pos_log_enable", false);
  // localization
  this->declare_parameter<bool>("localization.enable_localization", false);
  this->declare_parameter<bool>("localization.enable_map_incremental", true);
  this->declare_parameter<double>("localization.leaf_size", 0.05);
  this->declare_parameter<std::string>("localization.map_path", "");
  this->declare_parameter<double>("localization.init_z", 1.2);
  this->declare_parameter<double>("localization.tf_lookup_timeout", 5.0);
  this->declare_parameter<std::string>("localization.initial_pose_topic", "/initialpose");

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
  this->get_parameter("publish.dense_publish_en", params_.dense_publish_en);
  this->get_parameter("publish.scan_bodyframe_pub_en", params_.scan_bodyframe_pub_en);
  this->get_parameter("pcd_save.pcd_save_en", params_.pcd_save_en);
  this->get_parameter("map_file_path", params_.map_file_path);
  this->get_parameter("runtime_pos_log_enable", params_.runtime_pos_log_enable);
  this->get_parameter("localization.enable_localization", params_.enable_localization);
  this->get_parameter("localization.enable_map_incremental", params_.enable_map_incremental);
  this->get_parameter("localization.leaf_size", params_.leaf_size);
  this->get_parameter("localization.map_path", params_.map_path);
  this->get_parameter("localization.init_z", params_.init_z);
  this->get_parameter("localization.tf_lookup_timeout", params_.tf_lookup_timeout);
  this->get_parameter("localization.initial_pose_topic", params_.initial_pose_topic);

  num_max_iterations_ = params_.max_iteration;
  preprocess_->blind = params_.blind;
}

// ---------------------------------------------------------------------------
// Point transforms
// ---------------------------------------------------------------------------

void LaserMappingNode::point_body_to_world(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) + state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

template <typename T>
void LaserMappingNode::point_body_to_world(const Matrix<T, 3, 1> & pi, Matrix<T, 3, 1> & po)
{
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) + state_point_.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void LaserMappingNode::RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) + state_point_.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void LaserMappingNode::RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point_.offset_R_L_I * p_body_lidar + state_point_.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

template <typename T>
void LaserMappingNode::set_posestamp(T & out)
{
  out.pose.position.x = state_point_.pos(0);
  out.pose.position.y = state_point_.pos(1);
  out.pose.position.z = state_point_.pos(2);
  out.pose.orientation.x = geoQuat_.x;
  out.pose.orientation.y = geoQuat_.y;
  out.pose.orientation.z = geoQuat_.z;
  out.pose.orientation.w = geoQuat_.w;
}

void LaserMappingNode::points_cache_collect()
{
  PointVector points_history;
  ikdtree_.acquire_removed_points(points_history);
}

// ---------------------------------------------------------------------------
// Map FOV segmentation / incremental update
// ---------------------------------------------------------------------------

void LaserMappingNode::lasermap_fov_segment()
{
  cub_needrm_.clear();
  kdtree_delete_counter_ = 0;
  kdtree_delete_time_ = 0.0;
  point_body_to_world(XAxisPoint_body_, XAxisPoint_world_);
  V3D pos_LiD = pos_lid_;
  if (!Localmap_Initialized_)
  {
    for (int i = 0; i < 3; i++)
    {
      LocalMap_Points_.vertex_min[i] = pos_LiD(i) - params_.cube_side_length / 2.0;
      LocalMap_Points_.vertex_max[i] = pos_LiD(i) + params_.cube_side_length / 2.0;
    }
    Localmap_Initialized_ = true;
    return;
  }
  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++)
  {
    dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points_.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points_.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * params_.det_range ||
      dist_to_map_edge[i][1] <= MOV_THRESHOLD * params_.det_range)
    {
      need_move = true;
    }
  }
  if (!need_move) return;

  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points_;
  float mov_dist = std::max(
    (params_.cube_side_length - 2.0 * MOV_THRESHOLD * params_.det_range) * 0.5 * 0.9,
    double(params_.det_range * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++)
  {
    tmp_boxpoints = LocalMap_Points_;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * params_.det_range)
    {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points_.vertex_max[i] - mov_dist;
      cub_needrm_.push_back(tmp_boxpoints);
    }
    else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * params_.det_range)
    {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points_.vertex_min[i] + mov_dist;
      cub_needrm_.push_back(tmp_boxpoints);
    }
  }
  LocalMap_Points_ = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (cub_needrm_.size() > 0)
  {
    kdtree_delete_counter_ = ikdtree_.Delete_Point_Boxes(cub_needrm_);
  }
  kdtree_delete_time_ = omp_get_wtime() - delete_begin;
}

void LaserMappingNode::map_incremental()
{
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size_);
  PointNoNeedDownsample.reserve(feats_down_size_);
  for (int i = 0; i < feats_down_size_; i++)
  {
    /* transform to world frame */
    point_body_to_world(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
    /* decide if need add to map */
    if (!Nearest_Points_[i].empty() && flg_EKF_inited_)
    {
      const PointVector & points_near = Nearest_Points_[i];
      bool need_add = true;
      BoxPointType Box_of_Point;
      PointType downsample_result, mid_point;
      mid_point.x =
        floor(feats_down_world_->points[i].x / params_.filter_size_map) * params_.filter_size_map +
        0.5 * params_.filter_size_map;
      mid_point.y =
        floor(feats_down_world_->points[i].y / params_.filter_size_map) * params_.filter_size_map +
        0.5 * params_.filter_size_map;
      mid_point.z =
        floor(feats_down_world_->points[i].z / params_.filter_size_map) * params_.filter_size_map +
        0.5 * params_.filter_size_map;
      float dist = calc_dist(feats_down_world_->points[i], mid_point);
      if (fabs(points_near[0].x - mid_point.x) > 0.5 * params_.filter_size_map &&
        fabs(points_near[0].y - mid_point.y) > 0.5 * params_.filter_size_map &&
        fabs(points_near[0].z - mid_point.z) > 0.5 * params_.filter_size_map)
      {
        PointNoNeedDownsample.push_back(feats_down_world_->points[i]);
        continue;
      }
      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
      {
        if (points_near.size() < NUM_MATCH_POINTS) break;
        if (calc_dist(points_near[readd_i], mid_point) < dist)
        {
          need_add = false;
          break;
        }
      }
      if (need_add) PointToAdd.push_back(feats_down_world_->points[i]);
    }
    else
    {
      PointToAdd.push_back(feats_down_world_->points[i]);
    }
  }

  double st_time = omp_get_wtime();
  add_point_size_ = ikdtree_.Add_Points(PointToAdd, true);
  ikdtree_.Add_Points(PointNoNeedDownsample, false);
  add_point_size_ = PointToAdd.size() + PointNoNeedDownsample.size();
  kdtree_incremental_time_ = omp_get_wtime() - st_time;
}

void LaserMappingNode::load_map()
{
  RCLCPP_INFO(this->get_logger(), "Loading GlobalMap from %s ...", params_.map_path.c_str());

  PointCloudXYZI::Ptr localization_map(new PointCloudXYZI());
  if (pcl::io::loadPCDFile(params_.map_path, *localization_map) < 0)
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to load map file: %s", params_.map_path.c_str());
    throw std::runtime_error("localization map file not found: " + params_.map_path);
  }

  // Downsample the loaded map once; the scan matching uses this resolution.
  PointCloudXYZI::Ptr map_ds(new PointCloudXYZI());
  pcl::VoxelGrid<PointType> voxel_grid_filter;
  voxel_grid_filter.setLeafSize(params_.leaf_size, params_.leaf_size, params_.leaf_size);
  voxel_grid_filter.setInputCloud(localization_map);
  voxel_grid_filter.filter(*map_ds);
  RCLCPP_INFO(this->get_logger(), "GlobalMap ds point num: %zu", map_ds->size());

  PointVector points_to_add;
  points_to_add.reserve(map_ds->points.size());
  for (size_t i = 0; i < map_ds->points.size(); i++)
  {
    points_to_add.push_back(map_ds->points[i]);
  }

  downSizeFilterSurf_.setInputCloud(map_ds);
  downSizeFilterSurf_.filter(*feats_down_body_);
  feats_down_size_ = feats_down_body_->points.size();

  if (ikdtree_.Root_Node == nullptr)
  {
    RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
    if (feats_down_size_ > 5)
    {
      ikdtree_.set_downsample_param(params_.filter_size_map);
      feats_down_world_->resize(feats_down_size_);
      for (int i = 0; i < feats_down_size_; i++)
      {
        point_body_to_world(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
      }
      ikdtree_.Build(feats_down_world_->points);
    }
    return;
  }

  ikdtree_.Add_Points(points_to_add, false);  // load whole map.
  int featsFromMapNum = ikdtree_.validnum();
  kdtree_size_st_ = ikdtree_.size();

  RCLCPP_INFO(this->get_logger(), "Loaded GlobalMap from %s!", params_.map_path.c_str());
}

void LaserMappingNode::set_initial_localization(const geometry_msgs::msg::PoseWithCovarianceStamped & input_init)
{
  const auto & pos = input_init.pose.pose.position;
  Eigen::Vector3d p_map(pos.x, pos.y, pos.z);

  const auto & ori = input_init.pose.pose.orientation;
  Eigen::Quaterniond q_map(ori.w, ori.x, ori.y, ori.z);

  p_map[2] = params_.init_z;

  RCLCPP_INFO(
    this->get_logger(), "Init pose: pos=(%.3f, %.3f, %.3f), quat=(%.4f, %.4f, %.4f, %.4f)",
    p_map.x(), p_map.y(), p_map.z(), q_map.x(), q_map.y(), q_map.z(), q_map.w());

  auto init_state = kf_->get_x();
  init_state.rot = MTK::SO3(q_map);
  init_state.pos = p_map;
  kf_->change_x(init_state);
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
  s_plot11_[scan_count_] = omp_get_wtime() - preprocess_start_time;
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
      imu_process_->Reset();
      Lidar_T_wrt_IMU_ << VEC_FROM_ARRAY(params_.extrin_T);
      Lidar_R_wrt_IMU_ << MAT_FROM_ARRAY(params_.extrin_R);
      imu_process_->set_extrinsic(Lidar_T_wrt_IMU_, Lidar_R_wrt_IMU_);
      imu_process_->set_gyr_cov(V3D(params_.gyr_cov, params_.gyr_cov, params_.gyr_cov));
      imu_process_->set_acc_cov(V3D(params_.acc_cov, params_.acc_cov, params_.acc_cov));
      imu_process_->set_gyr_bias_cov(V3D(params_.b_gyr_cov, params_.b_gyr_cov, params_.b_gyr_cov));
      imu_process_->set_acc_bias_cov(V3D(params_.b_acc_cov, params_.b_acc_cov, params_.b_acc_cov));

      kf_ = std::make_unique<esekfom::esekf<state_ikfom, 12, input_ikfom>>();
      fill(epsi_, epsi_ + 23, 0.001);
      kf_->init_dyn_share(get_f, df_dx, df_dw, &LaserMappingNode::h_share_model, num_max_iterations_, epsi_);
      this->set_initial_localization(init_pose_);
      first_lidar_time_ = measures_.lidar_beg_time;
      imu_process_->first_lidar_time = first_lidar_time_;
      initial_pose_set_.store(false);
      return;
    }

    double t0, t1, t2, t3, t4, t5, match_start, solve_start, svd_time;

    match_time_ = 0;
    kdtree_search_time_ = 0.0;
    solve_time_ = 0;
    solve_const_H_time_ = 0;
    svd_time = 0;
    t0 = omp_get_wtime();

    imu_process_->Process(measures_, *kf_, feats_undistort_);
    state_point_ = kf_->get_x();
    pos_lid_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;

    if (feats_undistort_->empty() || (feats_undistort_ == NULL))
    {
      RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
      return;
    }

    flg_EKF_inited_ =
      (measures_.lidar_beg_time - first_lidar_time_) < INIT_TIME ? false : true;
    /*** Segment the map in lidar FOV ***/
    if (params_.enable_map_incremental)
    {
      lasermap_fov_segment();
    }

    /*** downsample the feature points in a scan ***/
    downSizeFilterSurf_.setInputCloud(feats_undistort_);
    downSizeFilterSurf_.filter(*feats_down_body_);
    t1 = omp_get_wtime();
    feats_down_size_ = feats_down_body_->points.size();
    /*** initialize the map kdtree ***/
    if (ikdtree_.Root_Node == nullptr)
    {
      RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
      if (feats_down_size_ > 5)
      {
        ikdtree_.set_downsample_param(params_.filter_size_map);
        feats_down_world_->resize(feats_down_size_);
        for (int i = 0; i < feats_down_size_; i++)
        {
          point_body_to_world(&(feats_down_body_->points[i]), &(feats_down_world_->points[i]));
        }
        ikdtree_.Build(feats_down_world_->points);
      }
      return;
    }
    int featsFromMapNum = ikdtree_.validnum();
    kdtree_size_st_ = ikdtree_.size();

    /*** ICP and iterated Kalman filter update ***/
    if (feats_down_size_ < 5)
    {
      RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
      return;
    }

    normvec_->resize(feats_down_size_);
    feats_down_world_->resize(feats_down_size_);

    V3D ext_euler = SO3ToEuler(state_point_.offset_R_L_I);
    fout_pre_ << setw(20) << measures_.lidar_beg_time - first_lidar_time_ << " "
              << euler_cur_.transpose() << " " << state_point_.pos.transpose() << " "
              << ext_euler.transpose() << " " << state_point_.offset_T_L_I.transpose() << " "
              << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
              << state_point_.ba.transpose() << " " << state_point_.grav << endl;

    pointSearchInd_surf_.resize(feats_down_size_);
    Nearest_Points_.resize(feats_down_size_);
    int rematch_num = 0;
    bool nearest_search_en = true;  //

    t2 = omp_get_wtime();

    /*** iterated state estimation ***/
    double t_update_start = omp_get_wtime();
    double solve_H_time = 0;
    kf_->update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
    state_point_ = kf_->get_x();
    euler_cur_ = SO3ToEuler(state_point_.rot);
    pos_lid_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I;
    geoQuat_.x = state_point_.rot.coeffs()[0];
    geoQuat_.y = state_point_.rot.coeffs()[1];
    geoQuat_.z = state_point_.rot.coeffs()[2];
    geoQuat_.w = state_point_.rot.coeffs()[3];

    double t_update_end = omp_get_wtime();

    /******* Publish odometry *******/
    publish_odometry();

    /*** add the feature points to map kdtree ***/
    t3 = omp_get_wtime();
    if (params_.enable_map_incremental)
    {
      map_incremental();
    }
    t5 = omp_get_wtime();

    /******* Publish points *******/
    if (params_.path_en) publish_path();
    if (params_.scan_publish_en) publish_frame_world();
    if (params_.scan_publish_en && params_.scan_bodyframe_pub_en) publish_frame_body();
    if (params_.effect_map_en) publish_effect_world();

    /*** Debug variables ***/
    if (params_.runtime_pos_log_enable)
    {
      frame_num_++;
      kdtree_size_end_ = ikdtree_.size();
      aver_time_consu_ =
        aver_time_consu_ * (frame_num_ - 1) / frame_num_ + (t5 - t0) / frame_num_;
      aver_time_icp_ =
        aver_time_icp_ * (frame_num_ - 1) / frame_num_ + (t_update_end - t_update_start) / frame_num_;
      aver_time_match_ =
        aver_time_match_ * (frame_num_ - 1) / frame_num_ + (match_time_) / frame_num_;
      aver_time_incre_ =
        aver_time_incre_ * (frame_num_ - 1) / frame_num_ + (kdtree_incremental_time_) / frame_num_;
      aver_time_solve_ =
        aver_time_solve_ * (frame_num_ - 1) / frame_num_ + (solve_time_ + solve_H_time) / frame_num_;
      aver_time_const_H_time_ =
        aver_time_const_H_time_ * (frame_num_ - 1) / frame_num_ + solve_time_ / frame_num_;
      T1_[time_log_counter_] = measures_.lidar_beg_time;
      s_plot_[time_log_counter_] = t5 - t0;
      s_plot2_[time_log_counter_] = feats_undistort_->points.size();
      s_plot3_[time_log_counter_] = kdtree_incremental_time_;
      s_plot4_[time_log_counter_] = kdtree_search_time_;
      s_plot5_[time_log_counter_] = kdtree_delete_counter_;
      s_plot6_[time_log_counter_] = kdtree_delete_time_;
      s_plot7_[time_log_counter_] = kdtree_size_st_;
      s_plot8_[time_log_counter_] = kdtree_size_end_;
      s_plot9_[time_log_counter_] = aver_time_consu_;
      s_plot10_[time_log_counter_] = add_point_size_;
      time_log_counter_++;
      printf(
        "[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f "
        " ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",
        t1 - t0, aver_time_match_, aver_time_solve_, t3 - t1, t5 - t3, aver_time_consu_,
        aver_time_icp_, aver_time_const_H_time_);
      ext_euler = SO3ToEuler(state_point_.offset_R_L_I);
      fout_out_ << setw(20) << measures_.lidar_beg_time - first_lidar_time_ << " "
                << euler_cur_.transpose() << " " << state_point_.pos.transpose() << " "
                << ext_euler.transpose() << " " << state_point_.offset_T_L_I.transpose() << " "
                << state_point_.vel.transpose() << " " << state_point_.bg.transpose() << " "
                << state_point_.ba.transpose() << " " << state_point_.grav << " "
                << feats_undistort_->points.size() << endl;
      dump_lio_state_to_log(fp_);
    }
  }
}

// ---------------------------------------------------------------------------
// Measurement model (called by the esekfom update through the trampoline)
// ---------------------------------------------------------------------------

void LaserMappingNode::h_share_model(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data)
{
  g_mapping_node->h_share_model_impl(s, ekfom_data);
}

void LaserMappingNode::h_share_model_impl(state_ikfom & s, esekfom::dyn_share_datastruct<double> & ekfom_data)
{
  double match_start = omp_get_wtime();
  laserCloudOri_->clear();
  corr_normvect_->clear();
  total_residual_ = 0.0;

  /** closest surface search and residual computation **/
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < feats_down_size_; i++)
  {
    PointType & point_body = feats_down_body_->points[i];
    PointType & point_world = feats_down_world_->points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;

    vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto & points_near = Nearest_Points_[i];

    if (ekfom_data.converge)
    {
      /** Find the closest surfaces in the map **/
      ikdtree_.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
      point_selected_surf_[i] =
        points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
    }

    if (!point_selected_surf_[i]) continue;

    VF(4) pabcd;
    point_selected_surf_[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f))
    {
      float pd2 =
        pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9)
      {
        point_selected_surf_[i] = true;
        normvec_->points[i].x = pabcd(0);
        normvec_->points[i].y = pabcd(1);
        normvec_->points[i].z = pabcd(2);
        normvec_->points[i].intensity = pd2;
        res_last_[i] = abs(pd2);
      }
    }
  }

  effct_feat_num_ = 0;

  for (int i = 0; i < feats_down_size_; i++)
  {
    if (point_selected_surf_[i])
    {
      laserCloudOri_->points[effct_feat_num_] = feats_down_body_->points[i];
      corr_normvect_->points[effct_feat_num_] = normvec_->points[i];
      total_residual_ += res_last_[i];
      effct_feat_num_++;
    }
  }

  if (effct_feat_num_ < 1)
  {
    ekfom_data.valid = false;
    std::cerr << "No Effective Points!" << std::endl;
    return;
  }

  res_mean_last_ = total_residual_ / effct_feat_num_;
  match_time_ += omp_get_wtime() - match_start;
  double solve_start_ = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = MatrixXd::Zero(effct_feat_num_, 12);  // 23
  ekfom_data.h.resize(effct_feat_num_);

  for (int i = 0; i < effct_feat_num_; i++)
  {
    const PointType & laser_p = laserCloudOri_->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType & norm_p = corr_normvect_->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (params_.extrinsic_est_en)
    {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A),
        VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    }
    else
    {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
  solve_time_ += omp_get_wtime() - solve_start_;
}

// ---------------------------------------------------------------------------
// Publishing helpers
// ---------------------------------------------------------------------------

void LaserMappingNode::publish_odometry()
{
  odomAftMapped_.header.frame_id = params_.map_frame;
  odomAftMapped_.child_frame_id = params_.base_frame;
  odomAftMapped_.header.stamp = get_ros_time(lidar_end_time_);
  set_posestamp(odomAftMapped_.pose);
  pub_odom_->publish(odomAftMapped_);

  auto P = kf_->get_P();
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
  set_posestamp(msg_body_pose_);
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
    PointCloudXYZI::Ptr laserCloudFullRes(params_.dense_publish_en ? feats_undistort_ : feats_down_body_);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
      RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
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
  int size = feats_undistort_->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
  {
    RGBpointBodyLidarToIMU(&feats_undistort_->points[i], &laserCloudIMUBody->points[i]);
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
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num_, 1));
  for (int i = 0; i < effct_feat_num_; i++)
  {
    RGBpointBodyToWorld(&laserCloudOri_->points[i], &laserCloudWorld->points[i]);
  }
  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time_);
  laserCloudFullRes3.header.frame_id = params_.map_frame;
  pub_laser_cloud_effect_->publish(laserCloudFullRes3);
}

void LaserMappingNode::publish_map()
{
  PointCloudXYZI::Ptr laserCloudFullRes(params_.dense_publish_en ? feats_undistort_ : feats_down_body_);
  int size = laserCloudFullRes->points.size();
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++)
  {
    RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
  }
  *pcl_wait_pub_ += *laserCloudWorld;

  // Downsample the accumulated map before publishing to RViz to keep it
  // responsive. The full-resolution map in pcl_wait_pub_ is kept intact for
  // save_to_pcd().
  PointCloudXYZI::Ptr laserCloudFullResMap(new PointCloudXYZI());
  downSizeFilterMap_.setInputCloud(pcl_wait_pub_);
  downSizeFilterMap_.filter(*laserCloudFullResMap);

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudFullResMap, laserCloudmsg);
  laserCloudmsg.header.stamp = get_ros_time(lidar_end_time_);
  laserCloudmsg.header.frame_id = params_.map_frame;
  pub_laser_cloud_map_->publish(laserCloudmsg);
}

void LaserMappingNode::save_to_pcd()
{
  pcl::PCDWriter pcd_writer;
  pcd_writer.writeBinary(params_.map_file_path, *pcl_wait_pub_);
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

void LaserMappingNode::dump_lio_state_to_log(FILE * fp)
{
  V3D rot_ang(Log(state_point_.rot.toRotationMatrix()));
  fprintf(fp, "%lf ", measures_.lidar_beg_time - first_lidar_time_);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                      // Angle
  fprintf(fp, "%lf %lf %lf ", state_point_.pos(0), state_point_.pos(1), state_point_.pos(2));  // Pos
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                           // omega
  fprintf(fp, "%lf %lf %lf ", state_point_.vel(0), state_point_.vel(1), state_point_.vel(2));  // Vel
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                           // Acc
  fprintf(fp, "%lf %lf %lf ", state_point_.bg(0), state_point_.bg(1), state_point_.bg(2));  // Bias_g
  fprintf(fp, "%lf %lf %lf ", state_point_.ba(0), state_point_.ba(1), state_point_.ba(2));  // Bias_a
  fprintf(fp, "%lf %lf %lf ", state_point_.grav[0], state_point_.grav[1], state_point_.grav[2]);
  fprintf(fp, "\r\n");
  fflush(fp);
}

}  // namespace fast_lio
