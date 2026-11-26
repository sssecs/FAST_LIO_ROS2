// Localization node: construction, parameters wiring, sensor callbacks and
// buffer synchronization. Per-scan processing lives in node/processing.cpp and
// publishing helpers in node/publishers.cpp.

#include <fast_lio/node/localization_node.hpp>

#include <functional>
#include <stdexcept>

#include <omp.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace fast_lio
{

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

LocalizationNode::LocalizationNode(const rclcpp::NodeOptions & options)
  : Node("fastlio_localization", options)
{
  preprocess_ = std::make_shared<Preprocess>();

  load_params();

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
    RCLCPP_INFO(
      this->get_logger(), "Static TF %s -> %s available.", params_.lidar_frame.c_str(),
      params_.base_frame.c_str());
  }
  catch (const tf2::TransformException & ex)
  {
    throw std::runtime_error(
            std::string("lidar -> base static TF not available at startup: ") + ex.what());
  }

  // --- algorithm core and prior map -------------------------------------------
  core_ = std::make_unique<LocalizationCore>(params_, this->get_logger());
  if (!params_.map_path.empty())
  {
    load_map();
  }
  else
  {
    RCLCPP_WARN(
      this->get_logger(), "No map_path configured; waiting for a map before localizing.");
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

  // Publish the loaded map once so RViz can display it before the first pose.
  if (map_)
  {
    publish_map();
  }

  auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
  timer_ = rclcpp::create_timer(
    this, this->get_clock(), period_ms, std::bind(&LocalizationNode::timer_callback, this));

  auto map_period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0));
  map_pub_timer_ = rclcpp::create_timer(
    this, this->get_clock(), map_period_ms, std::bind(&LocalizationNode::map_publish_callback, this));

  RCLCPP_INFO(
    this->get_logger(), "Node init finished. State: %s.", toString(state_.load()));
}

// ---------------------------------------------------------------------------
// Map loading
// ---------------------------------------------------------------------------

void LocalizationNode::load_map()
{
  RCLCPP_INFO(
    this->get_logger(), "Loading prior map from %s (leaf %.3f m)...", params_.map_path.c_str(),
    params_.leaf_size);
  auto map = std::make_shared<PriorMap>(params_.map_path, params_.leaf_size, params_.map_publish_leaf);
  RCLCPP_INFO(
    this->get_logger(), "Prior map ready: %zu raw -> %zu matching / %zu display points.",
    map->info().raw_points, map->info().map_points, map->info().publish_points);

  core_->set_map(map);

  if (params_.relocalization_en)
  {
    localizer_.setMap(map->cloudXyz());
    localizer_ready_ = true;
    RCLCPP_INFO(
      this->get_logger(), "Open3D relocalizer map ready: %zu points.", localizer_.mapPoints());
  }

  map_ = std::move(map);
  set_state(LocalizationState::kUnlocalized);
}

void LocalizationNode::set_state(LocalizationState state)
{
  const auto prev = state_.exchange(state);
  if (prev != state)
  {
    RCLCPP_INFO(this->get_logger(), "State: %s -> %s", toString(prev), toString(state));
  }
}

// ---------------------------------------------------------------------------
// Sensor callbacks
// ---------------------------------------------------------------------------

void LocalizationNode::standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr & msg)
{
  std::lock_guard<std::mutex> lock(mtx_buffer_);

  // Transform the scan from the lidar frame to the base frame. A failure here
  // must not kill the node: drop the frame and report it instead.
  sensor_msgs::msg::PointCloud2 cloud_transformed;
  try
  {
    tf2::doTransform(*msg, cloud_transformed, tf_lidar_to_base_);
  }
  catch (const tf2::TransformException & ex)
  {
    RCLCPP_ERROR(this->get_logger(), "LiDAR transform failed, dropping frame: %s", ex.what());
    return;
  }

  double cur_time = get_time_sec(msg->header.stamp);

  if (!is_first_lidar_ && cur_time < last_timestamp_lidar_)
  {
    RCLCPP_WARN(this->get_logger(), "lidar loop back, clear buffer");
    lidar_buffer_.clear();
  }
  is_first_lidar_ = false;

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  preprocess_->process(cloud_transformed, ptr);

  lidar_buffer_.push_back(ptr);
  time_buffer_.push_back(cur_time);
  last_timestamp_lidar_ = cur_time;
}

void LocalizationNode::imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr & msg_in)
{
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
      RCLCPP_WARN(this->get_logger(), "IMU loop back, clear buffer");
      imu_buffer_.clear();
    }
    last_timestamp_imu_ = timestamp;
    imu_buffer_.push_back(msg);
  }
}

void LocalizationNode::initial_pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
{
  if (state_.load() == LocalizationState::kNoMap)
  {
    RCLCPP_WARN(
      this->get_logger(), "Initial pose ignored: no map loaded (%s).", params_.initial_pose_topic.c_str());
    return;
  }
  init_pose_ = *msg;
  initial_pose_set_.store(true);
  RCLCPP_INFO(this->get_logger(), "Initial pose received and queued for relocalization.");
}

// ---------------------------------------------------------------------------
// Sensor synchronization
// ---------------------------------------------------------------------------

bool LocalizationNode::sync_packages(MeasureGroup & meas)
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
      RCLCPP_WARN(this->get_logger(), "Too few input points in scan.");
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

}  // namespace fast_lio
